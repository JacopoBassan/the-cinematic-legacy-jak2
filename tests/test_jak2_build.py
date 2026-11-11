"""Fail-closed compatibility and portable source-closure checks."""
import copy
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builder'))
from jak2 import build_release as builder
from jak2 import cinematic_eyes as eyes


class CinematicEyeBlinkTests(unittest.TestCase):
    @staticmethod
    def native_closure(lid, blink):
        # Native render-eyes converts negative encoded lids, then interpolates
        # from their resting closure to the closed endpoint using blink.
        return (1.0 + lid) + (-lid) * blink if lid < 0.0 else lid

    def test_all_authored_lid_bytes_keep_resting_expression_and_full_blink(self):
        for byte in range(-128, 128):
            with self.subTest(byte=byte):
                authored = byte / 128.0
                encoded = eyes.automatic_blink_lid(authored)
                expected_rest = 1.0 + authored if byte < 0 else authored
                self.assertEqual(self.native_closure(encoded, 0.0), expected_rest)
                self.assertEqual(self.native_closure(encoded, 1.0), 1.0)
                self.assertEqual(self.native_closure(encoded, .375),
                                 expected_rest + (1.0 - expected_rest) * .375)
                self.assertEqual(eyes.automatic_blink_lid(encoded), encoded)
        self.assertEqual(eyes.automatic_blink_lid(-115 / 128), -115 / 128)
        self.assertEqual(eyes.automatic_blink_lid(76 / 128), -52 / 128)
        self.assertEqual(eyes.automatic_blink_lid(0.0), -1.0)
        self.assertEqual(eyes.automatic_blink_lid(1.0), 1.0)
        for invalid in (-1.01, 1.01, float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                eyes.automatic_blink_lid(invalid)

    def test_one_frame_decode_keeps_side_specific_lids_and_packed_controls(self):
        frames = tuple(int.from_bytes(bytes((13, 27, lid & 255, 37, 43, 47, 64, 53)), 'little')
                       for lid in (-102, 76))
        before = tuple(frames)
        authored, encoded = eyes.one_frame_blink_lids(frames)
        self.assertEqual(authored, [-102 / 128, 76 / 128])
        self.assertEqual(encoded, [-102 / 128, -52 / 128])
        self.assertEqual(frames, before)
        with self.assertRaises(ValueError):
            eyes.one_frame_blink_lids(frames * 2)

    def test_source_generation_preserves_authored_idle_lids_without_game_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, directory, data = root / 'work', root / 'work/eyes', root / 'data'
            directory.mkdir(parents=True)
            (work / 'variant-audit.json').write_text(json.dumps({'modelInventory': {
                'sig-lod0': [{'file': 'atollext.fr3'}],
                'ashelin-lod0': [{'file': 'atollext.fr3'}]}}))
            native = data / 'goal_src/jak2/engine/process-drawable/process-drawable.gc'
            native.parent.mkdir(parents=True)
            native.write_text('''(in-package goal)
    (if (logtest? (-> this skel status) (joint-control-status eye-anim-valid eye-anim))
        (merc-eye-anim this)
        )''')
            def texture(name, index):
                return dict(name=name, width=64, height=64, pixels=b'\0' * 16384,
                            combo=65536 + index, page='synthetic-test-page')
            texture_rows = [texture(name, i) for i, name in enumerate(
                ('sig-eye', 'sig-eyelid', 'ashelin-eye', 'ashelin-eyelid'))]
            modules = {
                'fr3_models': types.SimpleNamespace(Reader=None, decompress=lambda _: b'', compress=None),
                'inspect_transformations': types.SimpleNamespace(textures=lambda _: (texture_rows, 0)),
                'npc_builder': types.SimpleNamespace(texture_bytes=None),
            }
            def eye_data(path):
                sig = path.name.startswith('sig-')
                lids = (-102, -115) if sig else (76, 76)
                words = tuple(int.from_bytes(bytes((13, 27, lid & 255, 37, 43, 47, 64, 53)), 'little')
                              for lid in lids)
                return (104 if sig else 100), 6, [0] * 60, words, 'synthetic-idle'
            output = directory / 'cinematic-eyes.gc'
            with patch.dict(sys.modules, modules), patch.object(eyes, 'eye_data', eye_data):
                report = eyes.generate(data, work, directory, output, ['sig', 'ashelin'], source_only=True)
            generated = output.read_text()
            self.assertIn('(set! (-> ctrl left lid) -0.796875)', generated)
            self.assertIn('(set! (-> ctrl right lid) -0.8984375)', generated)
            self.assertEqual(generated.count('(set! (-> ctrl left lid) -0.40625)'), 1)
            self.assertEqual(generated.count('(set! (-> ctrl right lid) -0.40625)'), 1)
            self.assertNotIn('(set! (-> ctrl left lid) -1.0)', generated)
            self.assertEqual(report[0]['authoredRestingLids'], [-102 / 128, -115 / 128])
            self.assertEqual(report[1]['automaticBlinkLids'], [-52 / 128, -52 / 128])
            self.assertTrue(all(row['autoBlink'] for row in report))


class PortableBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.disc = self.data / 'iso_data/jak2'
        self.disc.mkdir(parents=True)
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        (self.disc / 'buildinfo.json').write_text('[{"serial":"SCES-51608"}]')
        (self.disc / 'GAME.CGO').write_bytes(b'original-disc-archive')
        (self.tools / 'gk.exe').write_bytes(b'pinned-tool')
        self.config = dict(game='jak2', openGoalVersion='v0.3.8',
                           supportedSerials=['SCES-51608', 'SCUS-97265'],
                           regionalProfiles={'SCES-51608': {}, 'SCUS-97265': {}},
                           tools={'gk.exe': builder.sha(self.tools / 'gk.exe')},
                           inputs={'iso_data/jak2/GAME.CGO': builder.sha(self.disc / 'GAME.CGO')},
                           sourceTrees={}, fr3Inputs={})

    def tearDown(self):
        self.temporary.cleanup()

    def test_supported_profile_verifies_original_files_without_writes(self):
        before = {p: p.read_bytes() for p in self.data.rglob('*') if p.is_file()}
        result = builder.preflight(self.data, self.tools, self.config)
        self.assertEqual(result['selectedSerial'], 'SCES-51608')
        self.assertEqual(before, {p: p.read_bytes() for p in self.data.rglob('*') if p.is_file()})

    def test_modified_disc_and_tools_are_rejected(self):
        (self.disc / 'GAME.CGO').write_bytes(b'wrong-revision')
        with self.assertRaisesRegex(ValueError, 'Unsupported or modified'):
            builder.preflight(self.data, self.tools, self.config)
        (self.disc / 'GAME.CGO').write_bytes(b'original-disc-archive')
        (self.tools / 'gk.exe').write_bytes(b'new-unvalidated-runtime')
        with self.assertRaisesRegex(ValueError, 'Unsupported or modified'):
            builder.preflight(self.data, self.tools, self.config)

    def test_unknown_and_malformed_metadata_are_rejected(self):
        for value in ('[{"serial":"SCUS-00000"}]', '[]', '[{},{}]'):
            (self.disc / 'buildinfo.json').write_text(value)
            with self.assertRaises(ValueError):
                builder.regional_config(self.data, self.config)

    def test_verified_folder_can_select_usa_profile_without_metadata(self):
        (self.disc / 'buildinfo.json').unlink()
        (self.disc / 'SYSTEM.CNF').write_text('BOOT2 = cdrom0:\\SCUS_972.65;1\n')
        self.assertEqual(builder.regional_config(self.data, self.config)['selectedSerial'], 'SCUS-97265')

    def test_regional_inputs_cannot_mutate_shared_definition(self):
        config = copy.deepcopy(self.config)
        config['regionalProfiles']['SCUS-97265']['inputs'] = {'region-only': 'abc'}
        (self.disc / 'buildinfo.json').write_text('[{"serial":"SCUS-97265"}]')
        selected = builder.regional_config(self.data, config)
        self.assertEqual(selected['inputs']['region-only'], 'abc')
        self.assertNotIn('region-only', config['inputs'])

    def test_work_disc_metadata_does_not_mutate_original(self):
        selected = builder.regional_config(self.data, self.config)
        selected['regionalProfiles']['SCES-51608']['elfHash'] = 123456
        work = self.root / 'work'
        work.mkdir()
        original = (self.disc / 'buildinfo.json').read_bytes()
        staged = builder.prepare_disc(self.disc, work, selected)
        self.assertEqual(json.loads((staged / 'buildinfo.json').read_text()),
                         [{'serial': 'SCES-51608', 'elf_hash': 123456}])
        self.assertEqual((self.disc / 'buildinfo.json').read_bytes(), original)
        self.assertEqual((staged / 'GAME.CGO').read_bytes(), (self.disc / 'GAME.CGO').read_bytes())

    def test_metadata_hash_disagreement_is_rejected(self):
        self.config['regionalProfiles']['SCES-51608']['elfHash'] = 123456
        with self.assertRaisesRegex(ValueError, 'executable metadata'):
            builder.regional_config(self.data, self.config)

    def test_destinations_reject_installation_aliases_and_overlaps(self):
        for work, output in ((self.data / 'work', self.root / 'payload'),
                             (self.root, self.root / 'payload'),
                             (self.root / 'work', self.root / 'work/payload')):
            with self.assertRaises(ValueError):
                builder.destinations(self.data, work, output)

    def test_graphics_fingerprint_covers_assets_while_ignoring_unused_padding(self):
        import zstandard
        name = b'example-lod0'
        # Empty texture table, native unused Hfrag bytes, one Merc record.
        def raw(padding, model):
            return b'+\0' + struct.pack('<QQQ', 0, 0, 0) + padding + struct.pack('<QQ', 1, len(name)) + name + model + b'+\0'
        path = self.root / 'source.fr3'
        def fingerprint(padding, model):
            value = raw(padding, model)
            path.write_bytes(struct.pack('<Q', len(value)) + zstandard.ZstdCompressor(level=1).compress(value))
            return builder.fr3_asset_hash(path, name.decode())
        self.assertEqual(fingerprint(b'padding-A', b'geometry'), fingerprint(b'padding-B', b'geometry'))
        self.assertNotEqual(fingerprint(b'padding-A', b'geometry'), fingerprint(b'padding-A', b'changed!'))

    def test_source_closure_runs_from_clean_directory_without_study_inputs(self):
        config = json.loads((builder.HERE / 'source-closure.json').read_text())
        package = self.root / 'package'
        package.mkdir()
        for relative in config['files']:
            source = builder.ROOT / relative
            self.assertTrue(source.is_file(), relative)
            self.assertNotIn('_local', relative)
            self.assertNotIn(source.suffix.lower(), ('.fr3', '.glb', '.npy', '.cgo', '.dgo', '.go', '.gc'))
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        env = dict(os.environ, PYTHONPATH=str(package))
        result = subprocess.run([sys.executable, '-m', 'jak2.build_release', '--help'],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--data', result.stdout)

    def test_isolated_portable_stage_imports_its_declared_sibling(self):
        package = self.root / 'public source only'
        files = json.loads((builder.HERE / 'source-closure.json').read_text())['files']
        for relative in files:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(builder.ROOT / relative, destination)
        # This same-named module must never replace the Jak II eye adapter.
        (package / 'eye_alignment.py').write_text('raise AssertionError("Wrong eye adapter")\n')
        python = os.environ.get('CINEMATIC_PORTABLE_PYTHON', sys.executable)
        with patch.object(builder, 'HERE', package / 'jak2'):
            command = builder.stage_command('build_faces.py', '--data', self.data,
                                            '--work', self.root / 'missing work with spaces')
        # Exclude the portable interpreter's installed sources, so only the
        # copied, declared public closure can supply builder dependencies.
        isolated = ('import sys;from pathlib import Path;'
                    'blocked=Path(sys.executable).resolve().parents[1]/"source";'
                    'sys.path[:]=[p for p in sys.path if not Path(p).resolve().is_relative_to(blocked)];'
                    'sys.path[:0]=[str(Path(sys.argv[1]).parent.parent),'
                    'str(Path(sys.argv[1]).parent.parent/"stages")];')
        env = dict(os.environ, PYTHONPATH=str(package))
        result = subprocess.run([python, '-I', '-B', '-c', isolated + command[2],
                                 *map(str, command[3:])], cwd=self.root, env=env,
                                capture_output=True, text=True)
        # The real stage must finish its function-local imports and reach its
        # first unavailable game input. The previous direct-script launcher
        # instead raised ModuleNotFoundError for eye_alignment at this point.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)
        self.assertIn('GAME-materials.fr3', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertNotIn('Wrong eye adapter', result.stderr)
        self.assertFalse((self.root / 'missing work with spaces').exists())

    def test_all_static_and_function_local_imports_are_in_declared_closure(self):
        files = json.loads((builder.HERE / 'source-closure.json').read_text())['files']
        local = {Path(name).stem for name in files if name.endswith('.py')} | {'jak2'}
        allowed = set(sys.stdlib_module_names) | local | {'numpy', 'scipy', 'PIL', 'zstandard', '__future__'}
        for relative in files:
            if not relative.endswith('.py'):
                continue
            for node in ast.walk(ast.parse((builder.ROOT / relative).read_bytes())):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for name in names:
                    self.assertIn(name.split('.')[0], allowed, f'{relative}: undeclared dependency {name}')

    def test_context_preserves_ordered_native_index_pixels_palettes_and_provenance(self):
        from jak2.restore_context import restore, indexed_textures
        import zstandard
        def string(value):
            encoded = value.encode()
            return struct.pack('<Q', len(encoded)) + encoded
        def record(name, palette, levels):
            return (struct.pack('<HHIQ', 2, 2, 123, 4) + b'\0\1\2\3'
                    + bytes([palette]) * 1024 + string(name) + string('native-page')
                    + struct.pack('<Q', len(levels)) + b''.join(map(string, levels)))
        native_records = [record('jakbsmall-eyebrow', 11, ['GAME', 'PRI']),
                          record('skin', 17, ['GAME'])]
        donor = record('donor-extra', 19, ['donor'])
        header = b'+\0' + struct.pack('<QQ', 0, 0)
        tail = b'geometry-and-native-morph-sentinel'
        def save(path, records):
            raw = header + struct.pack('<Q', len(records)) + b''.join(records) + tail
            path.write_bytes(struct.pack('<Q', len(raw)) + zstandard.ZstdCompressor().compress(raw))
        native, candidate, output = [self.root / x for x in ('native.fr3', 'candidate.fr3', 'output.fr3')]
        save(native, native_records)
        save(candidate, [record('jakbsmall-eyebrow', 99, ['wrong']), donor])
        before = native.read_bytes()
        result = restore(candidate, native, output, self.root)
        raw = zstandard.ZstdDecompressor().decompress(output.read_bytes()[8:])
        records, end = indexed_textures(raw, len(header))
        self.assertEqual([r['packed'] for r in records], native_records + [donor])
        self.assertEqual(raw[end:], tail)
        self.assertEqual(native.read_bytes(), before)
        self.assertTrue(result['orderedNativeIndexRecordsExact'])

    def test_truncated_index_texture_metadata_is_rejected(self):
        from jak2.restore_context import indexed_textures
        with self.assertRaises((ValueError, struct.error)):
            indexed_textures(struct.pack('<Q', 1) + b'bad', 0)

    def test_graphics_fingerprint_includes_index_texture_palette(self):
        import zstandard
        path = self.root / 'source.fr3'
        name = b'example-lod0'
        def fingerprint(color):
            index = (struct.pack('<QHHIQ', 1, 1, 1, 123, 1) + b'\0'
                     + bytes([color]) * 1024 + struct.pack('<QQQ', 0, 0, 0))
            raw = (b'+\0' + struct.pack('<QQ', 0, 0) + index + b'unused-padding'
                   + struct.pack('<QQ', 1, len(name)) + name + b'geometry+\0')
            path.write_bytes(struct.pack('<Q', len(raw)) + zstandard.ZstdCompressor().compress(raw))
            return builder.fr3_asset_hash(path, name.decode())
        self.assertNotEqual(fingerprint(7), fingerprint(8))


if __name__ == '__main__':
    unittest.main()

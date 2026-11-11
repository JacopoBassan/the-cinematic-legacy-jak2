"""Build the six Jak II cinematic replacements from a verified local installation.

No extracted models, textures, animation data or generated GOAL are distributed.
The launcher supplies the original data and tools; every intermediate stays in
the caller's new work directory. No game or save is opened by this builder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import platform_tools
GAME = 'jak2'
FR3_FILES = ('GAME.fr3', 'atollext.fr3', 'underb.fr3', 'ctyasha.fr3', 'stadblmp.fr3')
ARCHIVES = ('CGO/KERNEL.CGO', 'CGO/GAME.CGO', 'CGO/COMMON.CGO',
            'DGO/HIPHOG.DGO', 'DGO/ATE.DGO', 'DGO/UNB.DGO',
            'DGO/CTYASHA.DGO', 'DGO/STADBLMP.DGO', 'DGO/GARAGE.DGO', 'DGO/LINTCSTB.DGO')
INVENTORY = {
    'sig-lod0': ['atollext.fr3', 'underb.fr3'],
    'sig-highres-lod0': ['atollext.fr3'],
    'ashelin-lod0': ['atollext.fr3', 'ctyasha.fr3'],
    'ashelin-highres-lod0': ['atollext.fr3'],
    'samos-lod0': ['stadblmp.fr3'], 'keira-lod0': ['stadblmp.fr3'],
    'samos-highres-lod0': ['lintcstb.fr3'], 'keira-highres-lod0': ['lintcstb.fr3'],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_hash(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Missing source directory: {folder.name}')
    files = sorted((p for p in folder.rglob('*') if p.is_file()),
                   key=lambda p: p.relative_to(folder).as_posix())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode() + b'\0' + sha(path).encode() + b'\n')
    return {'files': len(files), 'sha256': digest.hexdigest()}


def fr3_asset_hash(path, first_model, normalize_envmaps=False):
    """Cover consumed textures/Merc data, excluding native unused Hfrag padding."""
    import zstandard
    packed = Path(path).read_bytes()
    if len(packed) < 10:
        raise ValueError('Truncated graphics input')
    raw = zstandard.ZstdDecompressor().decompress(packed[8:])
    if len(raw) != struct.unpack_from('<Q', packed)[0] or raw[:2] != b'+\0' or raw[-2:] != b'+\0':
        raise ValueError('Unsupported graphics format')
    cursor = 2
    def number():
        nonlocal cursor
        value = struct.unpack_from('<Q', raw, cursor)[0]
        cursor += 8
        return value
    length = number()
    cursor += length
    count = number()
    for _ in range(count):
        cursor += 8
        length = number()
        cursor += length * 4
        length = number()
        cursor += length
        length = number()
        cursor += length
        cursor += 1
        if cursor > len(raw):
            raise ValueError('Truncated graphics texture table')
    from jak2.restore_context import indexed_textures
    _, index_end = indexed_textures(raw, cursor)
    encoded = first_model.encode('ascii')
    needle = struct.pack('<Q', len(encoded)) + encoded
    first = raw.find(needle, cursor)
    if first < cursor + 8 or raw.find(needle, first + 1) >= 0:
        raise ValueError('Unsupported graphics model section')
    merc = raw[first-8:]
    if normalize_envmaps:
        from fr3_fingerprint import canonical_merc
        merc = canonical_merc(merc)
    return hashlib.sha256(raw[:index_end] + merc).hexdigest()


def require_hash(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f'Unsupported or modified Jak II input: {Path(path).name}. Use an original supported OpenGOAL installation.')


def input_roots(data, config):
    """Optional explicit split roots support offline source/disc validation."""
    data = Path(data).resolve()
    source = Path(config.get('sourceData', data)).resolve()
    disc = Path(config.get('discData', data / 'iso_data/jak2')).resolve()
    return data, source, disc


def input_path(data, disc, relative):
    prefix = 'iso_data/jak2/'
    return disc / relative[len(prefix):] if relative.startswith(prefix) else data / relative


def regional_config(data, config):
    data, _, disc = input_roots(data, config)
    metadata = disc / 'buildinfo.json'
    if metadata.is_file():
        rows = json.loads(metadata.read_text(encoding='utf-8'))
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError('Invalid Jak II disc metadata')
        serial = rows[0].get('serial')
    else:
        # Also support a verified disc folder prepared before metadata export.
        # Exact ELF and every consumed archive are still checked below.
        text = (disc / 'SYSTEM.CNF').read_text(encoding='ascii')
        match = re.search(r'(SC[A-Z]{2})_(\d{3})\.(\d{2})', text)
        serial = f'{match[1]}-{match[2]}{match[3]}' if match else None
    profile = config.get('regionalProfiles', {}).get(serial)
    if serial not in config.get('supportedSerials', []) or profile is None:
        raise ValueError(f'Unsupported Jak II disc revision: {serial}')
    if metadata.is_file() and profile.get('elfHash') is not None and rows[0].get('elf_hash') != profile['elfHash']:
        raise ValueError('Unsupported executable metadata for ' + serial)
    result = dict(config)
    for field in ('inputs', 'fr3Inputs'):
        result[field] = {**config.get(field, {}), **profile.get(field, {})}
    result['selectedSerial'] = serial
    return result


def preflight(data, tools, config):
    data, tools = Path(data).resolve(), Path(tools).resolve()
    if config.get('game') != GAME or config.get('openGoalVersion') not in ('v0.3.8', 'v0.3.9'):
        raise ValueError('This builder requires a pinned, validated Jak II release definition')
    if sys.flags.optimize or sys.version_info < (3, 11):
        raise ValueError('Python 3.11 or newer without -O is required')
    for requirement in (ROOT / 'requirements.txt').read_text().splitlines():
        if not requirement or requirement.startswith('#'):
            continue
        package, expected = requirement.split('==')
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = 'not installed'
        if installed != expected:
            raise ValueError(f'{package} {expected} required; found {installed}')
    import numpy, scipy, PIL, zstandard  # noqa: F401: verify binary dependency loading
    selected = regional_config(data, config)
    _, source, disc = input_roots(data, selected)
    for name, expected in selected['tools'].items():
        require_hash(tools / name, expected)
    for name, expected in selected.get('source', {}).items():
        require_hash(source / name, expected)
    for name, expected in selected['inputs'].items():
        require_hash(input_path(data, disc, name), expected)
    for name, expected in selected['fr3Inputs'].items():
        if fr3_asset_hash(data / name, expected['firstModel'],
                          normalize_envmaps=selected.get('platform') == 'macos-arm64') != expected['sha256']:
            raise ValueError(f'Unsupported or modified native graphics: {Path(name).name}')
    for folder, expected in selected['sourceTrees'].items():
        if tree_hash(source / folder) != expected:
            raise ValueError(f'Unsupported source/configuration tree: {folder}')
    return selected


def run(command, log, env, cwd=None):
    print(f'Jak II: {log.stem}', flush=True)
    with Path(log).open('w', encoding='utf-8', newline='\n') as stream:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{log.name} failed ({result.returncode}):\n{Path(log).read_text(errors="replace")[-5000:]}')


def stage_command(script, *arguments):
    """Keep sibling imports working with the portable Python's isolated _pth.

    Embedded Python ignores PYTHONPATH and does not add a directly executed
    script's directory. Restore that normal script behavior for this child
    only, so same-named adapters resolve to the current stage.
    """
    bootstrap = ('import runpy,sys;from pathlib import Path;'
                 'script=Path(sys.argv[1]);'
                 'sys.path.insert(0,str(script.parent));'
                 'sys.argv=sys.argv[1:];'
                 'runpy.run_path(str(script),run_name="__main__")')
    return [sys.executable, '-c', bootstrap, (HERE / script).resolve(), *arguments]


def extract(disc, tools, project, archives, levels, work, env, log, config, *, rip=True):
    inputs = dict(dgo_names=list(archives), levels_to_extract=list(levels),
                  common_tpages=[3219], str_texture_file_names=[], str_file_names=[],
                  object_file_names=[], streamed_audio_file_names=[])
    path = work / (log.stem + '-inputs.json')
    path.write_text(json.dumps(inputs), encoding='utf-8')
    override = dict(inputs_file=path.as_posix(), rip_levels=rip, dump_objs=True,
                    process_game_text=False, process_game_count=False,
                    process_part_group_table=False, extract_collision=False)
    run([platform_tools.tool_path(tools, 'extractor', config), disc, '--folder', '--game', GAME,
         '--proj-path', project, '--decompile', '--decomp-config-override', json.dumps(override),
         '--disable-ansi', *platform_tools.compiler_options(config)], log, env, project)


def destinations(data, work, output):
    data, work, output = (Path(p).resolve() for p in (data, work, output))
    for target in (work, output):
        if target.exists():
            raise ValueError(f'Use a new build destination: {target}')
        if target == data or target.is_relative_to(data) or data.is_relative_to(target):
            raise ValueError('Build destinations must be outside the original installation')
    if work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError('Work and payload directories must be separate')
    return data, work, output


def prepare_disc(disc, work, config):
    """Use a small verified local disc view; never rewrite user metadata.

    Extractor selects its object map from buildinfo, even with --game jak2.
    A missing file otherwise silently selects Jak 1. The ELF itself and every
    consumed archive were already hash-checked against this exact profile.
    """
    target = work / 'disc-input'
    target.mkdir()
    prefix = 'iso_data/jak2/'
    for relative, expected in config['inputs'].items():
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):]
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(disc / name, destination)
        require_hash(destination, expected)
    profile = config['regionalProfiles'][config['selectedSerial']]
    (target / 'buildinfo.json').write_text(json.dumps([{
        'serial': config['selectedSerial'], 'elf_hash': profile['elfHash']}]), encoding='utf-8')
    return target


def build(data, tools, work, output, config):
    data, work, output = destinations(data, work, output)
    tools = Path(tools).resolve()
    config = preflight(data, tools, config)
    _, source_data, disc = input_roots(data, config)
    for protected in (tools, source_data, disc):
        for target in (work, output):
            if target == protected or target.is_relative_to(protected) or protected.is_relative_to(target):
                raise ValueError('Build destinations must be outside all injected input/tool roots')
    parent = work.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 2 * 1024**3:
        raise ValueError('At least 2 GiB of temporary free space is required')
    work.mkdir(parents=True)
    disc = prepare_disc(disc, work, config)
    project = work / 'import'
    project.mkdir()
    env = dict(os.environ, CINEMATIC_BUILD_WORK=str(work),
               PYTHONPATH=os.pathsep.join(str(x) for x in (HERE, ROOT, ROOT / 'stages')))
    for folder in ('goal_src', 'decompiler', 'game'):
        shutil.copytree(source_data / folder, project / folder)
    for folder, expected in config['sourceTrees'].items():
        if tree_hash(project / folder) != expected:
            raise ValueError('Source changed during build preparation')
    extract(disc, tools, project, ARCHIVES,
            [Path(x).name for x in ARCHIVES if x.startswith('DGO/')], work, env, work / '01-native-extraction.log', config)
    # Full native level context always comes from the verified installation.
    # The selective extraction is used only to obtain donor assets and GLBs.
    for filename in FR3_FILES:
        shutil.copy2(data / 'out/jak2/fr3' / filename, project / 'out/jak2/fr3' / filename)
    (work / 'variant-audit.json').write_text(json.dumps({'modelInventory': {
        name: [{'file': file} for file in files] for name, files in INVENTORY.items()}}), encoding='utf-8')
    step = 2
    def stage(script, *arguments):
        nonlocal step
        log = work / f'{step:02}-{Path(script).stem}.log'
        step += 1
        run(stage_command(script, *arguments), log, env, project)
    stage('build_prototypes.py', '--data', project, '--output', work / 'rig-prototypes')
    stage('fit_faces.py', '--data', project, '--output', work / 'face-refined')
    stage('import_prototypes.py', '--data', project, '--disc', disc,
          '--work', work, '--extractor', platform_tools.tool_path(tools, 'extractor', config), '--project', work / 'player-import')
    stage('restore_materials.py', '--data', project, '--work', work,
          '--input', work / 'player-import/out/jak2/fr3/GAME.fr3')
    stage('build_faces.py', '--data', project, '--work', work)
    stage('restore_textures.py', '--data', project, '--work', work)
    stage('restore_context.py', '--candidate', work / 'GAME-cinematic.fr3',
          '--original', data / 'out/jak2/fr3/GAME.fr3', '--output', work / 'GAME-context.fr3', '--work', work)
    stage('build_npcs.py', '--data', project, '--work', work, '--output', work / 'npc-geometry')
    stage('npc_materials.py', '--data', project, '--work', work, '--input', work / 'npc-geometry',
          '--output', work / 'npc-materials', '--only', 'sig', 'ashelin')
    source_glb = work / 'npc-samos-keira-source/decompiler_out/jak2/levels/stadblmp'
    source_glb.mkdir(parents=True)
    for name in ('samos-lod0.glb', 'keira-lod0.glb'):
        shutil.copy2(project / 'decompiler_out/jak2/levels/stadblmp' / name, source_glb / name)
    stage('build_samos_keira.py', '--data', project, '--work', work, '--source', project,
          '--output', work / 'npc-samos-keira-prototypes')
    stage('import_npcs.py', '--data', project, '--disc', disc,
          '--work', work, '--extractor', platform_tools.tool_path(tools, 'extractor', config))
    stage('npc_materials.py', '--data', project, '--work', work, '--input', work / 'npc-import/out/jak2/fr3',
          '--output', work / 'npc-sk-materials', '--only', 'samos', 'keira')
    stage('fit_npc_faces.py', '--work', work, '--output', work / 'npc-face-fits', '--model', 'keira',
          '--stage-input', work / 'npc-sk-materials/stadblmp.fr3',
          '--stage-output', work / 'npc-sk-faces/stadblmp.fr3')
    stage('stage_npcs.py', '--data', project, '--work', work, '--geometry', work / 'npc-materials',
          '--faces', work / 'npc-sk-faces', '--output', work / 'npc-ready')
    stage('validate_payload.py', '--data', data, '--work', work)
    runtime_sources = {
        'foreground-cinematic.gc': 'engine/gfx/foreground/foreground.gc',
        'texture-anim-cinematic.gc': 'engine/gfx/texture/texture-anim.gc',
        'process-drawable-cinematic.gc': 'engine/process-drawable/process-drawable.gc',
    }
    for name, relative in runtime_sources.items():
        shutil.copy2(work / 'npc-ready' / name, project / 'goal_src/jak2' / relative)
    # Compile native prerequisites, then the generated helper in the same type
    # environment. Insert it after native eye.o without rewriting other objects.
    helper = work / 'npc-ready/cinematic-eyes.gc'
    territory = config['regionalProfiles'][config['selectedSerial']]['territory']
    command = '(begin (seval (set! *default-territory* ' + str(territory) + ')) (make "$OUT/obj/foreground.o") (make "$OUT/obj/texture-anim.o") (make "$OUT/obj/process-drawable.o") (make "$OUT/obj/eye.o") (asm-file ' + json.dumps(helper.as_posix()) + ' :color :write))'
    run([platform_tools.tool_path(tools, 'goalc', config), '--game', GAME, '--proj-path', project, '--disable-ansi',
         '--port', '8282', '--debug-port', '8229', *platform_tools.compiler_options(config), '--cmd', command], work / '16-compile.log', env, project)
    from jak2.patch_archive import assemble
    archive_report = assemble(data / 'out/jak2/iso/GAME.CGO', project / 'out/jak2/obj', work / 'GAME-cinematic.CGO')
    output.mkdir(parents=True)
    generated = {'data/out/jak2/fr3/GAME.fr3': work / 'GAME-context.fr3',
                 'data/out/jak2/iso/GAME.CGO': work / 'GAME-cinematic.CGO'}
    generated.update({f'data/out/jak2/fr3/{name}': work / 'npc-ready' / name for name in FR3_FILES if name != 'GAME.fr3'})
    if set(generated) != set(config['payloadPaths']):
        raise ValueError('Generated payload set differs from the release definition')
    files = []
    for relative, source in sorted(generated.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require_hash(destination, sha(source))
        files.append({'path': relative, 'sha256': sha(destination), 'bytes': destination.stat().st_size})
    manifest = {'schemaVersion': 1, 'game': GAME, 'version': config['version'],
                'openGoalVersion': config['openGoalVersion'], 'platform': config['platform'],
                'displayName': 'Jak II: The Cinematic Legacy', 'author': '@jacopobassan',
                'website': 'https://jacopobassan.me', 'runtimeSha256': config['tools'][platform_tools.tool_path(tools, 'gk', config).name],
                'privateGeneratedPayload': True, 'releaseReady': False,
                'sourceDiscSerial': config['selectedSerial'],
                'features': {'models': ['Jak', 'Daxter', 'Sig', 'Ashelin', 'stadium Samos', 'stadium Keira'],
                             'nativeLiveEyes': True, 'daxterAuthoredEyeUVs': True,
                             'nativeDarkHorns': True, 'nativeAnimatedBeard': True,
                             'lazy64BitBlinkOwnership': True}, 'files': files}
    (output / 'archive-validation.json').write_text(json.dumps(archive_report, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(work / 'payload-validation.json', output / 'validation.json')
    (output / 'DO-NOT-PUBLISH.txt').write_text('Locally generated game assets. Distribute only the asset-free builder package.\n', encoding='utf-8')
    preflight(data, tools, config)
    temporary = output / 'manifest.json.new'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, output / 'manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', '--game-data', dest='data', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=HERE / 'compatibility.json')
    parser.add_argument('--platform', choices=platform_tools.PLATFORMS)
    parser.add_argument('--source-data', type=Path, help='Explicit original source root for offline validation')
    parser.add_argument('--disc-data', type=Path, help='Explicit original extracted disc folder for offline validation')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = platform_tools.select(json.loads(args.config.read_text(encoding='utf-8')), args.platform)
    for field, value in (('sourceData', args.source_data), ('discData', args.disc_data)):
        if value is not None:
            config[field] = str(value.resolve())
    if args.check:
        selected = preflight(args.data, args.tools, config)
        print('Verified original Jak II inputs: ' + selected['selectedSerial'])
    else:
        if args.work is None or args.output is None:
            parser.error('--work and --output are required for building')
        print(json.dumps(build(args.data, args.tools, args.work, args.output, config), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Jak II build stopped: {error}', file=sys.stderr)
        raise SystemExit(1)

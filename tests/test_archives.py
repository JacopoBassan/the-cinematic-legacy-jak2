"""Asset-free file-preservation checks for the Jak II archive assembler."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builder'))
from jak2 import patch_archive as patch


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(os.environ.get('CINEMATIC_TEST_TEMP', tempfile.gettempdir())).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix='cinematic-archive-', dir=self.scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.scratch))
        self.objects = self.root / 'objects'
        self.objects.mkdir()
        for name in (*patch.REPLACEMENTS, patch.HELPER):
            (self.objects / (name + '.o')).write_bytes(('new-' + name).encode())
        names = ['foreground', 'collectables', 'eye', 'texture-anim',
                 'collectables', 'process-drawable', 'unchanged']
        records = [patch.object_record(n, ('original-' + str(i)).encode())
                   for i, n in enumerate(names)]
        self.original = self.root / 'original.CGO'
        self.original.write_bytes(struct.pack('<I', len(names)) + b'GAME'.ljust(60, b'\0')
                                  + b''.join(records) + b'native-tail')
        self.before = self.original.read_bytes()
        self.output = self.root / 'new.CGO'

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.scratch))
        self.temporary.cleanup()

    def test_duplicate_native_records_and_tail_survive(self):
        result = patch.assemble(self.original, self.objects, self.output)
        self.assertEqual(result['nativeDuplicateRecordsPreserved'], {'collectables': 2})
        _, source, tail = patch.entries(self.before)
        _, output, actual_tail = patch.entries(self.output.read_bytes())
        native = [(n, raw) for n, raw in output if n != patch.HELPER]
        for old, new in zip(source, native):
            self.assertEqual(old[0], new[0])
            if old[0] not in patch.REPLACEMENTS:
                self.assertEqual(old, new)
        self.assertEqual(tail, actual_tail)
        self.assertEqual(self.original.read_bytes(), self.before)

    def test_cli_rejects_destructive_reports_before_assembly(self):
        existing = self.root / 'existing.json'
        existing.write_bytes(b'keep me')
        for report in (self.original, self.output, existing,
                       self.objects / 'foreground.o'):
            with self.subTest(report=report.name):
                result = subprocess.run([sys.executable, str(Path(patch.__file__)),
                    '--original', str(self.original), '--objects', str(self.objects),
                    '--output', str(self.output), '--report', str(report)],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.original.read_bytes(), self.before)
                self.assertEqual(existing.read_bytes(), b'keep me')
                self.assertEqual((self.objects / 'foreground.o').read_bytes(), b'new-foreground')

    def test_complete_report_and_concurrent_target_are_preserved(self):
        destination = patch.report_path(self.original, self.objects, self.output,
                                        self.root / 'report.json')
        patch.publish_report(destination, {'complete': True})
        self.assertEqual(json.loads(destination.read_text()), {'complete': True})
        before = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            patch.publish_report(destination, {'replacement': True})
        self.assertEqual(destination.read_bytes(), before)

    def test_existing_archive_is_never_overwritten(self):
        with self.assertRaises(ValueError):
            patch.assemble(self.original, self.objects, self.original)
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).digest(),
                         hashlib.sha256(self.before).digest())


if __name__ == '__main__':
    unittest.main()

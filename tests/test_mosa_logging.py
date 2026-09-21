import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_mosa_logged import relay, load_settings


class LoggingTests(unittest.TestCase):
    def test_only_exact_nvmap_messages_hidden_and_raw_preserved(self):
        lines = ['NvMapMemAllocInternalTagged: 1075072515 error 12\n',
                 'NvMapMemHandleAlloc: error 0\n',
                 'NvMapMemHandleAlloc: error 5\n',
                 'Corrupt JPEG data: premature end of data segment\n',
                 'RuntimeError: camera failed\n']
        raw, terminal = io.StringIO(), io.StringIO()
        self.assertEqual(relay(lines, raw, terminal, True), (1, 2))
        for line in lines:
            self.assertIn(line, raw.getvalue())
        self.assertNotIn(lines[0], terminal.getvalue())
        for line in lines[2:]:
            self.assertIn(line, terminal.getvalue())
        self.assertIn('JPEG warning #1', terminal.getvalue())

    def test_launcher_captures_native_fds_and_preserves_exit_code(self):
        launcher = Path(__file__).resolve().parents[1] / 'scripts/run_mosa_logged.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data.json').write_text('{"hide_nvmap_messages": true}')
            script = root / 'fake.py'
            script.write_text('import os, sys\nos.write(2, b"NvMapMemHandleAlloc: error 0\\n")\nos.write(1, b"visible\\n")\nsys.exit(7)\n')
            result = subprocess.run([sys.executable, str(launcher), '--script', str(script)],
                                    cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 7)
            self.assertNotIn('NvMapMemHandleAlloc:', result.stdout)
            self.assertIn('visible', result.stdout)
            self.assertIn('NvMapMemHandleAlloc:', next(root.glob('artifacts/runtime-logs/*.log')).read_text())

    def test_config_default_and_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text('{"url":"http://example.test" // comment\n}')
            self.assertFalse(load_settings(path))
            path.write_text('{"hide_nvmap_messages":"true"}')
            with self.assertRaises(ValueError):
                load_settings(path)

"""CSV compatibility, per-row visibility and reusable USB writer lifecycle."""
import ast
import csv
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

# Exercise the actual CSV implementation without importing the unrelated
# camera/inference stack (numpy/torch/OpenCV).
source = Path(__file__).resolve().parents[1] / 'scripts/mosa_jetson_bridge.py'
nodes = [node for node in ast.parse(source.read_text()).body
         if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and node.name in ('MeasurementLog', '_prepare_measurement_log')]
bridge = types.ModuleType('measurement_log_under_test')
bridge.__dict__.update(os=os, csv=csv, tempfile=tempfile, Path=Path)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), bridge.__dict__)
MeasurementLog = bridge.MeasurementLog


class MeasurementLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.path = self.folder / 'log.csv'
        self.log = MeasurementLog()
        self.addCleanup(self.log.close)
        self.header = ['Model', 'score']

    def rows(self, path=None):
        with (self.path if path is None else path).open(newline='', encoding='utf-8') as file:
            return list(csv.reader(file))

    def test_each_row_is_visible_before_close_with_identical_csv_bytes(self):
        expected = io.StringIO(newline='')
        writer = csv.writer(expected)
        writer.writerow(self.header + ['backend'])
        for row, backend in ((['Model A', '.2'], 'onnx'),
                             (['검사,"B"\n두번째 줄', '.3'], 'tensorrt')):
            self.log.append(self.path, self.header, row, backend)
            writer.writerow(row + [backend])
            self.assertEqual(self.path.read_bytes(), expected.getvalue().encode('utf-8'))

    def test_warm_appends_do_not_open_stat_or_reread_header(self):
        self.path.write_text('Model,score,backend\r\nold,.1,onnx\r\n', encoding='utf-8')
        self.log.append(self.path, self.header, ['first', '.2'], 'onnx')
        with patch.object(Path, 'open', side_effect=AssertionError('reopened CSV')), \
                patch.object(Path, 'stat', side_effect=AssertionError('stat on USB')), \
                patch.object(Path, 'exists', side_effect=AssertionError('exists on USB')):
            self.log.append(self.path, self.header, ['second', '.3'], 'tensorrt')
            self.log.append(self.path, self.header, ['third', '.4'], 'onnx')
        self.assertEqual(self.rows()[1:], [['old', '.1', 'onnx'], ['first', '.2', 'onnx'],
                                          ['second', '.3', 'tensorrt'], ['third', '.4', 'onnx']])

    def test_legacy_migration_preserves_rows_and_does_not_repeat(self):
        self.path.write_text('Model,score\r\nold,.1\r\n', encoding='utf-8')
        with patch.object(bridge.os, 'replace', wraps=bridge.os.replace) as replace:
            self.log.append(self.path, self.header, ['first', '.2'], 'onnx')
            self.log.append(self.path, self.header, ['second', '.3'], 'tensorrt')
        replace.assert_called_once()
        self.assertEqual(self.rows(), [self.header + ['backend'], ['old', '.1', 'unknown'],
                                       ['first', '.2', 'onnx'], ['second', '.3', 'tensorrt']])

    def test_path_switch_closes_old_file_and_preserves_both_logs(self):
        other = self.folder / 'other.csv'
        self.log.append(self.path, self.header, ['first', '.1'], 'onnx')
        first = self.log._file
        self.log.append(other, self.header, ['second', '.2'], 'tensorrt')
        self.assertTrue(first.closed)
        second = self.log._file
        self.log.append(self.path, self.header, ['third', '.3'], 'onnx')
        self.assertTrue(second.closed)
        self.assertEqual(self.rows(), [self.header + ['backend'], ['first', '.1', 'onnx'],
                                       ['third', '.3', 'onnx']])
        self.assertEqual(self.rows(other), [self.header + ['backend'], ['second', '.2', 'tensorrt']])

    def test_relative_and_absolute_paths_share_one_open_file(self):
        with patch.object(bridge.os, 'getcwd', return_value=str(self.folder)):
            self.log.append('log.csv', self.header, ['first', '.1'], 'onnx')
        with patch.object(Path, 'open', side_effect=AssertionError('reopened same CSV')):
            self.log.append(self.path, self.header, ['second', '.2'], 'onnx')
        self.assertEqual(len(self.rows()), 3)

    def test_header_change_is_revalidated_without_corrupting_existing_log(self):
        self.log.append(self.path, self.header, ['first', '.1'], 'onnx')
        before = self.path.read_bytes()
        file = self.log._file
        with self.assertRaisesRegex(ValueError, 'Unexpected measurement CSV header'):
            self.log.append(self.path, ['Model', 'other'], ['second', '.2'], 'onnx')
        self.assertTrue(file.closed)
        self.assertEqual(self.path.read_bytes(), before)
        self.log.append(self.path, self.header, ['third', '.3'], 'onnx')
        self.assertEqual(self.rows()[-1], ['third', '.3', 'onnx'])

    def test_failed_migration_keeps_original_and_cleans_temporary_file(self):
        self.path.write_text('Model,score\r\nold,.1\r\n', encoding='utf-8')
        before = self.path.read_bytes()
        with patch.object(bridge.os, 'replace', side_effect=OSError('USB replace failed')):
            with self.assertRaisesRegex(OSError, 'USB replace failed'):
                self.log.append(self.path, self.header, ['first', '.2'], 'onnx')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.folder.iterdir()), [self.path])

    def test_open_failure_can_be_retried_without_replaying_failed_row(self):
        with patch.object(Path, 'open', side_effect=OSError('USB missing')):
            with self.assertRaisesRegex(OSError, 'USB missing'):
                self.log.append(self.path, self.header, ['failed', '.1'], 'onnx')
        self.log.append(self.path, self.header, ['next', '.2'], 'onnx')
        self.assertEqual(self.rows(), [self.header + ['backend'], ['next', '.2', 'onnx']])

    def test_write_or_flush_failure_discards_handle_without_automatic_row_replay(self):
        for action in ('write', 'flush'):
            with self.subTest(action=action):
                failed, recovered = Mock(), Mock()
                failed.write.side_effect = lambda value: len(value)
                recovered.write.side_effect = lambda value: len(value)
                getattr(failed, action).side_effect = OSError(f'USB {action} failed')
                failed.close.side_effect = OSError('cleanup also failed')
                with patch.object(bridge, '_prepare_measurement_log', return_value=False), \
                        patch.object(Path, 'open', side_effect=[failed, recovered]):
                    with self.assertRaisesRegex(OSError, f'USB {action} failed'):
                        self.log.append(self.path, self.header, ['failed', '.1'], 'onnx')
                    failed.close.assert_called_once()
                    self.log.append(self.path, self.header, ['next', '.2'], 'onnx')
                    recovered.write.assert_called_once_with('next,.2,onnx\r\n')
                    recovered.flush.assert_called_once()
                    self.log.close()

    def test_close_is_idempotent_and_reenable_writes_one_header(self):
        self.log.append(self.path, self.header, ['first', '.1'], 'onnx')
        file = self.log._file
        self.log.close()
        self.log.close()
        self.assertTrue(file.closed)
        self.log.append(self.path, self.header, ['second', '.2'], 'onnx')
        self.assertEqual(self.rows(), [self.header + ['backend'], ['first', '.1', 'onnx'],
                                       ['second', '.2', 'onnx']])

    def test_close_failure_does_not_keep_a_broken_handle_cached(self):
        first, second = Mock(), Mock()
        with patch.object(bridge, '_prepare_measurement_log', return_value=False), \
                patch.object(Path, 'open', side_effect=[first, second]):
            self.log.append(self.path, self.header, ['first', '.1'], 'onnx')
            first.close.side_effect = OSError('USB close failed')
            with self.assertRaisesRegex(OSError, 'USB close failed'):
                self.log.close()
            self.log.append(self.path, self.header, ['second', '.2'], 'onnx')
            second.write.assert_called_once_with('second,.2,onnx\r\n')
            self.log.close()

    def test_missing_usb_directory_is_not_created_or_redirected(self):
        path = self.folder / 'missing-usb' / 'log.csv'
        with self.assertRaises(FileNotFoundError):
            self.log.append(path, self.header, ['first', '.1'], 'onnx')
        self.assertFalse(path.parent.exists())


if __name__ == '__main__':
    unittest.main()

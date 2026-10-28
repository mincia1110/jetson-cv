"""Real file descriptors and GUI callbacks, without models, camera or display."""
import ast
from concurrent.futures import Future
import csv
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mosa_storage import BackgroundStorageSync

# Match the existing headless GUI tests. Load only stdlib CSV code from the
# bridge so these durability tests also run without numpy/torch installed.
SOURCE = ROOT / 'MOSA_visualAD_comb_jetson.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
GUI = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'CameraGUI')
NS = dict(os=os, sys=sys, json=json, time=time, messagebox=Mock(),
          BackgroundStorageSync=BackgroundStorageSync)
exec(compile(ast.Module(body=[GUI], type_ignores=[]), str(SOURCE), 'exec'), NS)
bridge = ROOT / 'scripts/mosa_jetson_bridge.py'
nodes = [n for n in ast.parse(bridge.read_text()).body
         if isinstance(n, (ast.ClassDef, ast.FunctionDef))
         and n.name in ('MeasurementLog', '_prepare_measurement_log')]
LOG_NS = dict(os=os, csv=csv, tempfile=tempfile, Path=Path)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(bridge), 'exec'), LOG_NS)


class StorageSyncTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.images = self.folder / 'images'
        self.logs = self.folder / 'logs'
        self.images.mkdir()
        self.logs.mkdir()
        self.raw = self.images / 'sample.png'
        self.result = self.images / 'sample_resulf.jpg'
        self.raw.write_bytes(b'raw image')
        self.result.write_bytes(b'result image')
        self.log = LOG_NS['MeasurementLog']()
        self.addCleanup(self.log.close)
        self.csv = self.logs / 'log.csv'
        self.log.append(self.csv, ['score'], ['.1'], 'onnx')
        self.guard = Mock()
        self.storage = BackgroundStorageSync(self.guard)
        self.addCleanup(self.storage.close)
        printer = patch('mosa_storage.print')
        printer.start()
        self.addCleanup(printer.stop)

    def test_fsyncs_each_output_and_unique_parent_directories(self):
        seen = []
        def fsync(fd):
            entry = os.fstat(fd)
            seen.append((entry.st_ino, stat.S_ISDIR(entry.st_mode)))
        duplicate = self.log.duplicate_fd()
        with patch('mosa_storage.os.fsync', side_effect=fsync):
            timing = self.storage.submit([self.raw, self.result], csv_path=str(self.csv),
                                         csv_fd=duplicate).result(timeout=2)
        expected = [(p.stat().st_ino, p.is_dir()) for p in
                    (self.raw, self.result, self.csv, self.images, self.logs)]
        self.assertCountEqual(seen, expected)
        self.assertGreaterEqual(timing['sync_seconds'], 0)
        self.assertEqual(self.storage.pending, 0)
        with self.assertRaises(OSError):
            os.fstat(duplicate)
        self.assertFalse(self.log._file.closed)

    def test_slow_sync_does_not_block_submission_and_jobs_are_serial(self):
        entered, release = threading.Event(), threading.Event()
        threads = []
        def slow_fsync(fd):
            threads.append(threading.get_ident())
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test worker was not released')
        with patch('mosa_storage.os.fsync', side_effect=slow_fsync):
            try:
                first = self.storage.submit([self.raw])
                self.assertTrue(entered.wait(1))
                second = self.storage.submit([self.result])
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertEqual(self.storage.pending, 2)
            finally:
                release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        self.assertEqual(len(set(threads)), 1)
        self.assertNotIn(threading.get_ident(), threads)

    def test_image_sync_failure_still_syncs_csv_and_reports_once(self):
        seen = []
        def fsync(fd):
            inode = os.fstat(fd).st_ino
            seen.append(inode)
            if inode == self.raw.stat().st_ino:
                raise OSError('card writeback failed')
        duplicate = self.log.duplicate_fd()
        with patch('mosa_storage.os.fsync', side_effect=fsync):
            future = self.storage.submit([self.raw], csv_path=str(self.csv), csv_fd=duplicate)
            with self.assertRaisesRegex(RuntimeError, 'card writeback failed'):
                future.result(timeout=2)
        self.assertIn(self.csv.stat().st_ino, seen)
        self.assertEqual(len(self.storage.take_errors()), 1)
        self.assertEqual(self.storage.take_errors(), [])
        self.assertEqual(self.storage.pending, 0)
        with self.assertRaises(OSError):
            os.fstat(duplicate)

    def test_mount_loss_blocks_fsync_without_fallback(self):
        self.guard.side_effect = RuntimeError('JETSON_DATA is not mounted')
        duplicate = self.log.duplicate_fd()
        with patch('mosa_storage.os.fsync') as fsync:
            future = self.storage.submit([self.raw], csv_path=str(self.csv), csv_fd=duplicate)
            with self.assertRaisesRegex(RuntimeError, 'JETSON_DATA is not mounted'):
                future.result(timeout=2)
            fsync.assert_not_called()
        with self.assertRaises(OSError):
            os.fstat(duplicate)

    def test_directory_sync_failure_is_reported_and_descriptor_is_closed(self):
        directories = []
        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directories.append(fd)
                raise OSError('directory sync failed')
        with patch('mosa_storage.os.fsync', side_effect=fsync):
            with self.assertRaisesRegex(RuntimeError, 'directory sync failed'):
                self.storage.submit([self.raw]).result(timeout=2)
        for fd in directories:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_duplicate_survives_csv_close_and_path_switch(self):
        duplicate = self.log.duplicate_fd()
        inode = os.fstat(duplicate).st_ino
        self.log.close()
        other = self.logs / 'next.csv'
        self.log.append(other, ['score'], ['.2'], 'onnx')
        seen = []
        with patch('mosa_storage.os.fsync', side_effect=lambda fd: seen.append(os.fstat(fd).st_ino)):
            self.storage.submit([], csv_path=str(self.csv), csv_fd=duplicate).result(timeout=2)
        self.assertIn(inode, seen)
        self.assertNotIn(other.stat().st_ino, seen)
        self.log.append(other, ['score'], ['.3'], 'onnx')
        self.assertIn('.3,onnx', other.read_text())

    def test_submit_after_shutdown_closes_transferred_csv_descriptor(self):
        duplicate = self.log.duplicate_fd()
        self.storage.close()
        with self.assertRaises(RuntimeError):
            self.storage.submit([], csv_path=str(self.csv), csv_fd=duplicate)
        self.assertEqual(self.storage.pending, 0)
        with self.assertRaises(OSError):
            os.fstat(duplicate)

    def test_shutdown_drains_queued_work(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        def slow_fsync(fd):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test worker was not released')
        def close():
            self.storage.close()
            closed.set()
        with patch('mosa_storage.os.fsync', side_effect=slow_fsync):
            future = self.storage.submit([self.raw])
            self.assertTrue(entered.wait(1))
            closer = threading.Thread(target=close)
            closer.start()
            try:
                self.assertFalse(closed.is_set())
            finally:
                release.set()
                closer.join(timeout=2)
            self.assertTrue(closed.is_set())
            future.result(timeout=2)


class GuiStorageSyncTests(unittest.TestCase):
    def setUp(self):
        NS['messagebox'].reset_mock()
        self.app = NS['CameraGUI'].__new__(NS['CameraGUI'])
        self.app._storage_sync = Mock(pending=0)
        self.app._storage_sync.take_errors.return_value = []
        self.app._storage_sync_polling = False
        self.app._measurement_busy = False
        self.app._model_changing = False
        self.app._model_cleanup_failed = False
        self.app._camera_future = None
        self.app.running = True
        self.app.inferencer = Mock()
        self.app.measurement_log = Mock()
        for name in ('root', 'name_entry', 'threshold_entry', 'folder_button',
                     'measure_button', 'model_change_btn', 'last_save_label',
                     'result_label', 'result_container'):
            setattr(self.app, name, Mock())

    def test_background_sync_does_not_hold_busy_state_or_change_labels(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.png'
            path.write_bytes(b'image')
            def capture():
                self.app._pending_sync_images = [str(path)]
            def slow_fsync(fd):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test worker was not released')
            self.app._storage_sync = None
            self.app._capture_image_impl = Mock(side_effect=capture)
            with patch('mosa_storage.os.fsync', side_effect=slow_fsync), patch('mosa_storage.print'):
                try:
                    self.app.capture_image()
                    self.assertTrue(entered.wait(1))
                    self.assertFalse(self.app._measurement_busy)
                    self.app.measure_button.config.assert_called_with(state='normal')
                    self.app.capture_image()
                    self.assertEqual(self.app._capture_image_impl.call_count, 2)
                    self.assertEqual(self.app._storage_sync.pending, 2)
                finally:
                    release.set()
                    self.app._storage_sync.close()
            self.app._poll_storage_sync()
        for label in (self.app.result_label, self.app.last_save_label):
            label.config.assert_not_called()
        self.app.measurement_log.duplicate_fd.assert_not_called()
        NS['messagebox'].showerror.assert_not_called()

    def test_partial_output_is_queued_even_when_inference_fails(self):
        def fail():
            self.app._pending_sync_images = ['sample.png']
            raise RuntimeError('inference failed')
        self.app._capture_image_impl = Mock(side_effect=fail)
        self.app._show_camera_failure = Mock()
        self.app.capture_image()
        self.app._storage_sync.submit.assert_called_once_with(
            ['sample.png'], csv_path=None, csv_fd=None)
        self.assertFalse(self.app._measurement_busy)
        self.app._queue_storage_sync()
        self.app._storage_sync.submit.assert_called_once()

    def test_camera_completion_queues_outputs_without_waiting_for_sync(self):
        self.app._camera_recovery_started = threading.Event()
        self.app._camera_future = Future()
        self.app._camera_future.set_result({'frame': object(), 'report': {'reset_performed': False}})
        self.app._storage_sync.pending = 1
        def process(*args):
            self.app._pending_sync_images = ['sample.png', 'sample_resulf.jpg']
            self.app._pending_sync_csv = 'logs/log.csv'
        self.app._process_measurement = Mock(side_effect=process)
        self.app._poll_camera('sample', 'sample.png', 0)
        self.app._storage_sync.submit.assert_called_once_with(
            ['sample.png', 'sample_resulf.jpg'], csv_path='logs/log.csv',
            csv_fd=self.app.measurement_log.duplicate_fd.return_value)
        self.assertFalse(self.app._measurement_busy)

    def test_only_errors_show_dialogs_on_the_gui_thread(self):
        self.app._storage_sync_polling = True
        self.app._storage_sync.take_errors.return_value = ['card writeback failed']
        threads = []
        with patch.object(NS['messagebox'], 'showerror',
                          side_effect=lambda *args: threads.append(threading.get_ident())) as dialog:
            self.app._poll_storage_sync()
            dialog.assert_called_once_with('저장 동기화 실패', 'card writeback failed')
        self.assertEqual(threads, [threading.get_ident()])
        self.assertFalse(self.app._storage_sync_polling)
        self.app.result_label.config.assert_not_called()

    def test_main_waits_for_storage_after_window_closes(self):
        main = next(n for n in TREE.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "__name__ == '__main__'")
        events = []
        root, app = Mock(), Mock()
        app.camera = None
        root.mainloop.side_effect = lambda: events.append('window closed')
        app._storage_sync.close.side_effect = lambda: events.append('storage drained')
        app.measurement_log.close.side_effect = lambda: events.append('log closed')
        env = dict(__name__='__main__', print=Mock(), tk=Mock(Tk=Mock(return_value=root)),
                   show_model_selection_gui=Mock(return_value=('A', 'model.onnx', None, {})),
                   MOSAInferencer=Mock(), camera_init=Mock(return_value=(None, None, False)),
                   CameraGUI=Mock(return_value=app))
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(SOURCE), 'exec'), env)
        self.assertEqual(events, ['window closed', 'storage drained', 'log closed'])


if __name__ == '__main__':
    unittest.main()

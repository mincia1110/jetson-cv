"""Exercise actual Tk callback methods without a display, models or camera."""
import ast
from concurrent.futures import Future
import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'MOSA_visualAD_comb_jetson.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
gui = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CameraGUI')
namespace = {'json': json, 'messagebox': Mock(), 'os': os, 'time': time,
             'datetime': datetime}
exec(compile(ast.Module(body=[gui], type_ignores=[]), str(source), 'exec'), namespace)


class CameraFlowTests(unittest.TestCase):
    def setUp(self):
        namespace['messagebox'].reset_mock()
        self.app = namespace['CameraGUI'].__new__(namespace['CameraGUI'])
        self.app.running = True
        self.app._measurement_busy = True
        self.app._camera_future = Future()
        self.app._camera_recovery_started = threading.Event()
        for name in ('root', 'result_container', 'result_label', 'last_save_label',
                     'measure_button', 'name_entry', 'threshold_entry', 'folder_button'):
            setattr(self.app, name, Mock())
        self.app._process_measurement = Mock()

    def finish_recovery(self, after_reasons):
        self.app._camera_recovery_started.set()
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'FAIL\n(WAIT)')
        self.assertTrue(self.app._measurement_busy)
        self.app._camera_future.set_result({'frame': None, 'report': {
            'reset_performed': True, 'reset_succeeded': True,
            'manual_retry_required': True, 'allow_inference': False,
            'after': {'RG_diff': 18 if after_reasons else 5, 'reasons': after_reasons}}})
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'READY')
        self.assertEqual(self.app.result_container.config.call_args.kwargs['bg'], 'gray')
        self.assertFalse(self.app._measurement_busy)
        self.assertIsNone(self.app._camera_future)
        self.app._process_measurement.assert_not_called()

    def test_reset_pass_waits_for_operator_without_inference(self):
        self.finish_recovery([])

    def test_reset_done_with_bad_image_is_ready_but_not_a_pass(self):
        self.finish_recovery(['RG_gab error'])
        self.assertIn('아직 미달', self.app.last_save_label.config.call_args.kwargs['text'])

    def test_reset_failure_stays_fail(self):
        self.app._camera_future.set_result({'frame': None, 'report': {
            'reset_performed': True, 'manual_retry_required': False,
            'after': {'reasons': ['device missing']}}})
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'FAIL')
        self.app._process_measurement.assert_not_called()
        self.assertFalse(self.app._measurement_busy)

    def test_good_frame_continues_on_gui_thread(self):
        frame = object()
        self.app._camera_future.set_result({'frame': frame, 'report': {'reset_performed': False}})
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.app._process_measurement.assert_called_once_with('sample', 'sample.bmp', 0, frame)
        self.assertFalse(self.app._measurement_busy)

    def test_duplicate_click_and_closed_window_do_not_start_work(self):
        self.app._capture_image_impl = Mock()
        self.app.capture_image()
        self.app._capture_image_impl.assert_not_called()
        self.app.running = False
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.app.root.after.assert_not_called()

    def test_exception_releases_busy_state(self):
        self.app._camera_future.set_exception(RuntimeError('worker failed'))
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.assertIsNone(self.app._camera_future)
        self.assertFalse(self.app._measurement_busy)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'FAIL')
        self.app._process_measurement.assert_not_called()

    def test_explicit_reset_is_not_cleared_by_measurement_callback(self):
        self.app.camera = Mock()
        reset = Future()
        # Even if reconnect finishes immediately, its own callback must handle it.
        reset.set_result({})
        self.app.camera.reconnect.return_value = reset
        self.app._process_measurement.side_effect = lambda *args: self.app.reset_camera()
        self.app._camera_future.set_result({'frame': object(), 'report': {'reset_performed': False}})
        self.app._poll_camera('sample', 'sample.bmp', 0)
        self.assertIs(self.app._camera_future, reset)
        self.assertTrue(self.app._measurement_busy)
        self.app._poll_reset()
        self.assertFalse(self.app._measurement_busy)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'READY')

    def prepare_file_capture(self, folder, mode=0):
        self.app.save_dir = str(folder)
        self.app.Camera_or_png_flag = mode
        self.app.name_entry.get.return_value = 'sample'
        self.app.threshold_entry.get.return_value = '5'
        self.app.frame_lock = threading.Lock()
        self.app.load_json = Mock()

    def test_file_input_prefers_png_then_reads_legacy_bmp_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.prepare_file_capture(folder)
            bmp, png = folder / 'sample.bmp', folder / 'sample.png'
            bmp.touch()
            self.app._capture_image_impl()
            self.assertEqual(self.app._process_measurement.call_args.args[1], str(bmp))
            png.touch()
            self.app._capture_image_impl()
            self.assertEqual(self.app._process_measurement.call_args.args[1], str(png))
            namespace['messagebox'].askyesno.assert_not_called()
            self.app.load_json.assert_called()

    def test_missing_file_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare_file_capture(folder)
            self.app._capture_image_impl()
            self.app._process_measurement.assert_not_called()
            namespace['messagebox'].showwarning.assert_called_once()

    def test_camera_capture_targets_png_and_honors_overwrite_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare_file_capture(folder, mode=1)
            self.app.camera = Mock()
            self.app.camera_settings = {}
            expected = Path(folder) / 'sample.png'
            self.app._capture_image_impl()
            self.app.camera.prepare_inference.assert_called_once()
            self.assertEqual(self.app.root.after.call_args.args[3], str(expected))
            expected.touch()
            self.app.camera.reset_mock()
            namespace['messagebox'].askyesno.return_value = False
            self.app._capture_image_impl()
            self.app.camera.prepare_inference.assert_not_called()
            self.assertIn('sample.png', namespace['messagebox'].askyesno.call_args.args[1])

    def test_load_json_accepts_new_and_legacy_mode_keys_with_new_key_priority(self):
        data = dict(CSV_path='logs', BRIGHT_min=0, BRIGHT_max=255, RG_gab=18,
                    crop_v_offset=0, threshold=0, sub_shift=0, sub_scale=1,
                    Data_log_flag=0, reset_flag_en=False, AI_result_flag=0,
                    Brightness=16, capture_no=1, ExposureTime='1/8s')
        for keys, expected in (({'Camera_or_BMP_flag': 0}, 0),
                               ({'Camera_or_png_flag': 1}, 1),
                               ({'Camera_or_png_flag': 0, 'Camera_or_BMP_flag': 1}, 0)):
            with patch.dict(namespace, read_mosa_data=lambda: dict(data, **keys),
                            camera_config=lambda value: value):
                self.app.load_json()
            self.assertEqual(self.app.Camera_or_png_flag, expected)


if __name__ == '__main__':
    unittest.main()

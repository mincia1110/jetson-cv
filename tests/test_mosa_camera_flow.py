"""Exercise actual Tk callback methods without a display, models or camera."""
import ast
from concurrent.futures import Future
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock


source = Path(__file__).resolve().parents[1] / 'MOSA_visualAD_comb_jetson.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
gui = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CameraGUI')
namespace = {'json': json, 'messagebox': Mock()}
exec(compile(ast.Module(body=[gui], type_ignores=[]), str(source), 'exec'), namespace)


class CameraFlowTests(unittest.TestCase):
    def setUp(self):
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


if __name__ == '__main__':
    unittest.main()

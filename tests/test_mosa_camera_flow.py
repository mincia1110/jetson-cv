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
camera_init = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'camera_init')
namespace = {'json': json, 'messagebox': Mock(), 'os': os, 'time': time,
             'datetime': datetime, 'threading': threading}
exec(compile(ast.Module(body=[camera_init, gui], type_ignores=[]), str(source), 'exec'), namespace)


class CameraFlowTests(unittest.TestCase):
    def setUp(self):
        namespace['messagebox'].reset_mock()
        namespace['messagebox'].askyesno.return_value = True
        self.app = namespace['CameraGUI'].__new__(namespace['CameraGUI'])
        self.app.running = True
        self.app._measurement_busy = True
        self.app._camera_future = Future()
        self.app._camera_recovery_started = threading.Event()
        self.app.camera = None
        self.app.capture_thread = None
        self.app.current_frame = None
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

    def test_file_camera_file_switch_waits_for_ready_and_reuses_camera_and_model(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare_file_capture(folder, mode=1)
            self.app._measurement_busy = False
            self.app._camera_future = None
            self.app.camera_settings = {'Camera_or_png_flag': 1}
            self.app.inferencer = inferencer = Mock()
            self.app._start_camera_preview = Mock()
            camera = Mock()
            camera.ready = Future()
            captured = Future()
            camera.prepare_inference.return_value = captured
            create_camera = Mock(return_value=(camera, None, False))
            with patch.dict(namespace, camera_init=create_camera):
                self.app.capture_image()
                args = self.app.root.after.call_args.args[2:]
                self.assertTrue(self.app._measurement_busy)
                self.app.capture_image()  # A second click cannot start another connection.
                self.app._poll_camera_start(*args)
                camera.prepare_inference.assert_not_called()
                self.app._start_camera_preview.assert_not_called()

                camera.ready.set_result({})
                self.app._poll_camera_start(*args)
                camera.prepare_inference.assert_called_once()
                self.app._start_camera_preview.assert_called_once()
                self.assertTrue(self.app._measurement_busy)

                frame = object()
                captured.set_result({'frame': frame, 'report': {'reset_performed': False}})
                self.app._poll_camera(*args)
                self.app._process_measurement.assert_called_once_with(*args, frame)
                self.assertFalse(self.app._measurement_busy)

                # The next click rereads the mode; file input does not use the camera.
                Path(folder, 'sample.png').touch()
                self.app.load_json.side_effect = lambda: setattr(self.app, 'Camera_or_png_flag', 0)
                self.app.capture_image()
                self.assertEqual(self.app._process_measurement.call_count, 2)
                self.assertEqual(camera.prepare_inference.call_count, 1)

                self.app.load_json.side_effect = lambda: setattr(self.app, 'Camera_or_png_flag', 1)
                self.app.capture_image()
                self.assertEqual(camera.prepare_inference.call_count, 2)
                create_camera.assert_called_once_with(self.app.camera_settings, wait_ready=False)
                self.assertIs(self.app.inferencer, inferencer)
                camera.close.assert_not_called()

    def test_failed_lazy_connection_allows_file_inference_and_connection_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare_file_capture(folder, mode=1)
            self.app._measurement_busy = False
            self.app._camera_future = None
            self.app.camera_settings = {'Camera_or_png_flag': 1}
            self.app._start_camera_preview = Mock()
            failed, retry = Mock(), Mock()
            failed.ready, retry.ready = Future(), Future()
            create_camera = Mock(side_effect=[(failed, None, False), (retry, None, False)])
            with patch.dict(namespace, camera_init=create_camera):
                self.app.capture_image()
                args = self.app.root.after.call_args.args[2:]
                failed.ready.set_exception(RuntimeError('device missing'))
                self.app._poll_camera_start(*args)
                failed.close.assert_called_once()
                self.assertIsNone(self.app.camera)
                self.assertIsNone(self.app._camera_future)
                self.assertFalse(self.app._measurement_busy)
                self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'FAIL')
                self.app._process_measurement.assert_not_called()
                self.app._start_camera_preview.assert_not_called()

                self.app.Camera_or_png_flag = 0
                Path(folder, 'sample.png').touch()
                self.app.capture_image()
                self.app._process_measurement.assert_called_once()

                self.app.Camera_or_png_flag = 1
                self.app.capture_image()
                self.assertIs(self.app.camera, retry)
                self.assertIs(self.app._camera_future, retry.ready)
                self.assertTrue(self.app._measurement_busy)

    def test_completed_results_update_without_preview_and_file_mode_does_not_reset_camera(self):
        self.app.Camera_or_png_flag = 0
        self.app.reset_flag = False
        self.app.reset_flag_en = True
        for result, color in (('OK', 'green'), ('NG', 'red')):
            self.app.measure(result, 180)
            self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], result)
            self.assertEqual(self.app.result_container.config.call_args.kwargs['bg'], color)
        self.app.root.after.assert_not_called()
        self.app.Camera_or_png_flag = 1
        self.app.measure('OK', 180)
        self.assertEqual(self.app.result_label.config.call_args.kwargs['text'], 'OK')

    def test_file_mode_close_without_camera_or_preview_releases_model(self):
        self.app.inferencer = Mock()
        self.app.on_close()
        self.assertFalse(self.app.running)
        self.app.inferencer.close.assert_called_once()
        self.app.root.destroy.assert_called_once()

    def test_close_during_lazy_connection_does_not_resume_measurement(self):
        self.app.camera = Mock()
        self.app.inferencer = Mock()
        self.app.on_close()
        self.app._poll_camera_start('sample', 'sample.png', 0)
        self.app.camera.close.assert_called_once()
        self.app.root.after.assert_not_called()
        self.app._process_measurement.assert_not_called()


class CameraStartupTests(unittest.TestCase):
    def test_file_startup_never_constructs_camera_or_validates_device_settings(self):
        for mode in ({'Camera_or_png_flag': 0}, {'Camera_or_BMP_flag': 0},
                     {'Camera_or_png_flag': 0, 'Camera_or_BMP_flag': 1}):
            with self.subTest(mode=mode), patch.dict(namespace,
                    read_mosa_data=Mock(return_value=mode), MOSACamera=Mock(), camera_config=Mock()):
                self.assertEqual(namespace['camera_init'](), (None, None, False))
                namespace['MOSACamera'].assert_not_called()
                namespace['camera_config'].assert_not_called()

    def test_camera_startup_preserves_ready_wait_and_failed_startup_cleanup(self):
        config = {'Camera_or_BMP_flag': 1, 'device': '/dev/video2', 'fps': 10}
        camera = Mock()
        with patch.dict(namespace, read_mosa_data=Mock(return_value=config),
                        MOSACamera=Mock(return_value=camera), camera_config=lambda data: data):
            self.assertEqual(namespace['camera_init'](), (camera, None, False))
            camera.ready.result.assert_called_once()
            namespace['MOSACamera'].assert_called_once_with(
                device='/dev/video2', fps=10, initial_config=config)
            camera.ready.result.side_effect = RuntimeError('device missing')
            with self.assertRaisesRegex(RuntimeError, 'device missing'):
                namespace['camera_init']()
            camera.close.assert_called_once()

    def test_lazy_startup_returns_pending_worker_without_waiting(self):
        config = {'Camera_or_png_flag': 1}
        camera = Mock()
        camera.ready = Future()
        with patch.dict(namespace, read_mosa_data=Mock(), MOSACamera=Mock(return_value=camera),
                        camera_config=lambda data: data):
            self.assertEqual(namespace['camera_init'](config, wait_ready=False), (camera, None, False))
            self.assertFalse(camera.ready.done())
            namespace['read_mosa_data'].assert_not_called()

    def test_gui_starts_preview_only_when_camera_exists_and_only_once(self):
        for camera in (None, Mock()):
            with self.subTest(camera=camera), patch.dict(namespace,
                    np=Mock(), torch=Mock(), tk=Mock(), ttk=Mock(), threading=Mock()), \
                    patch.object(namespace['CameraGUI'], 'load_json',
                                 lambda app: setattr(app, 'init_threshold', 0)), \
                    patch.object(namespace['CameraGUI'], '_update_label'), \
                    patch.object(namespace['CameraGUI'], 'update_gui') as update:
                app = namespace['CameraGUI'](Mock(), camera, None, Mock(), 'Model A')
                if camera is None:
                    namespace['threading'].Thread.assert_not_called()
                    update.assert_not_called()
                    self.assertIsNone(app.capture_thread)
                    app.video_label.config.assert_called_with(text='파일 추론 모드 (PNG/BMP)')
                    app.camera = Mock()  # A later successful connection can start preview.
                app._start_camera_preview()
                app._start_camera_preview()
                namespace['threading'].Thread.assert_called_once()
                app.capture_thread.start.assert_called_once()
                update.assert_called_once()

    def test_main_finally_closes_camera_connected_after_file_mode_start(self):
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        root, app, camera, inferencer = Mock(), Mock(), Mock(), Mock()
        app.camera = None

        def close_mainloop():
            app.camera = camera
            raise RuntimeError('mainloop stopped')

        root.mainloop.side_effect = close_mainloop
        values = {'__name__': '__main__', 'print': Mock(), 'tk': Mock(Tk=Mock(return_value=root)),
                  'show_model_selection_gui': Mock(return_value=('Model A', 'model.onnx', None, {})),
                  'MOSAInferencer': Mock(return_value=inferencer),
                  'camera_init': Mock(return_value=(None, None, False)),
                  'CameraGUI': Mock(return_value=app)}
        with self.assertRaisesRegex(RuntimeError, 'mainloop stopped'):
            exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), values)
        self.assertFalse(app.running)
        camera.close.assert_called_once()
        inferencer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

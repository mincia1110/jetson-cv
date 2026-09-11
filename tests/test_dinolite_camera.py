import importlib.util
from pathlib import Path
import threading
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


# Hardware-free import: tests exercise ownership and failures, not OpenCV itself.
spec = importlib.util.spec_from_file_location(
    'camera_under_test', Path(__file__).resolve().parents[1] / 'scripts/dinolite_camera.py')
module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'cv2': types.SimpleNamespace()}):
    spec.loader.exec_module(module)


class FakeCamera(module.DinoLiteCamera):
    def __init__(self):
        self.calls = []
        self.sequence = 0
        self.fail = False
        self.brightness = 0
        super().__init__(width=4, height=3)

    def record(self, kind):
        self.calls.append((kind, threading.get_ident()))

    def _open(self):
        self.record('open')
        self._cap = object()
        self._publish(np.zeros((3, 4, 3), dtype=np.uint8))
        if self._fixed_config is not None:
            self._apply_fixed(self._fixed_config)
        return {'width': 4, 'height': 3}

    def _read(self):
        self.record('read')
        if self.fail:
            raise RuntimeError('disconnected')
        self.sequence += 1
        return np.full((3, 4, 3), self.sequence % 200, dtype=np.uint8)

    def _release(self):
        self.record('release')
        self._cap = None

    def _command(self, args):
        self.record(tuple(args))
        if args[-1].startswith('--set-ctrl=brightness='):
            self.brightness = int(args[-1].split('=')[-1])
        if args[-1] == '--get-ctrl=brightness':
            return f'brightness: {self.brightness}'

    def _capture(self, count, roi):
        if roi == (900, 1100, 0, 2590):
            return self._read()
        return super()._capture(count, roi)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = FakeCamera()
        self.camera.ready.result(timeout=2)

    def tearDown(self):
        self.camera.close()
        self.camera._thread.join(timeout=2)
        self.assertFalse(self.camera.is_alive())

    def test_burst_preserves_first_frame_outside_roi_and_one_owner(self):
        result = self.camera.capture(3, (1, 2, 1, 3)).result(timeout=2)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[1, 1, 0]), int(result[0, 0, 0]) + 1)
        self.camera.reconnect().result(timeout=2)
        self.assertEqual(len({tid for _, tid in self.camera.calls}), 1)
        self.assertNotEqual(self.camera.calls[0][1], threading.get_ident())

    def test_ae_payload_and_remembered_mode(self):
        self.camera.set_auto_exposure(True).result(timeout=2)
        self.camera.set_auto_exposure(False).result(timeout=2)
        commands = [kind for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertEqual([cmd[-1] for cmd in commands], ['05000003357810', '05070003357810'])
        self.assertFalse(self.camera._auto_exposure)

    def test_failed_read_discards_old_preview_and_reconnect_recovers(self):
        self.camera.fail = True
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            self.camera.capture(1, (0, 1, 0, 1)).result(timeout=2)
        frame, error = self.camera.latest()
        self.assertIsNone(frame)
        self.assertIn('disconnected', error)
        self.camera.fail = False
        self.camera.reconnect().result(timeout=2)
        self.assertIsNotNone(self.camera.latest()[0])

    def test_closed_camera_rejects_new_work(self):
        self.camera.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.camera.capture().result(timeout=2)

    def test_condition_reset_once_and_recheck_failure(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'reset_flag_en': True, 'settle_frames': 2, 'capture_no': 1,
                  'Brightness': 16, 'ExposureTime': '1/60s'}
        bad = {'reasons': ['BRIGHT error']}
        with patch.object(self.camera, '_assess', side_effect=[bad, bad]) as assess:
            result = self.camera.check_conditions(config).result(timeout=2)
        self.assertEqual(assess.call_count, 2)
        self.assertTrue(result['reset_performed'])
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['exposure_verified'])
        self.assertEqual(self.camera.controls['brightness'], 16)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertNotIn('05000003357810', commands)
        self.assertEqual(commands.count('05080001357810'), 2)
        self.assertFalse(result['allow_inference'])

    def test_apply_exception_recovers_and_returns_fresh_frame(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'reset_flag_en': True, 'Brightness': 16, 'ExposureTime': '1/60s'}
        original = self.camera._apply_fixed
        attempts = []
        def apply(values):
            attempts.append(values)
            if len(attempts) == 1:
                raise RuntimeError('command failed')
            return original(values)
        with patch.object(self.camera, '_apply_fixed', side_effect=apply), patch.object(
                self.camera, '_assess', return_value={'reasons': []}):
            result = self.camera.prepare_inference(config).result(timeout=2)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(result['report']['reset_performed'])
        self.assertTrue(result['report']['allow_inference'])
        self.assertIsNotNone(result['frame'])

    def test_recovery_exception_blocks_inference_without_raising(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'reset_flag_en': True, 'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(self.camera, '_apply_fixed', side_effect=RuntimeError('USB error')):
            result = self.camera.prepare_inference(config).result(timeout=2)
        self.assertTrue(result['report']['reset_performed'])
        self.assertFalse(result['report']['allow_inference'])
        self.assertIsNone(result['frame'])
        self.assertIn('USB error', result['report']['after']['reasons'][0])

    def test_reset_disabled_and_good_result_do_not_reset(self):
        for reasons, enabled in [(['BRIGHT error'], False), ([], True)]:
            config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                      'reset_flag_en': enabled, 'Brightness': 16, 'ExposureTime': '1/60s'}
            with patch.object(self.camera, '_assess', return_value={'reasons': reasons}) as assess:
                result = self.camera.check_conditions(config).result(timeout=2)
            self.assertEqual(assess.call_count, 1)
            self.assertFalse(result['reset_performed'])

    def test_dll_exposure_rejected_without_disrupting_stream(self):
        with self.assertRaisesRegex(ValueError, 'DLL'):
            self.camera.check_conditions({'BRIGHT_min': 0, 'BRIGHT_max': 255,
                                          'RG_gab': 255, 'Brightness': 16, 'ExposureValue': 10})
        self.assertIsNotNone(self.camera.latest()[0])

    def test_recovered_image_does_not_claim_absolute_exposure_verified(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'reset_flag_en': True, 'settle_frames': 1,
                  'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(self.camera, '_assess', side_effect=[
                {'reasons': ['Brightness setting mismatch']}, {'reasons': []}]):
            result = self.camera.check_conditions(config).result(timeout=2)
        self.assertEqual(result['status'], 'IMAGE_AND_BRIGHTNESS_OK')
        self.assertFalse(result['exposure_verified'])

    def test_prepare_returns_only_the_assessed_frame_and_reapplies_each_time(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureTime': '1/125s'}
        with patch.object(self.camera, '_assess', return_value={'reasons': []}) as assess:
            first = self.camera.prepare_inference(config).result(timeout=2)
            self.assertIs(first['frame'], assess.call_args.args[1])
            self.camera.brightness = 99
            second = self.camera.prepare_inference(config).result(timeout=2)
        self.assertEqual(second['report']['brightness_before_apply'], 99)
        self.assertEqual(self.camera.brightness, 16)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertEqual(commands.count('05040001357810'), 2)

    def test_failed_conditions_never_provide_inference_frame(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(self.camera, '_assess', return_value={'reasons': ['BRIGHT error']}):
            result = self.camera.prepare_inference(config).result(timeout=2)
        self.assertIsNone(result['frame'])

    def test_brightness_write_failure_blocks_measurement(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(self.camera, '_brightness', return_value=99):
            result = self.camera.prepare_inference(config).result(timeout=2)
        self.assertIsNone(result['frame'])
        self.assertFalse(result['report']['allow_inference'])
        self.assertFalse(result['report']['reset_performed'])
        self.assertIn('Brightness 적용 실패', result['report']['before']['reasons'][0])


if __name__ == '__main__':
    unittest.main()

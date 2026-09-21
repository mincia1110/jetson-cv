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
    def _rediscover_device(self):
        self.record('rediscover')

    def __init__(self):
        self.calls = []
        self.sequence = 0
        self.fail = False
        self.brightness = 0
        self.video_values = {}
        super().__init__(width=4, height=3)

    def record(self, kind):
        self.calls.append((kind, threading.get_ident()))

    def _open(self):
        self._windows_initialized = False
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
        if args[-1].startswith('--set-ctrl='):
            name, value = args[-1].split('=', 1)[1].split('=')
            self.video_values[name] = int(value)
        if args[-1].startswith('--get-ctrl='):
            name = args[-1].split('=', 1)[1]
            return f'{name}: {self.video_values[name]}'

    def _capture(self, count, roi):
        if roi == (900, 1100, 0, 2590):
            return self._read()
        return super()._capture(count, roi)


class CameraTests(unittest.TestCase):
    def test_rediscovery_updates_device_and_handles_missing_or_ambiguous_camera(self):
        camera = object.__new__(module.DinoLiteCamera)
        camera.device = '/dev/video0'
        camera._stop = threading.Event()
        with patch.object(camera, '_dinolite_nodes', return_value=['/dev/video2']), patch.object(module.Path, 'exists', return_value=True):
            camera._rediscover_device()
        self.assertEqual(camera.device, '/dev/video2')
        with patch.object(camera, '_dinolite_nodes', return_value=[]), patch.object(module.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                camera._rediscover_device()
        with patch.object(camera, '_dinolite_nodes', return_value=['/dev/video0', '/dev/video2']):
            with self.assertRaisesRegex(RuntimeError, 'Multiple'):
                camera._rediscover_device()

    def test_discovery_filters_usb_identity_and_metadata(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usb = root / 'usb'
            usb.mkdir()
            (usb / 'idVendor').write_text('a168')
            (usb / 'idProduct').write_text('0960')
            video = usb / 'video4linux'
            video.mkdir()
            for name, index in [('video2', '0'), ('video3', '1')]:
                entry = video / name
                entry.mkdir()
                (entry / 'index').write_text(index)
            self.assertEqual(module.DinoLiteCamera._dinolite_nodes(video), ['/dev/video2'])
            (usb / 'idVendor').write_text('1234')
            self.assertEqual(module.DinoLiteCamera._dinolite_nodes(video), [])

    def test_stream_restart_primes_twice_and_releases_on_failure(self):
        from unittest.mock import MagicMock
        camera = object.__new__(module.DinoLiteCamera)
        camera.width, camera.height, camera.fps = 2592, 1944, 10
        camera.device = '/dev/video0'
        camera._stop = threading.Event()
        camera._cap = None
        fake_cv = types.SimpleNamespace(CAP_V4L2=200, CAP_PROP_FOURCC=6,
                    CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5,
                    VideoWriter_fourcc=lambda *args: 123)
        caps = [MagicMock(), MagicMock()]
        for cap in caps:
            cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            cap.get.side_effect = lambda prop: 123 if prop == 6 else 30
        fake_cv.VideoCapture = MagicMock(side_effect=caps)
        with patch.object(module, 'cv2', fake_cv), patch.object(module.time, 'sleep'):
            camera._prime_windows_streams()
        for cap in caps:
            self.assertEqual(cap.read.call_count, 9)
            cap.release.assert_called_once()
            self.assertEqual([c.args for c in cap.set.call_args_list],
                             [(6, 123), (3, 640), (4, 480), (5, 30)])
        self.assertIsNone(camera._cap)
        bad = MagicMock()
        bad.read.return_value = (False, None)
        fake_cv.VideoCapture = MagicMock(return_value=bad)
        with patch.object(module, 'cv2', fake_cv), self.assertRaisesRegex(RuntimeError, 'expected 640x480'):
            camera._prime_windows_streams()
        bad.release.assert_called_once()
        self.assertIsNone(camera._cap)

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

    def test_full_profile_replays_unknown_writes_only_on_initialization(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureValue': 800,
                  'camera_profile': 'windows_800_full'}
        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_assess', return_value={'reasons': []}):
            self.camera.apply_initial(config).result(timeout=2)
            self.camera.prepare_inference(config).result(timeout=2)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertEqual(commands.count('05000000004600'), 1)
        self.assertEqual(commands.count('f2010000000000'), 1)
        self.assertEqual(commands.count('05320001357810'), 2)
        with self.assertRaises(ValueError):
            self.camera.apply_initial(dict(config, Brightness=20))

    def test_control_trials_remove_only_selected_writes_and_preserve_retry(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 18,
                  'Brightness': 16, 'camera_profile': 'windows_800_full'}
        for stage, awb_count in [('no_post_writes', 1), ('no_added_awb', 0)]:
            with self.subTest(stage=stage):
                self.camera.video_values.update(white_balance_automatic=0, power_line_frequency=2)
                self.camera._windows_initialized = False
                self.camera.calls.clear()
                selected = dict(config, windows_control_trial=stage)
                with patch.object(module.time, 'sleep'):
                    self.camera.apply_initial(selected).result(timeout=2)
                commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
                self.assertEqual(commands.count('--set-ctrl=brightness=16'), 1)
                self.assertEqual(commands.count('--set-ctrl=white_balance_temperature=5800'), 1)
                self.assertEqual(commands.count('--set-ctrl=white_balance_automatic=0'), awb_count)
                self.assertNotIn('--set-ctrl=power_line_frequency=2', commands)
                self.assertEqual(commands.count('05000000004600'), 1)
                self.camera.calls.clear()
                self.camera.apply_initial(selected).result(timeout=2)
                self.assertFalse(any(isinstance(kind, tuple) for kind, _ in self.camera.calls))
                with patch.object(module.time, 'sleep'):
                    self.camera.reconnect().result(timeout=2)
                commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
                self.assertEqual(commands.count('05000000004600'), 1)
                self.assertEqual(commands.count('--set-ctrl=brightness=16'), 1)

    def test_control_trial_mismatch_does_not_auto_correct(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 18,
                  'Brightness': 16, 'camera_profile': 'windows_800_full',
                  'windows_control_trial': 'no_post_writes'}
        self.camera.video_values['power_line_frequency'] = 1
        with patch.object(module.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'power_line_frequency readback mismatch'):
                self.camera.apply_initial(config).result(timeout=2)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertNotIn('--set-ctrl=power_line_frequency=2', commands)
        self.assertFalse(self.camera._windows_initialized)

    def test_usb_reconnect_awb_is_disabled_only_immediately_before_wb(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 18,
                  'Brightness': 16, 'camera_profile': 'windows_800_full',
                  'windows_control_trial': 'no_added_awb'}
        self.camera.video_values.update(white_balance_automatic=1, power_line_frequency=2)
        original = self.camera._command

        def enforce_inactive(args):
            if args[-1] == '--set-ctrl=white_balance_temperature=5800':
                if self.camera.video_values['white_balance_automatic']:
                    raise RuntimeError('white_balance_temperature: Permission denied')
            return original(args)

        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_command', side_effect=enforce_inactive):
            self.camera.apply_initial(config).result(timeout=2)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        off = '--set-ctrl=white_balance_automatic=0'
        self.assertEqual(commands.count(off), 1)
        position = commands.index(off)
        self.assertGreater(position, commands.index('--set-ctrl=gamma=5'))
        self.assertEqual(commands[position + 1], '--get-ctrl=white_balance_automatic')
        self.assertEqual(commands[position + 2], '--set-ctrl=white_balance_temperature=5800')

    def test_awb_refuses_to_disable_blocks_manual_wb(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 18,
                  'Brightness': 16, 'camera_profile': 'windows_800_full',
                  'windows_control_trial': 'no_added_awb'}
        self.camera.video_values.update(white_balance_automatic=1, power_line_frequency=2)
        original = self.camera._command

        def stuck_awb(args):
            result = original(args)
            if args[-1] == '--set-ctrl=white_balance_automatic=0':
                self.camera.video_values['white_balance_automatic'] = 1
            return result

        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_command', side_effect=stuck_awb):
            with self.assertRaisesRegex(RuntimeError, 'AWB remained ON'):
                self.camera.apply_initial(config).result(timeout=2)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        self.assertNotIn('--set-ctrl=white_balance_temperature=5800', commands)

    def test_windows_profile_sequence_no_time_overwrite_and_ae_only_on_reset(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureValue': 800, 'ExposureTime': '1/8s',
                  'camera_profile': 'windows_800'}
        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_assess', return_value={'reasons': []}):
            self.camera.apply_initial(config).result(timeout=2)
            self.camera.prepare_inference(config).result(timeout=2)
            commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
            self.assertEqual(commands.count('05000003357810'), 1)
            self.assertEqual(commands.count('05320001357810'), 2)
            self.assertNotIn('051f0001357810', commands)
            seq = ['0502000c357810', '0525000d357810', '05000000357810',
                   '05320001357810', '05000002357810']
            indices = [commands.index(command) for command in seq]
            self.assertEqual(indices, sorted(indices))
            self.camera.reconnect().result(timeout=2)
            commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
            self.assertEqual(commands.count('05000003357810'), 2)
        self.assertEqual(self.camera._fixed_config['ExposureTime'], 'DNX64:800')
        with self.assertRaises(ValueError):
            self.camera.apply_initial(dict(config, ExposureValue=400))

    def test_manual_white_balance_order_and_reconnect(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureTime': '1/60s',
                  'video_controls': {'white_balance_temperature': 5800,
                                     'white_balance_automatic': 0, 'gamma': 5}}
        self.camera.apply_initial(config).result(timeout=2)
        self.camera.reconnect().result(timeout=2)
        commands = [kind[-1] for kind, _ in self.camera.calls if isinstance(kind, tuple)]
        awb = '--set-ctrl=white_balance_automatic=0'
        temp = '--set-ctrl=white_balance_temperature=5800'
        self.assertLess(commands.index(awb), commands.index(temp))
        self.assertEqual(commands.count(awb), 2)
        self.assertEqual(commands.count(temp), 2)
        self.assertEqual(self.camera.video_values['gamma'], 5)

    def test_windows_manual_retry_preserves_post_reset_state(self):
        for profile in ('windows_800', 'windows_800_full'):
            with self.subTest(profile=profile), patch.object(module.time, 'sleep') as sleep:
                config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 10,
                          'reset_flag_en': True, 'Brightness': 16,
                          'camera_profile': profile}
                # Startup is separate from the failed measurement being tested.
                self.camera._windows_initialized = False
                self.camera.apply_initial(config).result(timeout=2)
                self.camera.calls.clear()
                bad = {'RG_diff': 18, 'reasons': ['RG_gab error']}
                with patch.object(self.camera, '_assess', side_effect=[bad, bad]):
                    result = self.camera.prepare_inference(
                        config, manual_retry=True,
                        on_recovery=lambda: self.camera.record('recovery_started')).result(timeout=2)
                report = result['report']
                self.assertIsNone(result['frame'])
                self.assertTrue(report['reset_succeeded'])
                self.assertTrue(report['manual_retry_required'])
                self.assertFalse(report['allow_inference'])
                self.assertFalse(report['image_conditions_ok'])
                self.assertEqual(report['settings_action'], 'preserved')
                self.assertEqual(report['after']['RG_diff'], 18)
                calls = [kind for kind, _ in self.camera.calls]
                self.assertLess(calls.index('recovery_started'), calls.index('release'))
                self.assertLess(calls.index('release'), calls.index('open'))
                self.assertEqual(calls.count('open'), 1)
                sleep.assert_any_call(1.0)
                commands = [kind[-1] for kind in calls if isinstance(kind, tuple)]
                self.assertEqual(commands.count('05320001357810'), 1)
                if profile == 'windows_800_full':
                    self.assertIn('05000000004600', commands)
                else:
                    self.assertEqual(commands.count('05000003357810'), 1)

                # The next operator click checks a new frame without undoing
                # reset state. Changing only a threshold must not rewrite WB.
                self.camera.calls.clear()
                with patch.object(self.camera, '_assess', return_value={'RG_diff': 5, 'reasons': []}):
                    retry = self.camera.prepare_inference(
                        dict(config, RG_gab=12), manual_retry=True).result(timeout=2)
                self.assertIsNotNone(retry['frame'])
                self.assertTrue(retry['report']['allow_inference'])
                self.assertFalse(retry['report']['reset_performed'])
                commands = [kind for kind, _ in self.camera.calls if isinstance(kind, tuple)]
                self.assertFalse(any('-S' in cmd or '--set-ctrl' in ' '.join(cmd) for cmd in commands))

    def test_manual_recovery_pass_still_requires_a_new_click(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 10,
                  'reset_flag_en': True, 'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_assess', side_effect=[
                {'reasons': ['RG_gab error']}, {'reasons': []}]):
            result = self.camera.prepare_inference(config, manual_retry=True).result(timeout=2)
        self.assertTrue(result['report']['image_conditions_ok'])
        self.assertEqual(result['report']['status'], 'RESET_DONE_RETRY_REQUIRED')
        self.assertFalse(result['report']['allow_inference'])
        self.assertIsNone(result['frame'])

    def test_windows_changed_brightness_is_applied_on_next_click(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 10,
                  'Brightness': 16, 'camera_profile': 'windows_800'}
        with patch.object(module.time, 'sleep'), patch.object(self.camera, '_assess', return_value={'reasons': []}):
            self.camera.apply_initial(config).result(timeout=2)
            result = self.camera.prepare_inference(dict(config, Brightness=20), manual_retry=True).result(timeout=2)
        self.assertEqual(self.camera.brightness, 20)
        self.assertEqual(result['report']['settings_action'], 'reapplied')

    def test_manual_reconnect_error_does_not_claim_ready(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 10,
                  'reset_flag_en': True, 'Brightness': 16, 'ExposureTime': '1/60s'}
        with patch.object(self.camera, '_assess', return_value={'reasons': ['RG_gab error']}), patch.object(
                self.camera, '_reconnect', side_effect=RuntimeError('device missing')):
            result = self.camera.prepare_inference(config, manual_retry=True).result(timeout=2)
        self.assertFalse(result['report']['reset_succeeded'])
        self.assertFalse(result['report']['manual_retry_required'])
        self.assertFalse(result['report']['allow_inference'])
        self.assertEqual(result['report']['status'], 'FAIL')
        self.assertIsNone(result['frame'])

    def test_video_controls_reject_invalid_and_inactive_settings(self):
        base = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                'Brightness': 16, 'ExposureTime': '1/60s'}
        for controls in ({'gamma': 13}, {'gain': 0},
                         {'white_balance_temperature': 5800}):
            with self.assertRaises(ValueError):
                self.camera.apply_initial(dict(base, video_controls=controls))

    def test_video_readback_mismatch_is_failure(self):
        config = {'BRIGHT_min': 0, 'BRIGHT_max': 255, 'RG_gab': 255,
                  'Brightness': 16, 'ExposureTime': '1/60s', 'video_controls': {'gamma': 5}}
        with patch.object(self.camera, '_video_control_values', return_value={'gamma': 6}):
            result = self.camera.prepare_inference(config).result(timeout=2)
        self.assertIsNone(result['frame'])
        self.assertIn('gamma', result['report']['before']['reasons'][0])

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

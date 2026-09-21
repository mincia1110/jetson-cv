"""Single-owner V4L2 camera worker. No DNX64 DLL or inference dependencies."""
from concurrent.futures import Future
import queue
import subprocess
import threading
import time
import json
from pathlib import Path

import cv2
import numpy as np

from camera_conditions import EXPOSURE_COMMANDS, assess_frame, validate_conditions


class DinoLiteCamera:
    def __init__(self, device='/dev/video0', width=2592, height=1944, fps=10,
                 controls=None, warmup_frames=5, initial_config=None):
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.controls = dict(controls or {})
        self.warmup_frames = warmup_frames
        self._requests = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._cap = None
        self._closed = False
        self._windows_initialized = False
        self._auto_exposure = None  # Preserve startup state until explicitly selected.
        self._fixed_config = validate_conditions(initial_config) if initial_config is not None else None
        self.ready = Future()
        self._thread = threading.Thread(target=self._run, daemon=True, name='dinolite')
        self._thread.start()

    def latest(self):
        """Return (BGR frame copy or None, error text or None)."""
        with self._lock:
            return (None if self._latest is None else self._latest.copy(), self._error)

    def _publish(self, frame=None, error=None):
        with self._lock:
            self._latest, self._error = frame, error

    def _command(self, args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or str(args))
        return result.stdout.strip()

    def _led(self, enabled):
        # AM7115MZT a168:0960: user verified these two payloads on hardware.
        self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2',
                       'f2010000000000' if enabled else 'f2000000000000'])

    def _ae(self, enabled):
        # User verified AE on -> change light -> off -> frozen exposure.
        self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2',
                       '05000003357810' if enabled else '05070003357810'])
        self._auto_exposure = enabled

    def _read(self):
        if self._stop.is_set():
            raise RuntimeError('Camera stopping')
        if self._cap is None:
            raise RuntimeError('Camera unavailable; reconnect first')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError('Camera frame acquisition failed; reconnect the camera')
        if frame.shape[:2] != (self.height, self.width):
            raise RuntimeError(f'Unexpected frame size {frame.shape}; requested '
                               f'{self.width}x{self.height}')
        return frame

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open(self):
        self._windows_initialized = False
        self._publish()
        self._release()
        restart = bool(self._fixed_config and self._fixed_config.get('windows_stream_restart'))
        if restart:
            self._prime_windows_streams()
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open {self.device}')
        for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                            (cv2.CAP_PROP_FPS, self.fps)):
            self._cap.set(prop, value)
        self._read()  # Start streaming before the vendor command.
        if restart:
            if (int(self._cap.get(cv2.CAP_PROP_FOURCC)) != cv2.VideoWriter_fourcc(*'MJPG')
                    or abs(self._cap.get(cv2.CAP_PROP_FPS) - 10) > 0.1):
                raise RuntimeError('Windows reset requires final MJPG 2592x1944 @ 10 fps')
            time.sleep(0.68)
        self._led(False)
        if self._auto_exposure is not None and self._fixed_config is None:
            self._ae(self._auto_exposure)
        controls = dict(self.controls)
        if self._fixed_config is not None:
            # Apply these once, in the profile's post-exposure order below.
            for name in ('brightness', *self._fixed_config.get('video_controls', {})):
                controls.pop(name, None)
        if controls:
            # Use Linux control names and validated Linux values, never DLL enum IDs.
            self._command(['v4l2-ctl', '-d', self.device, '--set-ctrl',
                           ','.join(f'{key}={int(value)}' for key, value in controls.items())])
        if self._fixed_config is not None:
            self._apply_fixed(self._fixed_config)
        frame = None
        for _ in range(max(1, self.warmup_frames)):
            frame = self._read()
        self._publish(frame)
        return {'width': frame.shape[1], 'height': frame.shape[0],
                'fps': self._cap.get(cv2.CAP_PROP_FPS)}

    def _prime_windows_streams(self):
        """Opt-in V4L2 approximation of successful Windows reset negotiation.

        Capture frames 8156/8212: YUY2 640x480 @30; 8268: MJPG @10.
        Windows YUY2 is Linux YUYV. Never replay raw USB SET_INTERFACE while
        uvcvideo owns the device. Intermediate frames must not reach the GUI.
        """
        if (self.width, self.height, self.fps) != (2592, 1944, 10):
            raise ValueError('windows_stream_restart requires 2592x1944 @ 10 fps')
        for stage in (1, 2):
            try:
                self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                if not self._cap.isOpened():
                    raise RuntimeError(f'Cannot open {self.device} for reset stage {stage}')
                for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')),
                                    (cv2.CAP_PROP_FRAME_WIDTH, 640),
                                    (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                                    (cv2.CAP_PROP_FPS, 30)):
                    self._cap.set(prop, value)
                # About 0.28 seconds streaming per intermediate Windows stage.
                for _ in range(9):
                    if self._stop.is_set():
                        raise RuntimeError('Camera stopping')
                    ok, frame = self._cap.read()
                    if not ok or frame is None or frame.shape[:2] != (480, 640):
                        raise RuntimeError(f'Windows reset stage {stage}: expected 640x480 frame')
                if (int(self._cap.get(cv2.CAP_PROP_FOURCC)) != cv2.VideoWriter_fourcc(*'YUYV')
                        or abs(self._cap.get(cv2.CAP_PROP_FPS) - 30) > 0.1):
                    raise RuntimeError(f'Windows reset stage {stage}: YUYV @30 unavailable')
                print(f'[camera reset] stream {stage}/3: YUYV 640x480 @30', flush=True)
            finally:
                self._release()
            time.sleep(0.72)
        print('[camera reset] stream 3/3: opening MJPG 2592x1944 @10', flush=True)

    def _capture(self, count, roi):
        # All read() calls live in this worker, including the preview and burst.
        if not isinstance(count, int) or not 1 <= count <= 100:
            raise ValueError('capture count must be an integer from 1 to 100')
        y0, y1, x0, x1 = roi
        if not (0 <= y0 < y1 <= self.height and 0 <= x0 < x1 <= self.width):
            raise ValueError(f'Invalid ROI: {roi}')
        first = None
        total = None
        for _ in range(count):
            frame = self._read()
            if first is None:
                first = frame.copy()
                total = np.zeros(frame[y0:y1, x0:x1].shape, dtype=np.float32)
            total += frame[y0:y1, x0:x1]
            self._publish(frame)
        # Preserve test.py's behavior: average ROI, keep first frame outside ROI.
        first[y0:y1, x0:x1] = (total / count).astype(np.uint8)
        return first

    def _reconnect(self):
        # Windows reset_camera releases the stream and waits one second before
        # reopening. Keep this on the camera worker, including the closed period.
        self._publish()
        self._release()
        time.sleep(1.0)
        return self._open()

    def _submit(self, action, *args):
        future = Future()
        with self._lock:
            if self._closed:
                future.set_exception(RuntimeError('Camera closed'))
            else:
                self._requests.put((future, action, args))
        return future

    def capture(self, count=1, roi=(900, 1100, 0, 2590)):
        return self._submit('capture', count, roi)

    def reconnect(self):
        """Reopen stream and reapply settings; this is not a USB/factory reset."""
        return self._submit('reconnect')

    def set_led(self, enabled):
        return self._submit('led', bool(enabled))

    def set_auto_exposure(self, enabled):
        return self._submit('ae', bool(enabled))

    def check_conditions(self, config):
        # Validate before queuing: invalid settings must not disrupt streaming.
        return self._submit('check', validate_conditions(config))

    def apply_initial(self, config):
        return self._submit('initial', validate_conditions(config))

    def prepare_inference(self, config, *, manual_retry=False, on_recovery=None):
        """Future -> {frame, report}; failed attempts never supply a frame.

        MOSA uses manual_retry: after recovery the operator must measure again.
        on_recovery runs on the worker; it must not call Tk APIs.
        """
        return self._submit('prepare', validate_conditions(config), manual_retry, on_recovery)

    def _replay_windows_capture(self, config):
        path = Path(__file__).with_name('windows_800_capture.json')
        sequence = json.loads(path.read_text(encoding='utf-8'))
        # Set the AWB state observed by GET_CUR in the Windows capture.
        if config['windows_control_trial'] != 'no_added_awb':
            self._command(['v4l2-ctl', '-d', self.device, '--set-ctrl=white_balance_automatic=0'])
        for item in sequence:
            time.sleep(item['delay_s'])
            if item['kind'] == 'xu':
                self._command(['uvcdynctrl', '-d', self.device, '-S',
                               f"4:{item['selector']}", item['payload']])
            else:
                self._command(['v4l2-ctl', '-d', self.device,
                               f"--set-ctrl={item['control']}={item['value']}"])
        self._auto_exposure = False

    def _apply_fixed(self, config):
        if self._cap is None:
            raise RuntimeError('Camera unavailable; reconnect first')
        trial = config.get('windows_control_trial', 'baseline')
        if self._windows_initialized and trial != 'baseline':
            if not self._preserve_windows_settings(config, True):
                raise RuntimeError('Camera trial settings changed; restart GUI before comparison')
            # Diagnostic callers must not silently reintroduce writes on retry.
            return {'mode': 'preserved', 'windows_control_trial': trial}
        self._led(False)
        full_replay = config.get('camera_profile') == 'windows_800_full' and not self._windows_initialized
        if full_replay:
            print(f'[camera reset] windows_control_trial={trial}', flush=True)
            self._replay_windows_capture(config)
        elif config.get('camera_profile') in ('windows_800', 'windows_800_full'):
            initializing = not self._windows_initialized
            if initializing:
                self._ae(True)
                time.sleep(0.5)  # Supplied Windows COMMAND_TIME, not capture timing.
            self._ae(False)
            for payload in ('0502000c357810', '0525000d357810', '05000000357810',
                            '05320001357810', '05000002357810'):
                self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2', payload])
                time.sleep(0.04)  # Captured write spacing approximately 35–38 ms.
            time.sleep(0.5)
            if initializing:
                # AE target writes observed after exposure in the supplied capture.
                for payload in ('0510001e3a7810', '0530001b3a7810',
                                '051200103a7810', '051e000f3a7810'):
                    self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2', payload])
                    time.sleep(0.04)
                # Known FLC level1 and all quadrants OFF; keep LED master OFF.
                for payload in ('f3010000000000', '05010003006200', '05100004006200'):
                    self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2', payload])
                    time.sleep(0.04)
        else:
            self._windows_initialized = False
            self._ae(False)
            self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2',
                           EXPOSURE_COMMANDS[config['ExposureTime']]])
        skip_post = full_replay and trial != 'baseline'
        if not skip_post:
            self._command(['v4l2-ctl', '-d', self.device,
                           f"--set-ctrl=brightness={config['Brightness']}"])
        actual = self._brightness()
        if actual != config['Brightness']:
            raise RuntimeError(f"Brightness 적용 실패: target={config['Brightness']}, actual={actual}")
        if skip_post:
            actual_controls = self._video_control_values(config)
            print(f'[camera reset] post replay readback only: {actual_controls}', flush=True)
            for name, requested in config['video_controls'].items():
                if actual_controls[name] != requested:
                    raise RuntimeError(f'{name} readback mismatch: target={requested}, '
                                       f'actual={actual_controls[name]}; trial does not correct controls')
        else:
            self._apply_video_controls(config)
        self._fixed_config = dict(config)
        self.controls['brightness'] = config['Brightness']
        for _ in range(config['settle_frames']):
            self._publish(self._read())
        self._windows_initialized = config.get('camera_profile') in ('windows_800', 'windows_800_full')
        return {'Brightness': actual, 'ExposureTime_requested': config['ExposureTime'],
                'exposure_readback': 'unavailable', 'mode': 'fixed'}

    def _video_control_values(self, config):
        values = {}
        for name in config.get('video_controls', {}):
            output = self._command(['v4l2-ctl', '-d', self.device, f'--get-ctrl={name}'])
            values[name] = int(output.rsplit(':', 1)[1].strip())
        return values

    def _apply_video_controls(self, config):
        controls = config.get('video_controls', {})
        # Separate commands ensure AWB is disabled before manual temperature.
        order = ('contrast', 'hue', 'saturation', 'sharpness', 'gamma',
                 'white_balance_automatic', 'white_balance_temperature', 'power_line_frequency')
        names = [name for name in order if name in controls]
        for name in names:
            self._command(['v4l2-ctl', '-d', self.device,
                           f'--set-ctrl={name}={controls[name]}'])
        actual = self._video_control_values(config)
        for name, requested in controls.items():
            if actual[name] != requested:
                raise RuntimeError(f'{name} 적용 실패: target={requested}, actual={actual[name]}')

    def _brightness(self):
        output = self._command(['v4l2-ctl', '-d', self.device, '--get-ctrl=brightness'])
        return int(output.rsplit(':', 1)[1].strip())

    def _assess(self, config, frame):
        report = assess_frame(frame, config)
        report['brightness_control'] = self._brightness()
        if config.get('Brightness') is not None and report['brightness_control'] != config['Brightness']:
            report['reasons'].append('Brightness setting mismatch')
        report['video_controls'] = self._video_control_values(config)
        for name, requested in config.get('video_controls', {}).items():
            if report['video_controls'][name] != requested:
                report['reasons'].append(f'{name} setting mismatch')
        report['ae_last_command'] = self._auto_exposure
        report['exposure_readback'] = 'unavailable: AE command state is not shutter readback'
        if self._auto_exposure is True:
            report['reasons'].append('AE ON: exposure is not frozen')
        return report

    def _preserve_windows_settings(self, config, manual_retry):
        # Rewriting exposure/WB on the next click can change the post-reset
        # state. In MOSA's Windows mode only write when requested values change.
        if not manual_retry or not self._windows_initialized or self._fixed_config is None:
            return False
        if config['camera_profile'] not in ('windows_800', 'windows_800_full'):
            return False
        keys = ('camera_profile', 'ExposureTime', 'Brightness', 'video_controls',
                'windows_stream_restart', 'windows_control_trial')
        return all(config.get(key) == self._fixed_config.get(key) for key in keys)

    def _check_conditions(self, config, manual_retry=False, on_recovery=None, include_frame=False):
        report = {'camera_profile': config.get('camera_profile', 'fixed'), 'before': None, 'reset_performed': False, 'after': None,
                  'windows_stream_restart': config.get('windows_stream_restart', False),
                  'windows_control_trial': config.get('windows_control_trial', 'baseline'),
                  'brightness_before_apply': None, 'fixed_settings_applied': False,
                  'ExposureTime_requested': config['ExposureTime'],
                  'exposure_mode': 'fixed', 'exposure_verified': False,
                  'recovery_enabled': config['reset_flag_en'],
                  'settings_action': 'not_applied', 'reset_succeeded': False,
                  'manual_retry_required': False}
        frame = None
        try:
            report['brightness_before_apply'] = self._brightness()
            if self._preserve_windows_settings(config, manual_retry):
                report['settings_action'] = 'preserved'
            else:
                # Legacy fixed-time mode still reapplies on every measurement.
                self._apply_fixed(config)
                report['fixed_settings_applied'] = True
                report['settings_action'] = 'reapplied'
            frame = self._capture(config['capture_no'], (900, 1100, 0, 2590))
            before = self._assess(config, frame)
        except Exception as exc:
            before = {'reasons': [f'{type(exc).__name__}: {exc}']}
        report['before'] = before
        if before['reasons'] and config['reset_flag_en']:
            report['reset_performed'] = True
            try:
                if on_recovery is not None:
                    on_recovery()
                # Save the requested values even if their first application failed.
                self._fixed_config = dict(config)
                self.controls['brightness'] = config['Brightness']
                self._reconnect()  # Close, wait, fully initialize, drain old frames.
                report['fixed_settings_applied'] = True
                frame = self._capture(config['capture_no'], (900, 1100, 0, 2590))
                report['after'] = self._assess(config, frame)
                report['reset_succeeded'] = True
                report['manual_retry_required'] = manual_retry
            except Exception as exc:
                self._release()
                self._publish(error=str(exc))
                report['after'] = {'reasons': [f'{type(exc).__name__}: {exc}']}
        final = report['after'] or before
        report['image_conditions_ok'] = not final['reasons']
        report['status'] = ('RESET_DONE_RETRY_REQUIRED' if report['manual_retry_required']
                            else 'FAIL' if final['reasons'] else 'IMAGE_AND_BRIGHTNESS_OK')
        report['allow_inference'] = not final['reasons'] and not report['manual_retry_required']
        if include_frame:
            return {'frame': frame if report['allow_inference'] else None, 'report': report}
        return report

    def close(self):
        """Request shutdown without blocking the GUI; worker releases the device."""
        with self._lock:
            self._closed = True
            self._stop.set()

    def is_alive(self):
        return self._thread.is_alive()

    def _run(self):
        try:
            try:
                self.ready.set_result(self._open())
            except Exception as exc:
                self._release()
                self._publish(error=str(exc))
                self.ready.set_exception(exc)
            while not self._stop.is_set():
                try:
                    future, action, args = self._requests.get(timeout=0.01)
                except queue.Empty:
                    if self._cap is not None:
                        try:
                            self._publish(self._read())
                        except Exception as exc:
                            self._release()
                            self._publish(error=str(exc))
                    continue
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if action == 'reconnect':
                        result = self._reconnect()
                    elif action == 'capture':
                        result = self._capture(*args)
                    elif action == 'check':
                        result = self._check_conditions(*args)
                    elif action == 'prepare':
                        result = self._check_conditions(*args, include_frame=True)
                    elif action == 'initial':
                        result = self._apply_fixed(*args)
                    else:
                        if self._cap is None:
                            raise RuntimeError('Camera unavailable; reconnect first')
                        result = self._ae(*args) if action == 'ae' else self._led(*args)
                    future.set_result(result)
                except Exception as exc:
                    self._release()
                    self._publish(error=str(exc))
                    future.set_exception(exc)
        finally:
            self._release()
            self._publish(error='Camera closed')
            while True:
                try:
                    future, _, _ = self._requests.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(RuntimeError('Camera closed'))

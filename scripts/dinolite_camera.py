"""Single-owner V4L2 camera worker. No DNX64 DLL or inference dependencies."""
from concurrent.futures import Future
import queue
import subprocess
import threading

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
        self._publish()
        self._release()
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open {self.device}')
        for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                            (cv2.CAP_PROP_FPS, self.fps)):
            self._cap.set(prop, value)
        self._read()  # Start streaming before the vendor command.
        self._led(False)
        if self._auto_exposure is not None and self._fixed_config is None:
            self._ae(self._auto_exposure)
        if self.controls:
            # Use Linux control names and validated Linux values, never DLL enum IDs.
            self._command(['v4l2-ctl', '-d', self.device, '--set-ctrl',
                           ','.join(f'{key}={int(value)}' for key, value in self.controls.items())])
        if self._fixed_config is not None:
            self._apply_fixed(self._fixed_config)
        frame = None
        for _ in range(max(1, self.warmup_frames)):
            frame = self._read()
        self._publish(frame)
        return {'width': frame.shape[1], 'height': frame.shape[0],
                'fps': self._cap.get(cv2.CAP_PROP_FPS)}

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

    def prepare_inference(self, config):
        """Future -> {frame, report}. frame is None on failed image checks."""
        return self._submit('prepare', validate_conditions(config))

    def _apply_fixed(self, config):
        if self._cap is None:
            raise RuntimeError('Camera unavailable; reconnect first')
        self._led(False)
        self._ae(False)
        self._command(['uvcdynctrl', '-d', self.device, '-S', '4:2',
                       EXPOSURE_COMMANDS[config['ExposureTime']]])
        self._command(['v4l2-ctl', '-d', self.device,
                       f"--set-ctrl=brightness={config['Brightness']}"])
        actual = self._brightness()
        if actual != config['Brightness']:
            raise RuntimeError(f"Brightness 적용 실패: target={config['Brightness']}, actual={actual}")
        self._fixed_config = dict(config)
        self.controls['brightness'] = config['Brightness']
        for _ in range(config['settle_frames']):
            self._publish(self._read())
        return {'Brightness': actual, 'ExposureTime_requested': config['ExposureTime'],
                'exposure_readback': 'unavailable', 'mode': 'fixed'}

    def _brightness(self):
        output = self._command(['v4l2-ctl', '-d', self.device, '--get-ctrl=brightness'])
        return int(output.rsplit(':', 1)[1].strip())

    def _assess(self, config, frame):
        report = assess_frame(frame, config)
        report['brightness_control'] = self._brightness()
        if config.get('Brightness') is not None and report['brightness_control'] != config['Brightness']:
            report['reasons'].append('Brightness setting mismatch')
        report['ae_last_command'] = self._auto_exposure
        report['exposure_readback'] = 'unavailable: AE command state is not shutter readback'
        if self._auto_exposure is True:
            report['reasons'].append('AE ON: exposure is not frozen')
        return report

    def _check_conditions(self, config, include_frame=False):
        brightness_before = self._brightness()
        # There is no proven exposure GET protocol. Re-send the fixed value on
        # EVERY measurement instead of pretending cached AE state is readback.
        self._apply_fixed(config)
        frame = self._capture(config['capture_no'], (900, 1100, 0, 2590))
        before = self._assess(config, frame)
        report = {'before': before, 'reset_performed': False, 'after': None,
                  'brightness_before_apply': brightness_before,
                  'fixed_settings_applied': True, 'ExposureTime_requested': config['ExposureTime'],
                  'exposure_mode': 'fixed', 'exposure_verified': False}
        if before['reasons'] and config['reset_flag_en']:
            self._open()
            # _open reapplies the fixed config; never switch AE on to adapt.
            report['reset_performed'] = True
            frame = self._capture(config['capture_no'], (900, 1100, 0, 2590))
            report['after'] = self._assess(config, frame)
        final = report['after'] or before
        report['status'] = 'FAIL' if final['reasons'] else 'IMAGE_AND_BRIGHTNESS_OK'
        report['allow_inference'] = not final['reasons']
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
                        result = self._open()
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

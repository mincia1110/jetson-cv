"""Single-owner V4L2 camera worker. No DNX64 DLL or inference dependencies."""
from concurrent.futures import Future
import queue
import subprocess
import threading

import cv2
import numpy as np


class DinoLiteCamera:
    def __init__(self, device='/dev/video0', width=2592, height=1944, fps=10,
                 controls=None, warmup_frames=5):
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
        if self._auto_exposure is not None:
            self._ae(self._auto_exposure)
        if self.controls:
            # Use Linux control names and validated Linux values, never DLL enum IDs.
            self._command(['v4l2-ctl', '-d', self.device, '--set-ctrl',
                           ','.join(f'{key}={int(value)}' for key, value in self.controls.items())])
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

import importlib.util
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np


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
        super().__init__(width=4, height=3)

    def record(self, kind):
        self.calls.append((kind, threading.get_ident()))

    def _open(self):
        self.record('open')
        self._cap = object()
        self._publish(np.zeros((3, 4, 3), dtype=np.uint8))
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


if __name__ == '__main__':
    unittest.main()

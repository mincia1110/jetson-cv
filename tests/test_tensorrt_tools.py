import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from scripts.trt_probe import compare_outputs
from scripts.trt_session import TensorRTSession

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def test_valid_and_outside_tolerance(self):
        ref = {'x': np.array([1.0])}
        self.assertTrue(compare_outputs(ref, ref, 1e-4, 1e-5)['pass'])
        self.assertFalse(compare_outputs({'x': np.array([2.0])}, ref, 1e-4, 1e-5)['pass'])

    def test_rejects_broadcastable_shapes_and_different_names(self):
        ref = {'x': np.ones((1, 2))}
        self.assertFalse(compare_outputs({'x': np.ones(2)}, ref, 0, 0)['pass'])
        self.assertFalse(compare_outputs({'y': np.ones((1, 2))}, ref, 0, 0)['pass'])

    def test_nonfinite_and_empty_outputs_fail(self):
        for value in (float('nan'), float('inf')):
            data = {'x': np.array([value])}
            self.assertFalse(compare_outputs(data, data, 0, 0)['pass'])
        self.assertFalse(compare_outputs({}, {}, 0, 0)['pass'])


class SessionTests(unittest.TestCase):
    def test_output_owned_and_input_dtype_shape_preserved(self):
        class FakeRunner:
            output = np.ones(2)

            def infer(self, inputs, **kwargs):
                self.inputs = inputs
                return {'out': self.output}

        session = TensorRTSession('unused.engine')
        session.runner = FakeRunner()
        view = np.ones((2, 4), dtype=np.float32)[:, ::2]
        result = session.infer({'image': view, 'scalar': np.array(3, dtype=np.int32)})
        self.assertEqual(session.runner.inputs['image'].dtype, view.dtype)
        self.assertTrue(session.runner.inputs['image'].flags.c_contiguous)
        self.assertEqual(session.runner.inputs['scalar'].shape, ())
        session.runner.output[:] = 7
        np.testing.assert_array_equal(result['out'], np.ones(2))

    def test_inactive_session_fails(self):
        with self.assertRaises(RuntimeError):
            TensorRTSession('missing.engine').infer({})


class BuildTests(unittest.TestCase):
    def test_failed_builder_is_not_masked_by_tee_or_stale_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake = root / 'trtexec'
            fake.write_text('#!/bin/sh\necho simulated-parser-failure\nexit 7\n')
            fake.chmod(0o755)
            model = root / 'sample model.onnx'
            model.write_bytes(b'fake')
            out = root / 'build result'
            cmd = ['bash', str(ROOT / 'scripts/build_engine.sh'), str(model), str(out)]
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            first = subprocess.run(cmd, env=env, capture_output=True)
            self.assertEqual(first.returncode, 7)
            self.assertIn('simulated-parser-failure', (out / 'build.log').read_text())
            (out / 'model.engine').write_bytes(b'keep existing')
            second = subprocess.run(cmd, env=env, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((out / 'model.engine').read_bytes(), b'keep existing')


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from inference_backend import ONNXSession
from measurement_pipeline import Pipeline
from compare_backends import compare


class IntegrationTests(unittest.TestCase):
    def test_missing_cuda_never_silently_uses_cpu(self):
        fake = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        with patch.dict(sys.modules, onnxruntime=fake):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                ONNXSession('unused').__enter__()

    def test_backend_shared_postprocessing_and_threshold_boundary(self):
        image = np.zeros((4, 4, 3), np.uint8)
        adapter = types.SimpleNamespace(
            preprocess=lambda frame, config: ({'input': np.ones((1, 3, 4, 4), np.float32)}, image),
            postprocess=lambda outputs, th, config: {'anomaly_map': outputs['map'], 'pred_score': .5})
        backend = types.SimpleNamespace(infer=lambda feed: {'map': np.full((4,4), .5, np.float32)})
        pipeline = Pipeline({})
        pipeline.adapter, pipeline.backend = adapter, backend
        self.assertEqual(pipeline.run(image, .5)['decision'], 'NG')
        self.assertEqual(pipeline.run(image, .6)['decision'], 'OK')

    def test_nonfinite_raw_output_rejected(self):
        pipeline = Pipeline({})
        pipeline.adapter = types.SimpleNamespace(preprocess=lambda f,c: ({'x':np.ones(1)}, None))
        pipeline.backend = types.SimpleNamespace(infer=lambda f: {'y': np.array([np.nan])})
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            pipeline.run(None, 0)

    def test_compare_rejects_broadcastable_shapes_and_nan(self):
        report = compare({'x':np.ones((1,2))}, {'x':np.ones(2)}, 1e-4, 1e-5)
        self.assertFalse(report['x']['pass'])
        self.assertFalse(compare({'x':np.array([])}, {'x':np.array([])}, 0, 0)['x']['pass'])
        report = compare({'x':np.array([np.nan])}, {'x':np.array([np.nan])}, 1e-4, 1e-5)
        self.assertFalse(report['x']['pass'])


if __name__ == '__main__':
    unittest.main()

"""Compare Jetson results against the unmodified Windows v03 source fixtures."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    import torch
    import onnxruntime
    import scipy
    import cv2
except ImportError:
    torch = None


def load_reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/fixtures/windows_v03' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'Needs torch scipy ONNX Runtime OpenCV')
class WindowsV03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoring = load_reference('windows_scoring', 'scoring.py')
        package = types.ModuleType('utils')
        package.__path__ = []
        original_path = sys.path[:]
        try:
            with patch.dict(sys.modules, {'utils': package, 'utils.scoring': cls.scoring}):
                cls.reference = load_reference('windows_inferencer', 'visualad_inferencer_v03.py')
            with patch.dict(sys.modules, {'visualad_inferencer_v03': cls.reference}):
                cls.wrapper = load_reference('windows_wrapper', 'func_inference_v03.py')
        finally:
            sys.path[:] = original_path

    def compare_reference(self, outputs, threshold):
        from visualad_adapter import postprocess
        session = Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name='input')]
        session.get_outputs.return_value = [types.SimpleNamespace(name=name) for name in outputs]
        session.run.side_effect = lambda names, feed: [outputs[name] for name in (outputs if names is None else names)]
        with patch.object(self.reference.ort, 'InferenceSession', return_value=session), \
                contextlib.redirect_stdout(io.StringIO()):
            inferencer = self.wrapper.load_model('reference.onnx')
            expected = self.wrapper.user_th_inference(inferencer, torch.zeros(3, 336, 336), threshold)
        got = postprocess(outputs, threshold, {})
        np.testing.assert_array_equal(got['anomaly_map'], expected[0].numpy())
        self.assertEqual(got['pred_label'], expected[1])
        self.assertEqual(got['pred_score'], expected[2])
        np.testing.assert_array_equal(got['pred_mask'], expected[3])
        expected_names = ['fused_anomaly_map'] if 'fused_anomaly_map' in outputs else None
        self.assertEqual(session.run.call_args.args[0], expected_names)
        return got

    def test_fused_shapes_resize_and_half_precision_match_windows(self):
        rng = np.random.default_rng(19)
        for shape in ((336, 336), (1, 24, 24), (1, 1, 13, 7)):
            for dtype in (np.float16, np.float32, np.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    outputs = {'fused_anomaly_map': rng.normal(size=shape).astype(dtype),
                               'unused_patch_tokens': np.zeros((1, 579, 8), np.float32)}
                    got = self.compare_reference(outputs, 0.7)
                    self.assertEqual(got['raw_anomaly_map'].shape, (336, 336))
                    self.assertEqual(got['raw_anomaly_map'].dtype, np.float32)

    def test_wrapper_overrides_default_threshold_and_zero_threshold(self):
        outputs = {'fused_anomaly_map': np.full((1, 1, 336, 336), 0.2, np.float32)}
        got = self.compare_reference(outputs, 0.1)
        self.assertTrue(got['pred_label'])
        self.assertTrue(got['pred_mask'].all())
        got = self.compare_reference(outputs, 0.3)
        self.assertFalse(got['pred_label'])  # Previous fused implementation used zero.
        self.assertFalse(got['pred_mask'].any())
        got = self.compare_reference(outputs, got['pred_score'])
        self.assertFalse(got['pred_label'])  # Strict > at the score boundary.

    def test_token_fallback_enhanced_and_plain_features_match_windows(self):
        rng = np.random.default_rng(20)
        for enhanced in (True, False):
            for layers in ((6, 12, 18, 24), (6, 18), (12,)):
                suffix = '_enhanced' if enhanced else ''
                outputs = {f'{kind}_features{suffix}': rng.normal(size=(1, 8)).astype(np.float32)
                           for kind in ('anomaly', 'normal')}
                for layer in layers:
                    outputs[f'patch_tokens_transformed_layer_{layer}'] = rng.normal(size=(1, 19, 8)).astype(np.float32)
                with self.subTest(enhanced=enhanced, layers=layers):
                    self.compare_reference(outputs, 0.5)

    def test_partial_enhanced_pair_falls_back_to_plain_pair(self):
        rng = np.random.default_rng(21)
        outputs = {f'{kind}_features': rng.normal(size=(1, 8)).astype(np.float32)
                   for kind in ('anomaly', 'normal')}
        outputs['anomaly_features_enhanced'] = np.ones((1, 8), np.float32)
        outputs['patch_tokens_transformed_layer_24'] = rng.normal(size=(1, 19, 8)).astype(np.float32)
        self.compare_reference(outputs, 0)

    def test_scoring_utilities_match_supplied_utils(self):
        from visualad_scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
        self.assertEqual(DEFAULT_TOPK_RATIO, self.scoring.DEFAULT_TOPK_RATIO)
        for tensor in (torch.arange(20).reshape(4, 5), torch.randn(2, 9, 7)):
            for mode in ('topk_mean', 'mean', 'softmax'):
                expected = self.scoring.reduce_anomaly_map(tensor, mode=mode, topk_ratio=0.13)
                self.assertTrue(torch.equal(reduce_anomaly_map(tensor, mode=mode, topk_ratio=0.13), expected))
        for ratio in (0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                reduce_anomaly_map(torch.zeros(4, 4), topk_ratio=ratio)


class OutputSelectionTests(unittest.TestCase):
    def run_fake_session(self, names, prefer_fused):
        from inference_backend import ONNXSession
        session = Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name='input')]
        session.get_outputs.return_value = [types.SimpleNamespace(name=name) for name in names]
        session.get_providers.return_value = ['CUDAExecutionProvider']
        outputs = {name: np.ones((1,), np.float32) for name in names}
        session.run.side_effect = lambda wanted, feed: [outputs[name] for name in wanted]
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider'],
                                    SessionOptions=Mock(), InferenceSession=Mock(return_value=session))
        with patch.dict(sys.modules, onnxruntime=ort):
            with ONNXSession('model.onnx', prefer_fused_output=prefer_fused) as backend:
                result = backend.infer({'input': np.zeros((1, 3, 336, 336), np.float32)})
        session.disable_fallback.assert_called_once()
        ort.SessionOptions.return_value.add_session_config_entry.assert_called_once_with(
            'session.disable_cpu_ep_fallback', '1')
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'],
                         [('CUDAExecutionProvider', {'use_tf32': '0'})])
        return result, session.run.call_args.args[0]

    def test_mosa_requests_only_combined_map(self):
        result, wanted = self.run_fake_session(['unused_tokens', 'fused_anomaly_map'], True)
        self.assertEqual(wanted, ['fused_anomaly_map'])
        self.assertEqual(set(result), {'fused_anomaly_map'})

    def test_token_models_and_raw_diagnostics_keep_all_outputs(self):
        for names, prefer in ((['anomaly_features', 'patch_tokens_transformed_layer_24'], True),
                              (['unused_tokens', 'fused_anomaly_map'], False)):
            result, wanted = self.run_fake_session(names, prefer)
            self.assertEqual(wanted, names)
            self.assertEqual(set(result), set(names))


if __name__ == '__main__':
    unittest.main()

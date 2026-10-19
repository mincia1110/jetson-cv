"""Real CPU ONNX smoke; GPU/TensorRT verification must run on Jetson."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

try:
    import onnx
    import onnxruntime
    import cv2
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, 'Requires onnx, onnxruntime and OpenCV')
class ONNXEndToEnd(unittest.TestCase):
    def test_fused_export_runs_on_existing_mosa_onnx_session(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Needs torch')
        from onnx import helper, TensorProto
        from mosa_jetson_bridge import MOSAInferencer
        from scipy.ndimage import gaussian_filter
        with tempfile.TemporaryDirectory() as temp:
            graph = helper.make_graph([
                helper.make_node('ReduceMean', ['input'], ['fused_anomaly_map'], axes=[1], keepdims=1),
                helper.make_node('Identity', ['input'], ['unused_tokens'])],
                'fused-smoke',
                [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 336, 336])],
                [helper.make_tensor_value_info('fused_anomaly_map', TensorProto.FLOAT, [1, 1, 336, 336]),
                 helper.make_tensor_value_info('unused_tokens', TensorProto.FLOAT, [1, 3, 336, 336])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            model.ir_version = 8
            path = Path(temp) / 'fused.onnx'
            onnx.save(model, path)
            inferencer = MOSAInferencer(path, {'backend': 'onnx', 'provider': 'CPUExecutionProvider'})
            session = inferencer.backend.session
            self.assertEqual(inferencer.backend.output_names, ['fused_anomaly_map'])
            try:
                tensor = torch.zeros(3, 336, 336)
                tensor[:, 10, 20] = 0.75
                for sign in (1, -1):
                    amap, label, score, mask = inferencer.infer(tensor * sign, 1)
                    expected = (tensor * sign).numpy().mean(axis=0)
                    filtered = gaussian_filter(expected, sigma=4)
                    np.testing.assert_allclose(amap, filtered, atol=1e-7)
                    self.assertEqual(score, float(torch.topk(torch.from_numpy(expected).flatten(), 6).values.mean()))
                    self.assertEqual(label, score > 1)
                    np.testing.assert_array_equal(mask, filtered > 1)
                    self.assertIs(inferencer.backend.session, session)
            finally:
                inferencer.close()

    def test_image_to_real_onnx_to_saved_measurement(self):
        from onnx import helper, TensorProto
        from measurement_pipeline import Pipeline, save_measurement
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            graph = helper.make_graph([
                helper.make_node('ReduceMean', ['image'], ['map'], axes=[1], keepdims=0),
                helper.make_node('ReduceMean', ['map'], ['score'], keepdims=0)], 'smoke',
                [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1,3,336,336])],
                [helper.make_tensor_value_info('map', TensorProto.FLOAT, [1,336,336]),
                 helper.make_tensor_value_info('score', TensorProto.FLOAT, [])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            model.ir_version = 8
            path = temp/'model.onnx'
            onnx.save(model, path)
            config = {'backend':'onnx', 'provider':'CPUExecutionProvider', 'onnx':str(path),
                      'adapter':str(Path(__file__).resolve().parents[1]/'examples/map_score_adapter.py'),
                      'input_name':'image', 'input_mean':[0,0,0], 'input_std':[1,1,1],
                      'map_output':'map', 'score_output':'score'}
            frame = np.zeros((1944,2592,3), np.uint8)
            frame[900:1100] = 255
            with Pipeline(config) as pipeline:
                result = pipeline.run(frame, .9)
            expected = result['feed']['image'].mean(axis=1)
            np.testing.assert_allclose(result['anomaly_map'], expected.squeeze(), atol=1e-6)
            self.assertAlmostEqual(result['pred_score'], float(expected.mean()), places=5)
            self.assertEqual(result['decision'], 'OK')
            folder = temp/'measurement'
            save_measurement(folder, frame, result, config, .9, {'mode':'file'})
            np.testing.assert_array_equal(cv2.imread(str(folder/'original.png')), frame)
            self.assertFalse((folder/'original.bmp').exists())
            self.assertTrue((folder/'inputs.npz').is_file())
            self.assertEqual(json.loads((folder/'report.json').read_text())['decision'], 'OK')
            with self.assertRaises(FileExistsError):
                save_measurement(folder, frame, result, config, .9, {})


if __name__ == '__main__':
    unittest.main()

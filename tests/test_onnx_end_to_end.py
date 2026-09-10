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
            self.assertTrue((folder/'inputs.npz').is_file())
            self.assertEqual(json.loads((folder/'report.json').read_text())['decision'], 'OK')
            with self.assertRaises(FileExistsError):
                save_measurement(folder, frame, result, config, .9, {})


if __name__ == '__main__':
    unittest.main()

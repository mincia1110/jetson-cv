import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import torch
    import mosa_jetson_bridge as bridge
except ImportError:
    bridge=None


@unittest.skipIf(bridge is None,'Needs torch scipy OpenCV')
class BridgeTests(unittest.TestCase):
    def test_normalized_input_not_changed_and_same_backend_closed(self):
        from unittest.mock import MagicMock
        backend=MagicMock()
        tensor=torch.randn(3,336,336)
        expected={'anomaly_map':np.zeros((336,336)), 'pred_label':False,
                  'pred_score':0.,'pred_mask':np.zeros((336,336),bool)}
        with patch.object(bridge,'open_backend',return_value=backend), patch.object(bridge,'postprocess',return_value=expected):
            model=bridge.MOSAInferencer('model.onnx',{'backend':'onnx'})
            model.infer(tensor,.5)
            np.testing.assert_array_equal(backend.infer.call_args.args[0]['input'],tensor.numpy()[None])
            model.close()
            model.close()
            backend.__exit__.assert_called_once()

    def test_shared_data_comments_and_camera_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text('''{
                // Original MOSA comment
                "model_A_path_vis": "https://example.com/model.onnx",
                "Brightness": 84, "ExposureTime": "1/8s",
                "BRIGHT_min": 150, "BRIGHT_max": 190, "RG_gab": 18,
                "reset_flag_en": 1
            }''', encoding='utf-8')
            data = bridge.read_mosa_data(path)
            self.assertEqual(data['model_A_path_vis'], 'https://example.com/model.onnx')
            settings = bridge.camera_config(data)
            self.assertEqual(settings['Brightness'], 84)
            self.assertEqual(settings['ExposureTime'], '1/8s')
            self.assertEqual(settings['settle_frames'], 5)
            self.assertTrue(settings['reset_flag_en'])

    def test_engine_selected_by_slot_not_shared_onnx_path(self):
        data = {'backend': 'tensorrt', 'model_A_path_vis': 'same.onnx',
                'model_B_path_vis': 'same.onnx', 'model_A_path_engine': 'a.engine',
                'model_B_path_engine': 'b.engine', 'engine': 'wrong.engine'}
        self.assertEqual(bridge.model_runtime(data, 'A')['engine'], 'a.engine')
        self.assertEqual(bridge.model_runtime(data, 'B')['engine'], 'b.engine')
        with self.assertRaisesRegex(ValueError, 'model_C_path_engine'):
            bridge.model_runtime(data, 'C')
        data['backend'] = 'onnx'
        self.assertNotIn('engine', bridge.model_runtime(data, 'C'))

    def test_mosa_averages_full_frame(self):
        camera=object.__new__(bridge.MOSACamera)
        camera.height,camera.width=3,4
        frames=iter([np.full((3,4,3),10,np.uint8),np.full((3,4,3),20,np.uint8)])
        camera._read=lambda:next(frames)
        camera._publish=lambda frame:None
        result=camera._capture(2,(1,2,1,2))
        np.testing.assert_array_equal(result,np.full((3,4,3),15,np.uint8))

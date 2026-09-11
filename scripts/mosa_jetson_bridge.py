"""Narrow compatibility bridge for the supplied MOSA main (no GUI redesign)."""
import json
import re
from pathlib import Path
import numpy as np
import torch
from camera_conditions import validate_conditions
from dinolite_camera import DinoLiteCamera
from inference_backend import open_backend
from visualad_adapter import postprocess


class MOSACamera(DinoLiteCamera):
    def _capture(self, count, roi):
        # The supplied MOSA main averages the WHOLE image, unlike old test.py.
        return super()._capture(count, (0, self.height, 0, self.width))


def read_mosa_data(path='data.json'):
    """Read the original MOSA config, preserving // inside JSON strings."""
    text = Path(path).read_text(encoding='utf-8-sig')
    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\r\n]*',
                  lambda match: '' if match.group().startswith('//') else match.group(), text)
    return json.loads(text)


def camera_config(data):
    config = dict(data)
    config['reset_flag_en'] = bool(config.get('reset_flag_en', False))
    config['exposure_reset_mode'] = 'fixed'
    return validate_conditions(config)


def model_runtime(data, slot):
    """Resolve by selected A–E slot, even when two slots share an ONNX path."""
    if slot not in ('A', 'B', 'C', 'D', 'E'):
        raise ValueError(f'Unknown model slot: {slot}')
    config = {key: data[key] for key in ('backend', 'provider', 'input_name') if key in data}
    config.setdefault('backend', 'onnx')
    if config['backend'] == 'tensorrt':
        key = f'model_{slot}_path_engine'
        engine = data.get(key)
        if not isinstance(engine, str) or not engine.strip():
            raise ValueError(f'data.json: {key}에 선택한 모델의 TensorRT engine 경로를 입력하세요.')
        config['engine'] = engine
    return config


class MOSAInferencer:
    """Called and closed on the original GUI thread, keeping TRT thread ownership."""
    def __init__(self, onnx_path, runtime):
        config = dict(runtime)
        config['onnx'] = onnx_path
        if config.get('backend', 'onnx') == 'tensorrt':
            if not config.get('engine'):
                raise ValueError('Selected model has no TensorRT engine path')
        config.setdefault('backend', 'onnx')
        self.backend = open_backend(config)
        self.backend.__enter__()
        self.input_name = config.get('input_name', 'input')

    def infer(self, tensor, threshold):
        # get_transform in the original main has ALREADY normalized this tensor.
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Expected original get_transform tensor')
        values = tensor.detach().cpu().numpy()
        if values.ndim == 3:
            values = values[None]
        if values.shape != (1,3,336,336) or values.dtype != np.float32:
            raise ValueError(f'Expected float32 [1,3,336,336], got {values.dtype} {values.shape}')
        result = postprocess(self.backend.infer({self.input_name: values}), threshold, {})
        return result['anomaly_map'], result['pred_label'], result['pred_score'], result['pred_mask']

    def close(self):
        if self.backend is not None:
            backend, self.backend = self.backend, None
            backend.__exit__(None,None,None)


def user_th_inference(inferencer, test_img, user_threshold):
    return inferencer.infer(test_img, user_threshold)

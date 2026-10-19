"""Interchangeable raw ONNX/TensorRT execution; model semantics stay in adapter."""
import numpy as np


class ONNXSession:
    def __init__(self, path, provider='CUDAExecutionProvider', prefer_fused_output=False):
        self.path, self.provider = str(path), provider
        self.prefer_fused_output = prefer_fused_output

    def __enter__(self):
        import onnxruntime as ort
        if self.provider not in ort.get_available_providers():
            raise RuntimeError(f'{self.provider} unavailable: {ort.get_available_providers()}')
        options = ort.SessionOptions()
        if self.provider != 'CPUExecutionProvider':
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        provider = (self.provider, {'use_tf32': '0'}) if self.provider == 'CUDAExecutionProvider' else self.provider
        self.session = ort.InferenceSession(self.path, sess_options=options, providers=[provider])
        self.session.disable_fallback()
        if self.session.get_providers()[0] != self.provider:
            raise RuntimeError('Requested execution provider was not activated')
        names = [item.name for item in self.session.get_outputs()]
        self.output_names = (['fused_anomaly_map'] if self.prefer_fused_output and
                             'fused_anomaly_map' in names else names)
        return self

    def infer(self, feed):
        expected = {item.name for item in self.session.get_inputs()}
        if set(feed) != expected:
            raise ValueError(f'Input names: expected {expected}, got {set(feed)}')
        values = self.session.run(self.output_names, {k: np.require(v, requirements=['C']) for k, v in feed.items()})
        return {name: np.array(v, copy=True) for name, v in zip(self.output_names, values)}

    def __exit__(self, *args):
        self.session = None


def open_backend(config):
    if config['backend'] == 'onnx':
        return ONNXSession(config['onnx'], config.get('provider', 'CUDAExecutionProvider'),
                           prefer_fused_output=config.get('prefer_fused_output', False))
    if config['backend'] == 'tensorrt':
        from trt_session import TensorRTSession
        return TensorRTSession(config['engine'], prefer_fused_output=config.get('prefer_fused_output', False))
    raise ValueError('backend must be onnx or tensorrt')

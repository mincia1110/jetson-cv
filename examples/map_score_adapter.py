"""Executable adapter ONLY for models with explicit final map and score outputs.
Requires an explicit input normalization and names in config; never assumes
VisualAD's encoder output is already an anomaly map. Use model_adapter.py when
existing user_th_inference has additional postprocessing.
"""
import numpy as np
from measurement_pipeline import visible_preprocess


def preprocess(frame_bgr, config):
    image = np.asarray(visible_preprocess(frame_bgr))
    mean = np.asarray(config['input_mean'], dtype=np.float32).reshape(1,1,3)
    std = np.asarray(config['input_std'], dtype=np.float32).reshape(1,1,3)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid normalization')
    tensor = (image.astype(np.float32)/255 - mean)/std
    return {config['input_name']: np.ascontiguousarray(tensor.transpose(2,0,1)[None])}, image


def postprocess(outputs, threshold, config):
    amap = np.asarray(outputs[config['map_output']]).squeeze()
    score = np.asarray(outputs[config['score_output']])
    if score.size != 1 or amap.shape != (336,336):
        raise ValueError('This adapter requires one score and a final 336x336 map')
    return {'anomaly_map': amap, 'pred_score': float(score.reshape(-1)[0]),
            'pred_label': int(float(score.reshape(-1)[0]) >= threshold),
            'pred_mask': amap >= threshold}

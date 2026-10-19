"""Shared Windows v03 VisualAD postprocessing (CPU torch/scipy).
Official get_transform produces a 336-square CLIP-normalized tensor. The
preceding test.py stage already produces exactly 336x336 RGB, so its resize
and center crop are identity operations here.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from measurement_pipeline import visible_preprocess
from visualad_scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map

LAYERS = (6, 12, 18, 24)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess(frame_bgr, config):
    image = np.asarray(visible_preprocess(frame_bgr).convert('RGB')).copy()
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255)
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return {config.get('input_name', 'input'): tensor.unsqueeze(0).numpy()}, image


def _tensor(value, name):
    arr = np.asarray(value)
    if arr.dtype not in (np.float16, np.float32, np.float64) or not np.isfinite(arr).all():
        raise ValueError(f'{name}: expected finite floating-point output, got {arr.dtype} {arr.shape}')
    return torch.from_numpy(np.array(arr, copy=True))


def _match_input_size(amap):
    # v03 accepts [H,W], [1,H,W], or [1,1,H,W] fused outputs and resizes
    # their raw values before TopK scoring and Gaussian filtering.
    if amap.ndim == 4 and amap.shape[:2] == (1, 1):
        amap = amap[0, 0]
    elif amap.ndim == 3 and amap.shape[0] == 1:
        amap = amap[0]
    if amap.ndim != 2 or min(amap.shape) < 1:
        raise ValueError(f'fused_anomaly_map: expected one nonempty HxW map, got {tuple(amap.shape)}')
    amap = amap.float()
    if amap.shape != (336, 336):
        amap = F.interpolate(amap[None, None], size=(336, 336),
                             mode='bilinear', align_corners=False)[0, 0]
    return amap


def _compute_from_tokens(outputs):
    anomaly = outputs.get('anomaly_features_enhanced')
    normal = outputs.get('normal_features_enhanced')
    if anomaly is None or normal is None:
        anomaly, normal = outputs.get('anomaly_features'), outputs.get('normal_features')
    if anomaly is None or normal is None:
        raise ValueError('Missing VisualAD outputs: anomaly_features / normal_features')
    anomaly = _tensor(anomaly, 'anomaly_features')
    normal = _tensor(normal, 'normal_features')
    if anomaly.ndim != 2 or anomaly.shape[0] != 1 or normal.shape != anomaly.shape or anomaly.shape[1] < 1:
        raise ValueError('VisualAD features must have matching [1, embedding_size] shapes')
    # v03 deliberately uses only the last available layer, not the old sum.
    name = next((f'patch_tokens_transformed_layer_{layer}' for layer in reversed(LAYERS)
                 if f'patch_tokens_transformed_layer_{layer}' in outputs), None)
    if name is None:
        raise ValueError('Missing VisualAD outputs: no patch tokens found')
    tokens = _tensor(outputs[name], name)
    if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != anomaly.shape[1]:
        raise ValueError(f'{name}: expected [1, tokens, {anomaly.shape[1]}]')
    count = tokens.shape[1] - 3
    side = math.isqrt(max(count, 0))
    if count < 1 or side * side != count:
        raise ValueError(f'{name}: expected a square patch grid after 3 special tokens')
    anomaly = F.normalize(anomaly, dim=1, eps=1e-8)
    normal = F.normalize(normal, dim=1, eps=1e-8)
    patches = F.normalize(tokens[:, 3:, :], dim=2, eps=1e-8)
    values = torch.cosine_similarity(patches, anomaly.unsqueeze(1), dim=2) - torch.cosine_similarity(
        patches, normal.unsqueeze(1), dim=2)
    values = torch.nan_to_num(values, nan=0.0)
    return F.interpolate(values.reshape(1, 1, side, side), size=(336, 336),
                         mode='bilinear', align_corners=False)[0, 0].float()


def postprocess(outputs, threshold, config):
    with torch.no_grad():
        if outputs.get('fused_anomaly_map') is not None:
            raw = _match_input_size(_tensor(outputs['fused_anomaly_map'], 'fused_anomaly_map'))
        else:
            raw = _compute_from_tokens(outputs)
        score = float(reduce_anomaly_map(raw, mode='topk_mean', topk_ratio=DEFAULT_TOPK_RATIO).item())
        raw_np = raw.numpy()
        filtered = gaussian_filter(raw_np, sigma=4)
    # func_inference_v03.user_th_inference applies the caller's threshold to
    # both the score and filtered map, overriding predict()'s default threshold.
    return {'anomaly_map': filtered, 'raw_anomaly_map': raw_np, 'pred_score': score,
            'pred_mask': filtered > threshold, 'pred_label': score > threshold}

"""Shared postprocessing from the user's VisualADInferencer (CPU torch/scipy).
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


def postprocess(outputs, threshold, config):
    names = ['anomaly_features_enhanced', 'normal_features_enhanced'] + [
        f'patch_tokens_transformed_layer_{layer}' for layer in LAYERS]
    missing = set(names) - set(outputs)
    if missing:
        raise ValueError(f'Missing VisualAD outputs: {sorted(missing)}')
    tensors = {}
    for name in names:
        arr = np.asarray(outputs[name])
        expected = (1, 579, 1024) if name.startswith('patch_') else (1, 1024)
        if arr.shape != expected or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f'{name}: expected finite float32 {expected}, got {arr.dtype} {arr.shape}')
        tensors[name] = torch.from_numpy(np.array(arr, copy=True))
    with torch.no_grad():
        anomaly = F.normalize(tensors[names[0]], dim=1, eps=1e-8)
        normal = F.normalize(tensors[names[1]], dim=1, eps=1e-8)
        maps = []
        for name in names[2:]:
            patches = F.normalize(tensors[name][:, 3:, :], dim=2, eps=1e-8)
            # Preserve the source's second normalization inside cosine_similarity.
            values = torch.cosine_similarity(patches, anomaly.unsqueeze(1), dim=2) - torch.cosine_similarity(
                patches, normal.unsqueeze(1), dim=2)
            values = torch.nan_to_num(values, nan=0.0)
            maps.append(F.interpolate(values.reshape(1, 1, 24, 24), size=(336, 336),
                                      mode='bilinear', align_corners=False).squeeze(1))
        raw = torch.stack(maps).sum(dim=0)[0]
        # Official utils/scoring.py: DEFAULT_TOPK_RATIO=0.01, ceil, not floor.
        k = max(1, math.ceil(raw.numel() * 0.01))
        score = float(torch.topk(raw.reshape(-1), k).values.mean().item())
        filtered = gaussian_filter(raw.numpy(), sigma=4)
    return {'anomaly_map': filtered, 'raw_anomaly_map': raw.numpy(), 'pred_score': score,
            'pred_mask': filtered > threshold, 'pred_label': score > threshold}

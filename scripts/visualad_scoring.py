"""Scoring utilities from the supplied Windows v03 utils/scoring.py."""
import math
from typing import Literal

import torch

DEFAULT_TOPK_RATIO = 0.00005


def reduce_anomaly_map(
    anomaly_map: torch.Tensor,
    mode: Literal["topk_mean", "softmax", "mean"] = "topk_mean",
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reduce one [H,W] map or a [B,H,W] batch using the Windows scoring rules."""
    if anomaly_map.dim() not in (2, 3):
        raise ValueError(f"anomaly_map must be 2D or 3D tensor, got shape {tuple(anomaly_map.shape)}")
    if not anomaly_map.is_floating_point():
        anomaly_map = anomaly_map.float()
    squeeze_output = anomaly_map.dim() == 2
    flat_map = anomaly_map.reshape(1 if squeeze_output else anomaly_map.shape[0], -1)

    if mode == "topk_mean":
        ratio = float(topk_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        k = max(1, int(math.ceil(flat_map.shape[-1] * ratio)))
        scores = torch.topk(flat_map, k, dim=-1).values.mean(dim=-1)
    elif mode == "softmax":
        weights = torch.softmax(flat_map / max(temperature, eps), dim=-1)
        scores = (weights * flat_map).sum(dim=-1)
    elif mode == "mean":
        scores = flat_map.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported reduction mode: {mode}")
    return scores.squeeze(0) if squeeze_output else scores


__all__ = ["reduce_anomaly_map", "DEFAULT_TOPK_RATIO"]

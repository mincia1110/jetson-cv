# VisualAD v03 (Combined ONNX) inference wrappers
# Mirrors func_inference.py but uses the Combined ONNX inferencer (visualad_inferencer_v03).
import os
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualad_inferencer_v03 import VisualADInferencer


def load_model(path_model):
    """
    Load the Combined ONNX model on CPU (low RAM).

    Args:
        path_model: Path to the Combined ONNX file (e.g. .../model_e3000.onnx)

    Returns:
        VisualADInferencer instance
    """
    inferencer = VisualADInferencer(path=path_model, device="CPU")
    return inferencer


def user_th_inference(inferencer, test_img, user_threshold):
    """
    Run inference with user-defined threshold.

    Args:
        inferencer: VisualADInferencer (loaded by load_model)
        test_img: torch tensor [C, H, W] / [1, C, H, W], PIL Image or path
        user_threshold: threshold for anomaly decision

    Returns:
        anomaly_map: [H, W] torch tensor
        pred_label:  bool (True = anomaly)
        pred_score:  float
        pred_mask:   [H, W] torch tensor (bool)
    """
    if isinstance(test_img, str):
        test_img = Image.open(test_img).convert('RGB')

    predictions = inferencer.predict(image=test_img)
    if predictions is not None:
        for prediction in predictions:
            anomaly_map = prediction.anomaly_map
            if isinstance(anomaly_map, np.ndarray):
                anomaly_map = torch.from_numpy(anomaly_map)

            pred_label = (prediction.pred_score > user_threshold)
            pred_score = prediction.pred_score
            pred_mask = (prediction.anomaly_map > user_threshold)

    return anomaly_map, pred_label, pred_score, pred_mask
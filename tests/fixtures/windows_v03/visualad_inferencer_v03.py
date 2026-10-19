"""
VisualAD Combined ONNX Inferencer (v03)

Based on visualad_inferencer.py, but optimized for the Combined ONNX model:
when the ONNX graph already exposes a `fused_anomaly_map` output, only that
single tensor is requested from the runtime (no 4x large patch-token tensors
are materialized), and it is resized to the input image size so it lines up
with GT masks / original images. This keeps the process footprint at the same
level as the PyTorch (v01) pipeline.

Non-combined ONNX models still work via the original token-based path.
"""
import os
import sys
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import onnxruntime as ort
from scipy.ndimage import gaussian_filter

# Add project root to path for utils import
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO


class VisualADOutput:
    """Anomalib-compatible output dataclass."""
    def __init__(self, anomaly_map, pred_score, pred_mask, pred_label=None):
        self.anomaly_map = anomaly_map  # [H, W] numpy array
        self.pred_score = pred_score    # scalar (0-1)
        self.pred_mask = pred_mask      # [H, W] numpy array (bool)
        self.pred_label = pred_label    # bool (True=anomaly, False=normal)


class VisualADInferencer:
    """
    VisualAD Combined ONNX Inferencer (CPU, low RAM).

    Usage:
        inferencer = VisualADInferencer(path=".../model_e3000.onnx", device="CPU")
        predictions = inferencer.predict(image=tensor_or_pil)
    """

    def __init__(self, path, device="CPU"):
        self.path = path
        self.device = device

        # CPU keeps host RAM low & is deterministic
        if device == "CPU":
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Combined ONNX exposes the pre-fused anomaly map directly
        self.is_combined = "fused_anomaly_map" in self.output_names

        self.image_size = 336
        self.features_list = [6, 12, 18, 24]
        self.sigma = 4  # Gaussian filter sigma

        print(f"VisualADInferencer (v03) loaded: {path}")
        print(f"  Device: {device}")
        print(f"  Combined ONNX (fused_anomaly_map): {self.is_combined}")
        print(f"  Outputs: {self.output_names}")

    def predict(self, image, user_threshold=0.5):
        # ---- Convert input to a normalized [1, C, H, W] numpy array ----
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            image_np = (image_np - mean) / std
            if image_np.ndim == 3:
                image_np = image_np.transpose(2, 0, 1)[np.newaxis, ...]
        elif isinstance(image, torch.Tensor):
            # Already normalized by the DataLoader transform
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image_np = image.cpu().numpy().astype(np.float32)
        else:
            image_np = np.array(image).astype(np.float32)
            if image_np.ndim == 3:
                image_np = image_np.transpose(2, 0, 1)[np.newaxis, ...]

        input_h, input_w = image_np.shape[2], image_np.shape[3]

        # ---- Run ONNX ----
        if self.is_combined:
            # Low-RAM fast path: request ONLY the fused map (no 4x token tensors)
            outs = self.session.run(["fused_anomaly_map"], {self.input_name: image_np})
            anomaly_map = self._to_hw(torch.from_numpy(outs[0]))
            anomaly_map = self._match_input_size(anomaly_map, input_h, input_w)
        else:
            # Fallback: original token-based computation (non-combined ONNX)
            outputs = self.session.run(None, {self.input_name: image_np})
            output_dict = {name: out for name, out in zip(self.output_names, outputs)}
            anomaly_map = self._to_hw(self._compute_from_tokens(output_dict))
            anomaly_map = self._match_input_size(anomaly_map, input_h, input_w)

        # ---- Shared post-processing (score, gaussian, mask, label) ----
        reduced_score = reduce_anomaly_map(
            anomaly_map[0], mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO
        )
        pred_score = float(reduced_score.item())

        anomaly_map_np = anomaly_map[0].cpu().numpy()
        filtered_map = gaussian_filter(anomaly_map_np, sigma=self.sigma)

        pred_mask = filtered_map > user_threshold
        pred_label = pred_score > user_threshold

        return [VisualADOutput(filtered_map, pred_score, pred_mask, pred_label)]

    def _to_hw(self, tensor):
        """
        Normalize an anomaly map of shape [H,W] / [1,H,W] / [1,1,H,W]
        to a single-channel [1,H,W] tensor.
        """
        tensor = tensor.float()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(1)          # [1,1,H,W] -> [1,H,W]
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)        # [H,W]     -> [1,H,W]
        return tensor                           # [1,H,W]

    def _match_input_size(self, anomaly_map, input_h, input_w):
        """
        Interpolate a [1,H,W] anomaly map to (input_h, input_w) if needed,
        returning a [1,H,W] tensor.
        """
        if anomaly_map.shape[1] != input_h or anomaly_map.shape[2] != input_w:
            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(1),        # [1,H,W] -> [1,1,H,W]
                size=(input_h, input_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)                         # [1,1,H,W] -> [1,H,W]
        return anomaly_map

    def _compute_from_tokens(self, output_dict):
        """Original token-based anomaly-map computation (non-combined ONNX only)."""
        anomaly_features = output_dict.get("anomaly_features_enhanced")
        normal_features = output_dict.get("normal_features_enhanced")
        if anomaly_features is None or normal_features is None:
            anomaly_features = output_dict.get("anomaly_features")
            normal_features = output_dict.get("normal_features")
        if anomaly_features is None or normal_features is None:
            raise ValueError("Missing required outputs: anomaly_features / normal_features")

        anomaly_features = torch.from_numpy(anomaly_features)
        normal_features = torch.from_numpy(normal_features)

        patch_tokens_list = []
        for layer in self.features_list:
            key = f"patch_tokens_transformed_layer_{layer}"
            if key in output_dict:
                patch_tokens_list.append(torch.from_numpy(output_dict[key]))
        if not patch_tokens_list:
            raise ValueError("No patch tokens found in ONNX outputs")

        patch_tokens = patch_tokens_list[-1][:, 3:, :]  # skip 3 special tokens
        single = [self._generate_anomaly_map(
            anomaly_features, normal_features, patch_tokens, self.image_size
        )]
        return torch.stack(single).sum(dim=0)

    def _generate_anomaly_map(self, anomaly_features, normal_features, patch_tokens, image_size):
        """Generate pixel-level anomaly map from features (kept for fallback path)."""
        B = anomaly_features.shape[0]

        anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
        normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
        patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)

        anomaly_sim = torch.cosine_similarity(
            patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2
        )
        normal_sim = torch.cosine_similarity(
            patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2
        )
        anomaly_score = torch.nan_to_num(anomaly_sim - normal_sim, nan=0.0)

        num_patches = anomaly_score.shape[1]
        patch_size = int(math.sqrt(num_patches))
        anomaly_map = anomaly_score.reshape(B, patch_size, patch_size)

        return F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
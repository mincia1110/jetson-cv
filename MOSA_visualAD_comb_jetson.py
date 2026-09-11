import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import datetime
import os
import threading
import time
# Jetson: DNX64 Windows DLL replaced by the narrow bridge below.
# from func_inference import load_model, user_th_inference_winclip, user_th_inference
# VisualAD execution is supplied by mosa_jetson_bridge; Subspace calls remain commented.
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
import numpy as np
import json
import re
import sys
import csv
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2.functional import to_dtype, to_image
import math
import time

from pathlib import Path
import pickle
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.subspacead.post_process.scoring import post_process_map, calculate_anomaly_scores
    from src.subspacead.config import parse_layer_indices, parse_grouped_layers
except ImportError:
    # SubspaceAD is inactive; original function bodies are retained below.
    pass


# PyInstaller 환경에서 리소스 경로 처리
def resource_path(relative_path):
    """PyInstaller 환경에서 리소스 경로를 얻는 함수"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 실행 환경
        return os.path.join(sys._MEIPASS, relative_path)
    # Python 실행 환경
    return os.path.join(os.path.abspath("."), relative_path)


# Add project root to path for utils imports
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.transforms import get_transform

# Jetson bridge lives beside this main in scripts/.
BRIDGE_DIR = Path(__file__).resolve().parent / 'scripts'
sys.path.insert(0, str(BRIDGE_DIR))
from mosa_jetson_bridge import MOSACamera, MOSAInferencer, camera_config, model_runtime, read_mosa_data, user_th_inference
def read_camera_settings():
    return camera_config(read_mosa_data())


# Constants
DNX64_PATH = 'C:\Program Files\DNX64\DNX64.dll'
DEVICE_INDEX = 0
QUERY_TIME = 0.05  # Buffer time for Dino-Lite to return value
COMMAND_TIME = 0.5  # Buffer time to allow Dino-Lite to process command

# -----------------------------
# 해상도 설정
# -----------------------------
CAMERA_FPS = 30
CAMERA_WIDTH = 2592
CAMERA_HEIGHT = 1944

DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 750
DISPLAY2_SIZE = 256

CROP_TOP = 200
CROP_BOTTOM = 200

# -----------------------------
# AI Model 설정
# -----------------------------
# 모델 선택 GUI 에서 선택된 값으로 설정됨




#model_path_vis = None
#model_path_sub = None




# -----------------------------
# 모델 선택 GUI 함수
# -----------------------------
def show_model_selection_gui():
    """
    프로그램 시작 시 AI 모델을 선택하는 GUI 창
    """

    print("[Model_selection]")
    selected_vis_model = {"path": None}
    selected_sub_model = {"path": None}
    selected_model_name = None
    selected_slot = None

    try:
        data = read_mosa_data()
        model_A_name = data.get("model_A_name", "Model A")
        model_A_path_vis = data.get("model_A_path_vis", None)
        model_A_path_sub = data.get("model_A_path_sub", None)
        model_B_name = data.get("model_B_name", "Model B")
        model_B_path_vis = data.get("model_B_path_vis", None)
        model_B_path_sub = data.get("model_B_path_sub", None)
        model_C_name = data.get("model_C_name", "Model C")
        model_C_path_vis = data.get("model_C_path_vis", None)
        model_C_path_sub = data.get("model_C_path_sub", None)
        model_D_name = data.get("model_D_name", "Model D")
        model_D_path_vis = data.get("model_D_path_vis", None)
        model_D_path_sub = data.get("model_D_path_sub", None)
        model_E_name = data.get("model_E_name", "Model E")
        model_E_path_vis = data.get("model_E_path_vis", None)
        model_E_path_sub = data.get("model_E_path_sub", None)
    except:
        messagebox.showwarning("경고", "JSON file load Failed!")
        sys.exit()


    def select_model_a():
        nonlocal selected_model_name, selected_slot
        selected_slot = "A"
        selected_model_name = model_A_name
        selected_vis_model["path"] = model_A_path_vis
        selected_sub_model["path"] = model_A_path_sub
        root.quit()
        root.destroy()

    def select_model_b():
        nonlocal selected_model_name, selected_slot
        selected_slot = "B"
        selected_model_name = model_B_name
        selected_vis_model["path"] = model_B_path_vis
        selected_sub_model["path"] = model_B_path_sub
        root.quit()
        root.destroy()

    def select_model_c():
        nonlocal selected_model_name, selected_slot
        selected_slot = "C"
        selected_model_name = model_C_name
        selected_vis_model["path"] = model_C_path_vis
        selected_sub_model["path"] = model_C_path_sub
        root.quit()
        root.destroy()

    def select_model_d():
        nonlocal selected_model_name, selected_slot
        selected_slot = "D"
        selected_model_name = model_D_name
        selected_vis_model["path"] = model_D_path_vis
        selected_sub_model["path"] = model_D_path_sub
        root.quit()
        root.destroy()

    def select_model_e():
        nonlocal selected_model_name, selected_slot
        selected_slot = "E"
        selected_model_name = model_E_name
        selected_vis_model["path"] = model_E_path_vis
        selected_sub_model["path"] = model_E_path_sub
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Model Selection")
    root.geometry("300x320")
    root.resizable(False, False)







    # 중앙 정렬을 위한 프레임
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    # 제목 레이블
    title_label = ttk.Label(
        main_frame,
        text="측정 모델 선택",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=(0, 20))

    # 버튼 프레임
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", pady=10)

    # Selection A 버튼
    btn_a = ttk.Button(
        button_frame,
        text=model_A_name,
        command=select_model_a,
        width=20
    )
    btn_a.pack(pady=5)

    # Selection B 버튼
    btn_b = ttk.Button(
        button_frame,
        text=model_B_name,
        command=select_model_b,
        width=20
    )
    btn_b.pack(pady=5)

    # Selection C 버튼
    btn_c = ttk.Button(
        button_frame,
        text=model_C_name,
        command=select_model_c,
        width=20
    )
    btn_c.pack(pady=5)

    # Selection D 버튼
    btn_d = ttk.Button(
        button_frame,
        text=model_D_name,
        command=select_model_d,
        width=20
    )
    btn_d.pack(pady=5)

    # Selection E 버튼
    btn_e = ttk.Button(
        button_frame,
        text=model_E_name,
        command=select_model_e,
        width=20
    )
    btn_e.pack(pady=5)

    # 창을 화면 중앙에 표시
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()

    return selected_model_name, selected_vis_model["path"], selected_sub_model["path"], (model_runtime(data, selected_slot) if selected_slot else None)






class SimpleInferenceModel(nn.Module):
    def __init__(self, feature_extractor, config, pca_params, device="cpu"):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.config = config
        self.pca_params = pca_params
        self.device = device

        self.image_res = config.get("image_res", 256)
        self.layers = config.get("layers", "-12,-13,-14,-15,-16,-17,-18")
        self.agg_method = config.get("agg_method", "mean")
        self.grouped_layers = config.get("grouped_layers")
        self.docrop = config.get("docrop", False)
        self.use_clahe = config.get("use_clahe", False)
        self.dino_saliency_layer = config.get("dino_saliency_layer", 6)
        self.score_method = config.get("score_method", "reconstruction")
        self.drop_k = config.get("drop_k", 0)

        self.pca_dict = self._build_pca_dict(pca_params)

    def _build_pca_dict(self, pca_params):
        mean_val = pca_params.get("mean", pca_params.get("mu"))
        components_val = pca_params.get("components")
        eigvals_val = pca_params.get("explained_variance", pca_params.get("eigvals"))
        k_val = pca_params.get("n_components", pca_params.get("k", 0))

        return {
            "mu": mean_val,
            "components": components_val,
            "eigvals": eigvals_val,
            "k": k_val,
            "eps": pca_params.get("eps", 1e-6),
        }


def load_subspace_model(config_path, cache_dir=None, device="cpu"):
    from transformers import AutoModel, AutoImageProcessor

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    pca_params_file = config.get("pca_params_file", "pca_params.pkl")
    pca_params_path = config_path.parent / pca_params_file

    with open(pca_params_path, "rb") as f:
        pca_params = pickle.load(f)

    model_ckpt = config.get("model_ckpt", "facebook/dinov2-with-registers-small")
    threshold_img = config.get("threshold_img", None)
    from_pretrained_kwargs = {"local_files_only": True}
    if cache_dir:
        from_pretrained_kwargs["cache_dir"] = cache_dir

    feature_extractor = AutoModel.from_pretrained(model_ckpt, **from_pretrained_kwargs)
    feature_extractor = feature_extractor.eval().to(device)

    try:
        processor = AutoImageProcessor.from_pretrained(model_ckpt, **from_pretrained_kwargs)
        feature_extractor.processor = processor
    except OSError:
        feature_extractor.processor = None

    model = SimpleInferenceModel(feature_extractor, config, pca_params, device)
    return model, threshold_img



def extract_subspace_features(model, pil_img):
    from transformers import AutoImageProcessor

    device = model.device
    model_ckpt = model.config.get("model_ckpt", "facebook/dinov2-with-registers-small")

    processor = getattr(model.feature_extractor, "processor", None)
    if processor is None:
        try:
            processor = AutoImageProcessor.from_pretrained(model_ckpt, local_files_only=True)
        except OSError:
            processor = None

    size = {"height": model.image_res, "width": model.image_res}

    if processor is not None:
        inputs = processor(
            images=[pil_img],
            return_tensors="pt",
            do_resize=True,
            size=size,
            do_center_crop=False,
            crop_size=size,
        ).to(device)
    else:
        img_array = np.array(pil_img.resize((model.image_res, model.image_res))).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        inputs = {"pixel_values": torch.from_numpy(img_array).to(device)}

    cfg = model.feature_extractor.config
    ps = getattr(cfg, "patch_size", 14)
    num_reg = getattr(cfg, "num_register_tokens", 0)
    drop_front = 1 + num_reg
    h_p, w_p = model.image_res // ps, model.image_res // ps
    n_expected = h_p * w_p

    with torch.no_grad():
        outputs = model.feature_extractor(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            output_attentions=False
        )
        hidden_states = outputs.hidden_states

    layers = parse_layer_indices(model.layers)
    grouped_layers = parse_grouped_layers(model.grouped_layers) if model.grouped_layers else []

    def spatial_from_seq(seq_tokens, drop_front, n_expected, h_p, w_p):
        B, N, C = seq_tokens.shape
        tokens = seq_tokens[:, drop_front:drop_front + n_expected, :]
        return tokens.reshape(B, h_p, w_p, C)

    if model.agg_method == "group":
        all_layer_indices = sorted(list(set(idx for group in grouped_layers for idx in group)))
        layer_tensors = {li: spatial_from_seq(hidden_states[li], drop_front, n_expected, h_p, w_p) for li in
                         all_layer_indices}
        fused_groups = [torch.stack([layer_tensors[li] for li in group], dim=0).mean(dim=0) for group in grouped_layers]
        fused = torch.cat(fused_groups, dim=-1)
    else:
        feats = [spatial_from_seq(hidden_states[li], drop_front, n_expected, h_p, w_p) for li in layers]
        if model.agg_method == "concat":
            fused = torch.cat(feats, dim=-1)
        elif model.agg_method == "mean":
            fused = torch.stack(feats, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {model.agg_method}")

    b, h, w, c = fused.shape
    tokens_reshaped = fused.reshape(b * h * w, c).cpu().numpy()

    scores = calculate_anomaly_scores(
        tokens_reshaped,
        model.pca_dict,
        model.score_method,
        model.drop_k
    )

    anomaly_map = scores.reshape(h_p, w_p)
    anomaly_map_processed = post_process_map(anomaly_map, model.image_res)

    # SubspaceAD score는 map 기반으로 얻지만,
    # 여기서는 참고용 scalar score만 리턴
    img_score = float(np.percentile(anomaly_map_processed, 99.925))

    return anomaly_map_processed, img_score


def normalize_subspace_to_visual_scale(subspace_scores, visual_scores):
    """
    SubspaceAD score를 VisualAD score 분포 범위에 맞춰 정규화
    """
    visual_scores = np.asarray(visual_scores, dtype=np.float32)
    subspace_scores = np.asarray(subspace_scores, dtype=np.float32)

    v_min = float(np.min(visual_scores))
    v_max = float(np.max(visual_scores))
    s_min = float(np.min(subspace_scores))
    s_max = float(np.max(subspace_scores))

    if abs(s_max - s_min) < 1e-8:
        return np.full_like(subspace_scores, fill_value=v_min, dtype=np.float32)

    subspace_scaled = (subspace_scores - s_min) / (s_max - s_min + 1e-8)
    subspace_scaled = subspace_scaled * (v_max - v_min) + v_min
    return subspace_scaled


def resize_and_match_subspace_map(subspace_map, target_shape):
    """
    subspace_map: [H, W] or [1, H, W] or [B, H, W]
    target_shape: (H, W)
    return: [H, W]
    """
    if isinstance(subspace_map, np.ndarray):
        subspace_map = torch.from_numpy(subspace_map)

    if subspace_map.dim() == 2:
        subspace_map = subspace_map.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    elif subspace_map.dim() == 3:
        subspace_map = subspace_map.unsqueeze(1)  # [B, 1, h, w]
    elif subspace_map.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported subspace_map shape: {subspace_map.shape}")

    resized = F.interpolate(subspace_map.float(), size=target_shape, mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).squeeze(0)
    return resized


def minmax_normalize_tensor(x, eps=1e-8):
    """
    x: torch.Tensor
    return: 0~1로 정규화된 tensor
    """
    x_min = x.amin()
    x_max = x.amax()
    if (x_max - x_min).abs() < eps:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)


def fuse_visual_and_subspace_maps(visual_map, subspace_map, th_tensor, visual_score, sub_shift, sub_scale):
    """
    visual_map: torch.Tensor [H, W] or [1, H, W]
    subspace_map: torch.Tensor [h, w] or [1, h, w]
    return:
        fused_map: torch.Tensor [H, W]
    """
    if isinstance(visual_map, np.ndarray):
        visual_map = torch.from_numpy(visual_map)
    if isinstance(subspace_map, np.ndarray):
        subspace_map = torch.from_numpy(subspace_map)

    if visual_map.dim() == 3 and visual_map.shape[0] == 1:
        visual_map = visual_map.squeeze(0)
    if subspace_map.dim() == 3 and subspace_map.shape[0] == 1:
        subspace_map = subspace_map.squeeze(0)

    if visual_map.dim() != 2:
        raise ValueError(f"visual_map must be 2D, got {visual_map.shape}")

    target_shape = visual_map.shape

    # 1) subspace_map 크기 맞추기
    subspace_resized = resize_and_match_subspace_map(subspace_map, target_shape)

    # 2) subspace_map 0~1 정규화
    # subspace_norm = minmax_normalize_tensor(subspace_resized)

    # 3) visual_map 범위로 스케일링
    # visual_min = visual_map.amin()
    # visual_max = visual_map.amax()
    subspace_shifted = subspace_resized - th_tensor - sub_shift
    subspace_scaled = subspace_shifted / sub_scale

    if visual_score < -2.5:
        fused_map = visual_map
    else:
        fused_map = torch.maximum(visual_map, subspace_scaled)

    # 4) pixel-wise max
    # fused_map = torch.maximum(visual_map, subspace_scaled)

    return fused_map


def compute_score_from_map(anomaly_map, mode="max", topk=300):
    """
    anomaly_map: torch.Tensor [H, W] or [1, H, W]
    mode:
        - "max": max 값 사용
        - "topk": 상위 topk 평균 사용
    """
    if isinstance(anomaly_map, np.ndarray):
        anomaly_map = torch.from_numpy(anomaly_map)

    if anomaly_map.dim() == 3 and anomaly_map.shape[0] == 1:
        anomaly_map = anomaly_map.squeeze(0)

    flat = anomaly_map.reshape(-1)

    if mode == "max":
        return float(flat.max().item())
    elif mode == "topk":
        k = min(topk, flat.numel())
        vals, _ = torch.topk(flat, k=k)
        return float(vals.mean().item())
    else:
        raise ValueError(f"Unknown score mode: {mode}")


V_s = 900
V_e = 1100
H_s = 0
H_e = 2590

sq_size = 600

img_size_h = 448
img_size_w = 448


# -----------------------------
# 카메라 초기화
# -----------------------------`
def camera_init():
    runtime = read_mosa_data()
    camera = MOSACamera(device=runtime.get('device', '/dev/video0'),
                        fps=runtime.get('fps', 10), initial_config=read_camera_settings())
    try:
        camera.ready.result()
    except Exception:
        camera.close()
        raise
    return camera, None, False


class CameraGUI:
    def __init__(self, root, camera, microscope, inferencer, subspace_model, threshold_img, selected_model_name):
        self.camera = camera
        self.microscope = microscope
        self.inferencer = inferencer
        self.subspace_model = subspace_model
        self.threshold_img = threshold_img
        self.selected_model_name = selected_model_name
        self.load_json()
        self.root = root
        self.root.title("MOSA Measurement (NDC Task)")

        # 저장 폴더
        self.save_dir = None

        # 프레임 공유
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # -----------------------------
        # Inference with gathered data
        # -----------------------------
        self.pred_scores = torch.tensor([])
        self.gt_labels = torch.tensor([])

        # -----------------------------
        # 전체 레이아웃
        # -----------------------------
        main_frame = ttk.Frame(root, padding=10)
        main_frame.grid(row=0, column=0)

        # =============================
        # 좌측: 메인 영상
        # =============================
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=(5, 5))
        ttk.Label(
            self.video_label,
            text="MAIN",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")
        self.video_label = ttk.Label(self.video_label)
        self.video_label.pack()

        # =============================
        # 좌측: 프래임
        # =============================
        rgb_box = tk.LabelFrame(
            main_frame,
            text="[ 검사 결과 ]",
            font=("Arial", 11, "bold"),
            bd=0,  # 선 두께
            # relief="solid",
            labelanchor="n",
            highlightthickness=3,  # 바깥 테두리
            highlightbackground="gray"  # 테두리 색
        )
        rgb_box.grid(row=1, column=0, pady=5)

        # =============================
        # 좌측: 중간 영상
        # =============================
        self.r_label = ttk.Label(rgb_box)
        self.r_label.grid(row=0, column=0, padx=(10, 10), pady=5)
        ttk.Label(
            self.r_label,
            text="- 측정 사진 전처리",
            font=("Arial", 10)
        ).pack(anchor="w")
        self.r_label = ttk.Label(self.r_label)
        self.r_label.pack()

        # =============================
        # 좌측: 아래 영상1
        # =============================
        self.g_label = ttk.Label(rgb_box)
        self.g_label.grid(row=1, column=0, padx=(10, 10), pady=5)
        ttk.Label(
            self.g_label,
            text="- AI 분석 결과(NG 영역)",
            font=("Arial", 10)
        ).pack(anchor="w")
        self.g_label = ttk.Label(self.g_label)
        self.g_label.pack()

        # =============================
        # 좌측: 아래 영상2
        # =============================
        self.b_label = ttk.Label(rgb_box)
        self.b_label.grid(row=2, column=0, padx=(10, 10), pady=5)
        ttk.Label(
            self.b_label,
            text="- AI 분석 결과(Heatmap)",
            font=("Arial", 10)
        ).pack(anchor="w")
        self.b_label = ttk.Label(self.b_label)
        self.b_label.pack()

        crop_img_1 = np.empty((200, 1000))
        crop_img_2 = np.empty((200, 1000))
        crop_img_3 = np.empty((200, 1000))
        crop_img_1.fill(200)
        crop_img_2.fill(200)
        crop_img_3.fill(200)
        crop_img_1 = (crop_img_1).astype(np.uint8)
        crop_img_2 = (crop_img_2).astype(np.uint8)
        crop_img_3 = (crop_img_3).astype(np.uint8)
        self._update_label(self.r_label, crop_img_1)
        self._update_label(self.g_label, crop_img_2)
        self._update_label(self.b_label, crop_img_3)

        # =============================
        # 우측: 컨트롤 패널
        # =============================
        right_frame = ttk.Frame(
            main_frame,
            padding=(20, 0),
            width=300
        )
        right_frame.grid(row=0, column=3, rowspan=2, padx=15, pady=20, sticky="nsew")
        right_frame.grid_propagate(False)


        # =============================
        # 선택 모델 표시
        # =============================
        model_frame = ttk.Frame(right_frame)
        model_frame.pack(fill="x", pady=5)

        self.model_name_label = ttk.Label(
            model_frame,
            text="Model: " + self.selected_model_name,
            font=("Arial", 20, "bold"),
            wraplength=260,
            relief="solid",
            borderwidth=1,
            padding=5,
            background="yellow"
        )
        self.model_name_label.pack(fill="x", pady=2, anchor="center")
        # =============================
        # 저장 폴더
        # =============================
        folder_frame = ttk.Frame(right_frame)
        folder_frame.pack(fill="x", pady=5)

        self.folder_label = ttk.Label(
            folder_frame,
            text="저장 폴더:\n(선택되지 않음)",
            wraplength=260,
            foreground="red"
        )
        self.folder_label.pack(anchor="w")

        ttk.Button(
            folder_frame,
            text="저장 폴더 선택",
            command=self.select_save_folder
        ).pack(fill="x", pady=3)

        # =============================
        # 파일명
        # =============================
        name_frame = ttk.Frame(right_frame)
        name_frame.pack(fill="x", pady=10)

        ttk.Label(name_frame, text="측정 이름 (파일명)").pack(anchor="w")
        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack(fill="x", pady=3)

        # =============================
        # Threshold 값
        # =============================
        threshold_frame = ttk.Frame(right_frame)
        threshold_frame.pack(fill="x", pady=3)

        self.default_value = tk.IntVar(value=self.init_threshold)

        ttk.Label(threshold_frame, text="Threshold(-20~20) : ").pack(side=tk.LEFT)
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.default_value)
        self.threshold_entry.pack(side=tk.LEFT, fill="x", expand=True, pady=3)

        # =============================
        # 파일명 + 저장 버튼
        # =============================
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill="x", pady=3)
        style = ttk.Style()
        style.configure(
            "Big.TButton",
            background="lightblue",
            font=("Arial", 18, "bold")
        )
        measure_button = ttk.Button(
            button_frame,
            text="측 정",
            command=self.capture_image,
            style="Big.TButton"
        )
        measure_button.pack(fill="x", pady=(20, 10), ipady=20)

        # =====================
        # ▶ OK / NG 상태 박스
        # =====================
        self.result_container = tk.Frame(
            right_frame,
            width=200,
            height=200,
            bg="gray",
            bd=1,
            relief="solid"
        )
        self.result_container.pack(pady=50)
        self.result_container.pack_propagate(False)

        self.result_label = tk.Label(
            self.result_container,
            text="READY",
            font=("Arial", 35, "bold"),
            fg="white",
            bg="gray"
        )
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

        # =============================
        # 마지막 저장 정보 (하단 고정)
        # =============================
        self.last_save_label = ttk.Label(
            right_frame,
            text="마지막 저장 파일:\n없음",
            foreground="blue",
            anchor="w",
            justify="left"
        )
        self.last_save_label.pack(side="bottom", fill="x", pady=5)

        # -----------------------------
        # 캡처 스레드 시작
        # -----------------------------
        self.capture_thread = threading.Thread(
            target=self.camera_capture_loop,
            daemon=True
        )
        self.capture_thread.start()

        try:
            import pyi_splash
            pyi_splash.close()
        except:
            pass

        # GUI 업데이트
        self.update_gui()

        # ret, frame = self.camera.read()
        # bright_sampling = cv2.mean(frame[1944-10:1944, 0:2590 , :])
        # RG_diff = int(bright_sampling[2] - bright_sampling[1])
        # print(">  ",bright_sampling)
        # print(">> ",RG_diff)

    def load_json(self):
        try:
            data = read_mosa_data()
            self.CSV_path = data["CSV_path"]
            self.BRIGHT_min = data["BRIGHT_min"]
            self.BRIGHT_max = data["BRIGHT_max"]
            self.RG_gab = data["RG_gab"]
            self.crop_v_offset = data["crop_v_offset"]
            self.init_threshold = data["threshold"]
            self.sub_shift  = data["sub_shift"]
            self.sub_scale  = data["sub_scale"]
            self.Data_log_flag = data["Data_log_flag"]
            self.Camera_or_BMP_flag = data["Camera_or_BMP_flag"]
            self.reset_flag_en = data["reset_flag_en"]
            self.AI_result_flag = data["AI_result_flag"]
            self.ExposureValue = data.get("ExposureValue")
            self.Brightness = data["Brightness"]
            self.capture_no = data["capture_no"]
            settings = camera_config(data)
            self.camera_settings = settings
            self.Brightness = settings['Brightness']
            self.ExposureTime = settings['ExposureTime']
            self.BRIGHT_min = settings['BRIGHT_min']
            self.BRIGHT_max = settings['BRIGHT_max']
            self.RG_gab = settings['RG_gab']
            self.reset_flag_en = settings['reset_flag_en']
            self.capture_no = settings['capture_no']
            return False  # Fixed settings are applied and checked at each capture.


        except:
            messagebox.showwarning("경고", "JSON file load Failed!")
            sys.exit()

    def reset_camera(self):
        self.camera.reconnect().result()
        self.result_container.config(bg="gray")
        self.result_label.config(text="READY", bg="gray", font=("Arial", 35, "bold"))

    def camera_capture_loop(self):
        while self.running:
            frame, error = self.camera.latest()
            if frame is not None:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                with self.frame_lock:
                    self.current_frame = None
            time.sleep(0.03)

    # -----------------------------
    # GUI 영상 업데이트
    # -----------------------------
    def update_gui(self):
        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()

        if frame is not None:
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            h, w, _ = display_frame.shape

            crop_ratio = 0.2
            crop_h = round(DISPLAY_HEIGHT * crop_ratio)
            y_start = max((h - crop_h) // 2, 0)
            y_end = y_start + crop_h

            # ▶ 메인 영상
            display_frame_cropped = display_frame[y_start:y_end, :]
            self._update_label(self.video_label, display_frame_cropped)

        self.root.after(15, self.update_gui)

    def _update_label(self, label, display_frame):
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # -----------------------------
    # 저장 폴더 선택
    # -----------------------------
    def select_save_folder(self):
        folder = filedialog.askdirectory(title="이미지 저장 폴더 선택")
        if folder:
            self.save_dir = folder
            self.folder_label.config(
                text=f"저장 폴더:\n{self.save_dir}",
                foreground="black"
            )

    # ---------------------
    # 측정 버튼 로직
    # ---------------------
    def measure(self, result, bright_b):
        if self.current_frame is None:
            return

        if self.reset_flag == True and self.reset_flag_en == True:
            self.root.after(500, self.reset_camera)
            self.result_container.config(bg="blue")
            self.result_label.config(text="FAIL\n(WAIT)", bg="blue", font=("Arial", 35, "bold"))

        else:
            if result == "OK":
                self.result_container.config(bg="green")
                self.result_label.config(text="OK", bg="green", font=("Arial", 80, "bold"))
            else:
                self.result_container.config(bg="red")
                self.result_label.config(text="NG", bg="red", font=("Arial", 80, "bold"))

    # ---------------------
    # Wait 표시 함수
    # ---------------------
    def show_wait_status(self):
        """측정 중 Running 상태 표시"""
        self.result_container.config(bg="gray")
        self.result_label.config(text="Running", bg="gray", font=("Arial", 32, "bold"))

    # -----------------------------
    # 이미지 저장 (원본)
    # -----------------------------
    def capture_image(self):
        try:
            self._capture_image_impl()
        except Exception as exc:
            self.result_container.config(bg='blue')
            self.result_label.config(text='FAIL', bg='blue')
            messagebox.showerror('Measurement failed', str(exc))

    def _capture_image_impl(self):
        # 측정 시작 시 "Wait" 표시
        self.show_wait_status()
        self.root.update_idletasks()  # UI 즉시 업데이트
        if self.save_dir is None:
            messagebox.showwarning("경고", "저장 폴더를 선택해주세요.")
            return

        name = self.name_entry.get().strip()
        if name == "":
            messagebox.showwarning("경고", "측정 이름을 입력해주세요.")
            return

        try:
            threshold = float(self.threshold_entry.get())
            if threshold > 20 or threshold < -20:
                messagebox.showwarning("경고", "threshold 값의 범위는 -20~20 사이입니다.")
                return
            else:
                self.user_threshold = threshold / 10
        except ValueError:
            messagebox.showwarning("경고", "threshold 값(-20~20)을 입력해주세요.")
            return

        with self.frame_lock:
            if self.current_frame is None:
                messagebox.showerror("오류", "카메라 프레임이 없습니다.")
                return
            # frame_to_save = self.current_frame.copy()
            # print(" jpg type : ", type(frame_to_save))

            '''
            num_fr = 10
            s_intv = 0.0001
            frame_sum = None
            valid_count = 0

            for i in range(num_fr):
                ret, frame = self.camera.read()

                if not ret or frame is None:
                    if i < num_fr - 1:
                        time.sleep(s_intv)
                    continue

                frame = frame.astype(np.float32)

                # if ret:
                #     with self.frame_lock:
                #         frame = frame.copy().astype(np.float32)

                #frame = cv2.GaussianBlur(frame, (15,15),0)
                if frame_sum is None:
                    frame_sum = frame
                else:
                    frame_sum += frame
                    #frame_sum = np.maximum(frame_sum, frame)
                valid_count += 1


            time.sleep(s_intv)

            if valid_count > 0:
                avg_frame = (frame_sum / valid_count).astype(np.uint8)
                #avg_frame = cv2.cvtColor(avg_frame, cv2.COLOR_RGB2BGR)

                with self.frame_lock:
                    self.current_frame = avg_frame
            '''

            # frame_to_save = frame_sum

            # time.sleep(0.01)
            # frame_to_save_4 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_5 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_6 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_7 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_8 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_9 = self.current_frame.copy()
            # time.sleep(0.01)
            # frame_to_save_10 = self.current_frame.copy()
            # frame_to_save = (frame_to_save_1.astype(np.float32) + frame_to_save_2.astype(np.float32) + frame_to_save_3.astype(np.float32) + frame_to_save_4.astype(np.float32) + frame_to_save_5.astype(np.float32)
            #                  + frame_to_save_6.astype(np.float32) + frame_to_save_7.astype(np.float32) + frame_to_save_8.astype(np.float32) + frame_to_save_9.astype(np.float32) + frame_to_save_10.astype(np.float32)
            #                  ) / 10.0

        filepath = os.path.join(self.save_dir, f"{name}.bmp")

        # ✅ 동일 파일 존재 시 덮어쓰기 확인
        if os.path.exists(filepath):
            overwrite = messagebox.askyesno(
                "파일 덮어쓰기 확인",
                f"이미 존재하는 파일입니다:\n\n{name}.bmp\n\n덮어쓰시겠습니까?"
            )
            if not overwrite:
                return  # 저장 취소

        start = time.time()
        first_time = start

        self.load_json()
        self.reset_flag = False
        self.camera_report = {'mode': 'file', 'exposure_readback': 'unavailable'}
        if self.Camera_or_BMP_flag == 1:
            captured = self.camera.prepare_inference(self.camera_settings).result()
            self.camera_report = captured['report']
            if captured['frame'] is None:
                self.result_container.config(bg='blue')
                self.result_label.config(text='FAIL', bg='blue')
                messagebox.showerror('Camera condition failed', json.dumps(self.camera_report, ensure_ascii=False))
                return
            frame_to_save = captured['frame']

        if self.Camera_or_BMP_flag == 1:
            if not cv2.imwrite(filepath, frame_to_save):
                raise RuntimeError(f'Image write failed: {filepath}')
        # cv2.imwrite(filepath, frame_to_save)

        now = datetime.datetime.now()
        date = f"{now.strftime('%Y-%m-%d, %H:%M:%S')}"
        self.last_save_label.config(
            text=f"마지막 저장 파일:\n{name}.bmp\n{date}"
        )

        # ===========================================================================
        # ============================== 이미지 AI 분석 ===============================
        # ===========================================================================
        visual_scores = []
        subspace_scores = []
        visual_maps = []
        subspace_maps = []

        fused_maps = []
        fused_scores = []

        frame_to_save = Image.open(filepath).convert("RGB")

        r_s = V_s
        r_e = V_e
        c_s = H_s
        c_e = H_e

        # threshold_img = self.threshold_sub

        # ------------image 전저리-------------

        image_np = np.array(frame_to_save)

        image_np = cv2.GaussianBlur(image_np, (7, 7), 0)
        norm_imgo = cv2.normalize(image_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        crop_wh = norm_imgo[r_s:r_e, c_s:c_e, :]
        # crop_wh = cv2.resize(crop_wh, (crop_wh.shape[1] // 2, crop_wh.shape[0] // 2))
        crop_wh = np.clip((crop_wh - 0.3) / (1 - 0.3), 0, 1)
        gamma = 1.4
        crop_wh = np.power(crop_wh, gamma)
        crop_wh = cv2.resize(crop_wh, (336, 168))
        #crop_wh = cv2.resize(crop_wh, (336, 70))
        black_image = np.zeros((336, 336, 3), dtype=np.float32)
        #black_image = np.ones((336, 336, 3), dtype=np.float32) * 255
        # h = crop_wh.shape[0]
        # start_row = (sq_size - h) // 2
        # end_row = start_row + h

        x_offset = (black_image.shape[1] - crop_wh.shape[1]) // 2
        y_offset = (black_image.shape[0] - crop_wh.shape[0]) // 2

        black_image[y_offset:y_offset + crop_wh.shape[0], x_offset:x_offset + crop_wh.shape[1], :] = crop_wh

        # Preprocessing using utils.transforms.get_transform
        # image = Image.fromarray((black_image * 255).astype(np.uint8)).resize((sq_size, sq_size))

        # image = Image.fromarray((black_image * 255).astype(np.uint8)).resize((sq_size, sq_size), resample=Image.BICUBIC)
        image = Image.fromarray((black_image * 255).astype(np.uint8))

        class Args:
            image_size = 336
            features_list = [6, 12, 18, 24]

        preprocess, _ = get_transform(Args())
        crop_wh_pil = preprocess(image)

        # Brightness check
        bright_sampling = cv2.mean(image_np[1944 - 10:1944, 0:2590, :])
        bright_b = int(bright_sampling[0])
        RG_diff = int(bright_sampling[0] - bright_sampling[1])
        print("--------------------------------------------------------------------------------")
        print("[bright_sampling] : ", bright_sampling)
        print("bright_b : ", bright_b)
        print("[RG_diff] : ", RG_diff)
        print("threshold : ", self.user_threshold)

        if (self.BRIGHT_min > bright_b or bright_b > self.BRIGHT_max):
            self.reset_flag = True
            print("-> Camera reset [BRIGHT error] ")
        elif (self.RG_gab < RG_diff):
            self.reset_flag = True
            print("-> Camera reset [RG_gab error] ")

        end = time.time()
        sec = end - start
        start = end
        print("- Time( 전처리 ) : ", datetime.timedelta(seconds=sec))

        # Keep the original image checks, but never infer after a failed check.
        if self.reset_flag:
            self.result_container.config(bg='blue')
            self.result_label.config(text='FAIL', bg='blue')
            return

        # AI inference
        # anomaly_map, pred_label, pred_score, pred_mask = user_th_inference(
        #    inferencer, crop_wh_pil, self.user_threshold
        # )

        visual_map, pred_label, visual_score, pred_mask = user_th_inference(
             self.inferencer, crop_wh_pil, self.user_threshold
        )

        end = time.time()
        sec = end - start
        start = end
        print("- Time(anomaly) : ", datetime.timedelta(seconds=sec))

        #img_size_h = 448
        #img_size_w = 448
        blur_kernel = 5

        # crop_whs = norm_imgo[r_s:r_e, c_s:c_e, :]
        # crop_whs = cv2.resize(crop_whs, (crop_whs.shape[1] // 2, crop_whs.shape[0] // 2))
        # crop_whs = np.clip((crop_whs - 0.2) / (1 - 0.2), 0, 1)  # pre1
        #
        # crop_whs = np.power(crop_whs, 1.4)
        # crop_whs = cv2.resize(crop_whs, (800, r_e - r_s))
        # black_images = np.zeros((800, 800, 3), dtype=np.float32)
        #
        # hs = crop_whs.shape[0]
        # start_row = (800 - hs) // 2
        # end_row = start_row + hs
        #
        # x_offsets = (black_images.shape[1] - crop_whs.shape[1]) // 2
        # y_offsets = (black_images.shape[0] - crop_whs.shape[0]) // 2
        #
        # black_images[y_offsets:y_offsets + crop_whs.shape[0], x_offsets:x_offsets + crop_whs.shape[1], :] = crop_whs
        # black_images = cv2.resize(black_images, (img_size_w, img_size_h))
        #
        # if blur_kernel > 0 and blur_kernel % 2 == 1:
        #     black_images = cv2.GaussianBlur(black_images, (blur_kernel, blur_kernel), 0)
        # test_img = Image.fromarray((black_images * 255).astype(np.uint8))  # NumPy to PIL

        # image_sub = items["img_sub"].to(device)
        # subspace_map, subspace_score = extract_subspace_features(subspace_model, pil_img)

        # config_path = "./exported_models/X4024_1_600_672/json/model_config.json"
        # cache_dir = "./dinov2_cache"
        # model = load_model(config_path, cache_dir=cache_dir)

        # subspace_map, pred_label, subspace_score, pred_mask = user_th_inference_sub(self.subspace_model, test_img, 0.5)
        #
        # fused_map = fuse_visual_and_subspace_maps(visual_map, subspace_map, self.threshold_img, visual_score, self.sub_shift, self.sub_scale)
        # pred_score = compute_score_from_map(fused_map, mode="topk")

        # visual_scores.append(visual_score)
        # #subspace_scores.append(subspace_score)
        # visual_maps.append(visual_map)
        # #subspace_maps.append(subspace_map)
        # fused_maps.append(fused_map)
        # fused_scores.append(pred_score)

        # for v_map, s_map in zip(visual_maps, subspace_maps):
        #     fused_maps = fuse_visual_and_subspace_maps(v_map, s_map)
        #     #fused_maps.append(fused_map)
        #
        #     # VisualAD 방식대로 score 계산: max 기반
        #     pred_score = compute_score_from_map(fused_maps, mode="max")
        #     #pred_score.append(score)

        #pred_score = np.asarray(pred_score, dtype=np.float32)
        pred_score = visual_score
        # --------------------[결과창 : 전처리]-------------------------
        vimg = Image.fromarray((black_image * 255).astype(np.uint8))
        img_tmp1 = np.array(vimg)
        h = img_tmp1.shape[0]
        start_row = h // 2 - h // 8 + int(h * self.crop_v_offset)
        end_row = h // 2 + h // 8 + int(h * self.crop_v_offset)
        print("h1 : ", h)
        img_show1 = img_tmp1[start_row:end_row, :]
        img_show1 = cv2.cvtColor(img_show1, cv2.COLOR_RGB2BGR)
        img_show1 = cv2.resize(img_show1, (1000, 200))

        end = time.time()
        sec = end - start
        start = end
        print("- Time(image 1) : ", datetime.timedelta(seconds=sec))

        # --------------------[결과창 : Heatmap]-------------------------
        ref_image = cv2.resize(black_image, (336, 336))
        vimg = Image.fromarray((ref_image * 255).astype(np.uint8))

        print('vimg : ', ref_image.shape)
        print('visual_map : ', visual_map.shape)

        plt.close('all')
        plt.imshow(vimg, interpolation="none", alpha=0.5)
        #plt.imshow(fused_map, interpolation="none", alpha=0.5, cmap="jet")
        plt.imshow(visual_map, interpolation="none", alpha=0.5, cmap="jet")
        plt.axis('off')
        filepath_tmp = os.path.join(self.save_dir, f"{name}_heatmap.png")
        plt.savefig(filepath_tmp, bbox_inches='tight', pad_inches=0)
        img_show3 = cv2.imread(filepath_tmp)
        os.remove(filepath_tmp)
        img_show3 = np.array(img_show3)
        img_show3 = cv2.resize(img_show3, (img_size_w, img_size_h))
        h = img_show3.shape[0]
        start_row = h // 2 - h // 8 + int(h * self.crop_v_offset)
        end_row = h // 2 + h // 8 + int(h * self.crop_v_offset)
        print("h2 : ", h)
        img_show3 = img_show3[start_row:end_row, :]
        img_show3 = cv2.resize(img_show3, (1000, 200))

        # --------------------[결과창 : contour]-------------------------
        threshold_dec = self.user_threshold
        #all_amap = fused_map.squeeze().cpu().numpy()
        all_amap = visual_map.squeeze()
        binary_mask = (all_amap >= threshold_dec).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_overlay = img_tmp1.copy()
        fill_color = np.array([255, 0, 0], dtype=np.uint8)  # 빨강 (RGB)
        alpha = 0.3  # 투명도: 작을수록 더 투명함

        fill_layer = contour_overlay.copy()
        cv2.drawContours(fill_layer, contours, -1, tuple(fill_color.tolist()), thickness=cv2.FILLED)
        contour_overlay = cv2.addWeighted(fill_layer, alpha, contour_overlay, 1 - alpha, 0)
        cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), thickness=1)  # 얇은 빨강선

        # if pred_score >= self.user_threshold:
        #     print("==========NG===========")
        #     print(pred_score)
        #     print(self.user_threshold)
        #     fig_contour = plt.figure(figsize=(4, 4))
        #     plt.imshow(contour_overlay)
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.close(fig_contour)
        # else:
        #     print("==========OK===========")
        #     print(pred_score)
        #     print(self.user_threshold)
        #     fig_contour = plt.figure(figsize=(4, 4))
        #     plt.imshow(img_tmp1)
        #     plt.axis('off')
        #     plt.tight_layout()

        img_show2 = contour_overlay
        img_show2 = cv2.resize(img_show2, (img_size_w, img_size_h))
        img_show2 = img_show2[start_row:end_row, :]
        img_show2 = cv2.cvtColor(img_show2, cv2.COLOR_RGB2BGR)
        img_show2 = cv2.resize(img_show2, (1000, 200))

        if pred_score < self.user_threshold:
            img_show2 = img_show1

        end = time.time()
        sec = end - start
        start = end
        print("- Time(image 2) : ", datetime.timedelta(seconds=sec))

        if self.reset_flag == True and self.reset_flag_en == True:
            img_show1[:, :] = [100, 100, 100]
            img_show2[:, :] = [100, 100, 100]
            img_show3[:, :] = [100, 100, 100]
        else:
            combined_img = cv2.vconcat([img_show3, img_show2])
            filepath_combined = os.path.join(self.save_dir, f"{name}_resulf.jpg")
            cv2.imwrite(filepath_combined, combined_img)

        if (pred_score < self.user_threshold):
            result = "OK"
        else:
            result = "NG"

        self._update_label(self.r_label, img_show1)
        self._update_label(self.g_label, img_show2)
        self._update_label(self.b_label, img_show3)
        self.measure(result, bright_b)

        if self.AI_result_flag == 1:
            plt.close('all')
            fig = plt.figure(figsize=(12, 5))
            fig.suptitle(" anomaly score : " + str(pred_score) + " anomaly label : " + str(pred_label))

            plt.subplot(1, 3, 1)
            plt.imshow(vimg)
            plt.axis('off')
            plt.title("Test Image")

            plt.subplot(1, 3, 2)
            plt.imshow(vimg, interpolation="none", alpha=0.6)
            plt.imshow(visual_map, interpolation="none", alpha=0.4)
            plt.axis('off')
            plt.title("Heatmap")

            plt.subplot(1, 3, 3)
            plt.imshow(vimg, interpolation="none", alpha=0.6)
            plt.imshow(pred_mask, interpolation="none", alpha=0.4)
            plt.axis('off')
            plt.title("Image + Prediction Label")

            # replaced_img_fn = img_fn.replace("/", "_").replace(".bmp", "")
            # filepath = os.path.join(self.save_dir, f"{name}.PNG")
            # plt.savefig(filepath)
            plt.show()

        # =============== log 파일 저장 ================
        if self.Data_log_flag == 1:
            csvfilepath = os.path.join(self.CSV_path, "log.csv")
            file_exists = os.path.isfile(csvfilepath)  # 파일이 없으면 헤더 생성
            AutoExposure = 'OFF commanded; readback unavailable'
            Exposure = self.ExposureTime + ' requested; readback unavailable'
            Brightness = self.camera_report.get('after') or self.camera_report.get('before', {})
            Brightness = Brightness.get('brightness_control', 'unavailable')
            with open(csvfilepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:  # 처음 생성 시 헤더 작성
                    writer.writerow(
                        ["Date", "Model", "File name", "NG/OK ", "anomaly score", "anomaly label", "threshold", "밝기값", "RG_diff",
                         "Reset", "AutoExposure", "ExposureValue", "Brightness", "BRIGHT_min", "BRIGHT_max", "RG_gab"])
                writer.writerow([date, self.selected_model_name, f"{name}.bmp", result, str(pred_score), str(pred_label), self.user_threshold,
                                 bright_sampling, RG_diff, self.reset_flag, AutoExposure, Exposure, Brightness,
                                 self.BRIGHT_min, self.BRIGHT_max, self.RG_gab])  # CSV에 저장

        sec = time.time() - first_time
        print("= [Total Time]  : ", datetime.timedelta(seconds=sec))

        print("anomaly score : ", str(pred_score))
        print("anomaly label : ", str(pred_label))

    # -----------------------------
    # 종료
    # -----------------------------
    def on_close(self):
        self.running = False
        time.sleep(0.05)
        self.capture_thread.join(timeout=1)
        self.camera.close()
        self.inferencer.close()
        self.root.destroy()


# -----------------------------
# 실행
# -----------------------------


if __name__ == "__main__":
    print("import librarys......")

    # Splash 화면 닫기 (모델 선택 GUI 가 splash 에 가려지지 않도록)
    try:
        import pyi_splash

        pyi_splash.close()
    except:
        pass

    # 모델 선택 GUI 먼저 표시
    selected_model_name, model_path_vis, model_path_sub, runtime = show_model_selection_gui()

    if model_path_vis is None:
        print("모델이 선택되지 않았습니다. 프로그램을 종료합니다.")
        sys.exit(0)

    print(f"-vis 모델: {model_path_vis}")
    print(f"-sub 모델: {model_path_sub}")

    # 모델 로드
    inferencer = MOSAInferencer(model_path_vis, runtime)
    subspace_config_path = model_path_sub
    cache_dir_path = "./dinov2_cache"

    subspace_model, threshold_img = None, None
    # subspace_model, threshold_img = load_subspace_model(
    # subspace_config_path,
    # cache_dir=cache_dir_path,
    # device="cpu"
    # )

    # 메인 프로그램 실행
    root = tk.Tk()
    camera = None
    try:
        camera, microscope, error = camera_init()
        app = CameraGUI(root, camera, microscope, inferencer, subspace_model, threshold_img, selected_model_name)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    finally:
        if camera is not None:
            camera.close()
        inferencer.close()

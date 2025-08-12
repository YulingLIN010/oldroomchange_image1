
# utils/depth_onnx.py
# CPU 版單眼深度推論（MiDaS v2.1 Small, ONNX Runtime）
# 黑白 L8 PNG：白=近、黑=遠（或依需求在前端用 opacity 疊出深度效果）

import os
import io
import base64
import hashlib
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort

try:
    import cv2  # opencv-python-headless
except Exception as e:
    cv2 = None

DEFAULT_MODEL_URL = os.getenv("DEPTH_MODEL_URL",
    "https://github.com/isl-org/MiDaS/releases/download/v3_1/midas_v21_small.onnx"
)
DEFAULT_MODEL_PATH = os.getenv("DEPTH_MODEL_PATH",
    "models/midas_v21_small.onnx"
)
INPUT_SIZE = int(os.getenv("DEPTH_INPUT_SIZE", "256"))

_session = None

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _download(url: str, path: str):
    _ensure_dir(path)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def get_session(model_path: str = DEFAULT_MODEL_PATH):
    """Lazy 建立 ONNX Runtime Session（CPU 執行）。"""
    global _session
    if _session is not None:
        return _session
    if not os.path.exists(model_path):
        _download(DEFAULT_MODEL_URL, model_path)
    providers = ["CPUExecutionProvider"]
    _session = ort.InferenceSession(model_path, providers=providers)
    return _session

def _preprocess_pil(pil: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """將 PIL 影像轉成 MiDaS small 需要的 NCHW tensor（float32）。"""
    # 允許沒有 cv2 的情況：用 PIL 做 resize；有 cv2 則用它（速度較快）。
    if cv2 is not None:
        img = np.array(pil.convert("RGB"))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
    else:
        img = pil.convert("RGB").resize((size, size), Image.BICUBIC)
        img = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0)  # -> NCHW
    return img.astype(np.float32)

def infer_depth_map(pil: Image.Image) -> np.ndarray:
    """回傳 0..1 的 float32 深度圖（會自動放大回原圖尺寸）。"""
    sess = get_session()
    inp = _preprocess_pil(pil, INPUT_SIZE)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: inp})[0]  # shape: (1,1,H,W)
    depth = out[0, 0]
    # normalize 到 0..1
    depth = depth - depth.min()
    if depth.max() > 1e-6:
        depth = depth / depth.max()
    # resize 回到原圖尺寸
    w, h = pil.size
    if cv2 is not None:
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
        depth_resized = np.asarray(depth_img).astype(np.float32) / 255.0
    return depth_resized

def depth_to_png_bytes(depth01: np.ndarray) -> bytes:
    """將 0..1 深度圖轉 L8 PNG bytes。"""
    arr = (np.clip(depth01, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def depth_to_b64(depth01: np.ndarray) -> str:
    """回傳 base64 (無 data: 前綴)。"""
    by = depth_to_png_bytes(depth01)
    return base64.b64encode(by).decode("ascii")

def image_b64_to_pil(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def pil_to_b64(pil: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -*- coding: utf-8 -*-
"""
Flask 應用主程式（v1.7L 合併調整版）
對齊前端「單一最終遮罩」流程：
- /analyze：接受 multipart 或 JSON；回傳 initial 結構遮罩（combined_mask_b64）與（若可用）depth_b64。
- /masks/save：接受 multipart 或 JSON；最終只存一張 L 遮罩（白=可編輯、黑=保護）。
- /render/batch：沿用你既有生圖流程，套用最終遮罩；支援 logo 疊圖。
- 其它舊端點與靜態檔路由保留，兼容既有客戶端。

遮罩語意統一：
- 內部 L 遮罩（單通道 L8）：白(255)=可編輯、黑(0)=保護
- OpenAI 透明遮罩：透明(alpha=0)=可編輯、不透明(alpha=255)=保護
- 前端 UI 遮罩（若用）：白=鎖定、透明=可改（僅做顯示與手繪，最終仍轉成 L 遮罩）
"""
import os, io, uuid, time, threading, base64, json, traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from PIL import Image, ImageOps
import numpy as np
import cv2

# ==== 相依匯入（優先 utils.*；若沒有 utils，再嘗試同層） ====
try:
    from utils.dalle_api import edit_image_with_mask, generate_image
except ModuleNotFoundError:
    from dalle_api import edit_image_with_mask, generate_image

try:
    from utils.image_logo import add_logo
except ModuleNotFoundError:
    from image_logo import add_logo

try:
    from utils.prompt_templates import load_styles, build_style_prompt, make_prompt
except ModuleNotFoundError:
    def load_styles(): return [{"name":"現代風 Modern"}]
    def build_style_prompt(style, colors=None, enforce_hard_rules=True):
        pal = colors or {}
        return (
            f"室內設計風格：{style}。保持原相機視角與透視，不改變樑柱、門窗與結構。"
            f"僅在遮罩圈選區更新表面材質與家具，色系：{pal}。高品質、寫實、無文字。"
        )
    def make_prompt(*args, **kwargs): return build_style_prompt(*args, **kwargs)

# 深度（已停用 MiDaS）
HAS_DEPTH = False  # /analyze 不回 depth_b64；其餘功能不變

# ==== 目錄設定 ====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MASK_DIR = os.path.join(BASE_DIR, "masks")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo", "LOGO.png")

for d in (JOBS_DIR, UPLOAD_DIR, RESULT_DIR, MASK_DIR, STATIC_DIR, os.path.dirname(LOGO_PATH)):
    os.makedirs(d, exist_ok=True)

# ==== Flask & CORS ====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB 限制（與前端一致）
CORS(app,
     resources={r"/*": {"origins": ["*"]}},
     supports_credentials=False,
     expose_headers="*",
     allow_headers=["Content-Type","Authorization"],
     methods=["GET","POST","OPTIONS"])

# ---- Global JSON error handler ----
@app.errorhandler(Exception)
def handle_any_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    app.logger.exception("Unhandled error")
    return jsonify({"error": str(e), "trace": traceback.format_exc()}), code

def _public_base_url():
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    if host:
        return f"{proto}://{host}".rstrip("/")
    return request.url_root[:-1] if request.url_root.endswith("/") else request.url_root

def _abs_url(path: str) -> str:
    if not path: return path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return _public_base_url() + path

# ==== 舊任務狀態（向下相容） ====
@dataclass
class Task:
    status: str
    error: Optional[str] = None
    original_image_url: Optional[str] = None
    mask_url: Optional[str] = None
    styled_image_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)

TASKS: Dict[str, Task] = {}
TASKS_LOCK = threading.Lock()

# ==== 工具：EXIF 方向修正 ====
def correct_image_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img.getexif()
        orientation = exif.get(0x0112)
        if orientation == 3:
            return img.rotate(180, expand=True)
        if orientation == 6:
            return img.rotate(270, expand=True)
        if orientation == 8:
            return img.rotate(90, expand=True)
    except Exception:
        pass
    return img

# ==== 工具：儲存上傳圖（FileStorage）====
def save_upload_image(file_storage, out_path: str) -> str:
    im = Image.open(file_storage.stream).convert("RGB")
    im = correct_image_orientation(im)
    im.save(out_path, "JPEG", quality=92)
    return out_path

def save_b64_image(image_b64: str, out_path: str) -> str:
    if image_b64.startswith("data:"):
        _, b64 = image_b64.split(",", 1)
    else:
        b64 = image_b64
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = correct_image_orientation(im)
    im.save(out_path, "JPEG", quality=92)
    return out_path

# ==== 遮罩轉換 ====
def build_mask_l(image_path: str, mode: str = "smart", opts: dict = None) -> np.ndarray:
    """
    產生 L 遮罩（白=可改，黑=保護）；回傳 numpy uint8 (H,W)。
    mode: 'full' → 全白；'smart' → 邊緣/窗戶/邊框保護
    """
    img = cv2.imread(image_path)
    if img is None:
        app.logger.error(f"build_mask_l: failed to read image at {image_path}; fallback to full-white mask")
        try:
            _im = Image.open(image_path).convert("RGB")
            w, h = _im.size
        except Exception:
            w, h = 1024, 768
        return np.full((h, w), 255, dtype=np.uint8)
    h, w = img.shape[:2]
    opts = opts or {}
    edge_thresh = int(opts.get("edge_thresh", 28))
    dilate_px = int(opts.get("dilate_px", 6))
    margin_ratio = float(opts.get("margin_ratio", 0.04))
    protect_windows = bool(opts.get("protect_windows", True))
    window_bright_thresh = int(opts.get("window_bright_thresh", 215))

    if mode in ("all", "full"):
        return np.full((h, w), 255, dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 邊緣保護（黑）
    t1, t2 = edge_thresh, max(edge_thresh * 3, edge_thresh + 1)
    edges = cv2.Canny(gray, t1, t2)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[edges > 0] = 0

    # 2) 膨脹保護（擴大黑區）
    k = max(1, int(dilate_px))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask, kernel, iterations=1)

    # 3) 高亮窗戶保護（黑）
    if protect_windows:
        _, bright = cv2.threshold(gray, window_bright_thresh, 255, cv2.THRESH_BINARY)
        win_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), max(3, k)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, win_kernel, iterations=1)
        mask[bright > 0] = 0

    # 4) 外框保護（黑）
    m = int(round(min(h, w) * margin_ratio))
    if m > 0:
        mask[:m, :] = 0; mask[-m:, :] = 0; mask[:, :m] = 0; mask[:, -m:] = 0

    return mask

def l_to_ui_rgba(mask_l: np.ndarray) -> Image.Image:
    """L 遮罩（白=可改、黑=保護）→ 前端 UI 遮罩（白=鎖、透明=可改）"""
    h, w = mask_l.shape[:2]
    alpha = 255 - mask_l  # 白→0、黑→255
    rgba = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba

def ui_png_to_l(ui_png_path: str) -> np.ndarray:
    """前端 UI PNG（白=鎖、透明=可改）→ L 遮罩（白=可改、黑=保護）"""
    im = Image.open(ui_png_path).convert("RGBA")
    a = np.array(im.split()[-1])  # alpha
    l = np.where(a > 0, 0, 255).astype(np.uint8)
    return l

def b64_to_l(b64png: str, size_wh: Optional[tuple]=None) -> np.ndarray:
    """base64 PNG → L 遮罩；若 size_wh 指定，則 NEAREST resize。"""
    if b64png.startswith("data:"):
        _, b64 = b64png.split(",", 1)
    else:
        b64 = b64png
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw))
    if im.mode != "L":
        im = im.convert("L")
    if size_wh and im.size != size_wh:
        im = im.resize(size_wh, Image.NEAREST)
    arr = np.array(im)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    return arr

def l_to_alpha_png_path(mask_l: np.ndarray, ref_image_path: str, out_path: str) -> str:
    """L 遮罩（白=可改、黑=保護）→ OpenAI 透明遮罩（alpha=0 可編輯；alpha=255 保護）"""
    with Image.open(ref_image_path) as ref:
        w, h = ref.size
    if mask_l.shape[:2] != (h, w):
        mask_l = cv2.resize(mask_l, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = 255 - mask_l  # 白→0、黑→255
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    rgba.save(out_path, "PNG")
    return out_path

def pil_to_b64(im: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ==== Jobs 記憶體索引 ====
@dataclass
class Job:
    id: str
    dir: str
    original: str              # <dir>/original.jpg
    mask_l_path: Optional[str] = None  # 內部 L 遮罩路徑（白=可改）
    ui_mask_path: Optional[str] = None # 前端 UI 遮罩（白=鎖）

JOBS: Dict[str, Job] = {}

def _new_job() -> Job:
    jid = str(uuid.uuid4())
    jdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(jdir, exist_ok=True)
    os.makedirs(os.path.join(jdir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(jdir, "outputs"), exist_ok=True)
    return Job(id=jid, dir=jdir, original="")

def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)

# ==== API: /analyze ====
@app.post("/analyze")
def analyze():
    """
    接受 multipart（image）或 JSON（image_b64 / image_data_url）
    建立 jobId，存 original.jpg，產初始 L 遮罩與 UI 遮罩；
    回傳：
    {
      jobId,
      original,                                 # 原圖 URL
      masks: { editable_surface: <ui_mask_url> },# 保留舊欄位（相容）
      combined_mask_b64,                        # L 遮罩（白=可編）
      depth_b64?                                # 若可用則附上
    }
    """
    image_path = None
    # multipart
    if "image" in request.files and request.files["image"].filename:
        job = _new_job()
        image_path = os.path.join(job.dir, "original.jpg")
        save_upload_image(request.files["image"], image_path)
    else:
        # JSON
        data = request.get_json(force=True, silent=True) or {}
        image_b64 = data.get("image_b64")
        if not image_b64 and data.get("image_data_url"):
            durl = data["image_data_url"]
            idx = durl.find("base64,")
            if idx >= 0:
                image_b64 = durl[idx+7:]
        if not image_b64:
            return jsonify({"error": "image or image_b64 is required"}), 400
        job = _new_job()
        image_path = os.path.join(job.dir, "original.jpg")
        save_b64_image(image_b64, image_path)

    job.original = image_path

    # 初始 L 遮罩（S）
    l = build_mask_l(image_path, mode="smart", opts={
        "edge_thresh": 28, "dilate_px": 6, "margin_ratio": 0.04,
        "protect_windows": True, "window_bright_thresh": 215
    })
    l_path = os.path.join(job.dir, "masks", "initial_L.png")
    Image.fromarray(l, mode="L").save(l_path, "PNG")
    job.mask_l_path = l_path

    # UI 遮罩（白=鎖、透明=可改），僅作參考（若前端需要）
    ui_im = l_to_ui_rgba(l)
    ui_path = os.path.join(job.dir, "masks", "ui_mask.png")
    ui_im.save(ui_path, "PNG")
    job.ui_mask_path = ui_path

    JOBS[job.id] = job

    # 深度
    depth_b64 = None
    if HAS_DEPTH:
        try:
            pil = Image.open(image_path).convert("RGB")
            d01 = infer_depth_map(pil)
            depth_b64 = depth_to_b64(d01)
        except Exception:
            depth_b64 = None

    return jsonify({
        "jobId": job.id,
        "original": _abs_url(f"/jobs/{job.id}/original.jpg"),
        "masks": {"editable_surface": _abs_url(f"/jobs/{job.id}/masks/ui_mask.png")},
        "combined_mask_b64": pil_to_b64(Image.fromarray(l, mode="L"), "PNG"),
        "depth_b64": depth_b64
    })

# ==== API: /masks/save ====
@app.post("/masks/save")
def save_mask():
    """
    接收「最終遮罩」並儲存為 L 遮罩（白=可編輯、黑=保護）。
    支援兩種呼叫：
    1) multipart: jobId + mask(file: PNG，前端 UI 遮罩 or L 遮罩)
    2) JSON: { job_id|jobId, mask_final_b64|combined_mask_b64|mask_b64 }
    """
    job_id = None
    mask_l = None

    if request.files:
        job_id = request.form.get("jobId") or request.form.get("job_id")
        if not job_id or job_id not in JOBS:
            return jsonify({"error": "invalid jobId"}), 400
        if "mask" not in request.files:
            return jsonify({"error": "mask file required"}), 400
        job = JOBS[job_id]
        tmp = os.path.join(job.dir, "masks", "incoming.png")
        request.files["mask"].save(tmp)
        # 嘗試判斷：若是 UI PNG（看 alpha），轉 L；若是 L 灰階，直接讀
        im = Image.open(tmp).convert("RGBA")
        if im.split()[-1].getextrema() != (255, 255):  # 有透明度 → 視為 UI
            mask_l = ui_png_to_l(tmp)
        else:
            mask_l = np.array(im.convert("L"))
            mask_l = np.where(mask_l > 127, 255, 0).astype(np.uint8)
    else:
        data = request.get_json(force=True, silent=True) or {}
        job_id = data.get("job_id") or data.get("jobId")
        if not job_id or job_id not in JOBS:
            return jsonify({"error": "invalid jobId"}), 400
        b64 = data.get("mask_final_b64") or data.get("combined_mask_b64") or data.get("mask_b64")
        if not b64:
            return jsonify({"error": "missing mask"}), 400
        # decode → L
        job = JOBS[job_id]
        with Image.open(job.original) as ref:
            W, H = ref.size
        mask_l = b64_to_l(b64, size_wh=(W, H))

    l_path = os.path.join(JOBS[job_id].dir, "masks", "final_L.png")
    Image.fromarray(mask_l, mode="L").save(l_path, "PNG")
    JOBS[job_id].mask_l_path = l_path
    return jsonify({"ok": True, "jobId": job_id})

# ==== API: /render/batch ====
@app.post("/render/batch")
def render_batch():
    """
    JSON: { jobId, styles: [..max3], palette:{main,acc1,acc2,acc3}, logo:{pos,scale,opacity} }
    回傳：{ images: [{style,url}], qc: {...} }
    """
    data = request.get_json(force=True, silent=False)
    job_id = (data.get("jobId") or "").strip()
    styles: List[str] = data.get("styles") or []
    palette = data.get("palette") or {}
    logo = data.get("logo") or {}

    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid jobId"}), 400
    if not styles:
        return jsonify({"error": "styles required"}), 400

    job = JOBS[job_id]
    if not job.mask_l_path or not os.path.exists(job.mask_l_path):
        # 若使用者未儲存，沿用 initial
        init_path = os.path.join(job.dir, "masks", "initial_L.png")
        if not os.path.exists(init_path):
            return jsonify({"error": "mask not ready"}), 400
        job.mask_l_path = init_path

    # 準備 OpenAI 透明遮罩
    alpha_path = os.path.join(job.dir, "masks", "oa_alpha.png")
    l = cv2.imread(job.mask_l_path, cv2.IMREAD_GRAYSCALE)
    if l is None: return jsonify({"error": "bad mask"}), 400
    l_to_alpha_png_path(l, job.original, alpha_path)

    results = []
    qc = {}
    for style in styles[:3]:
        prompt = build_style_prompt(style, colors=palette, enforce_hard_rules=True)
        out_png = os.path.join(job.dir, "outputs", f"{style}_{uuid.uuid4().hex[:6]}.png")
        try:
            img_bytes = edit_image_with_mask(job.original, alpha_path, prompt, size="1024x1024")
            # 存暫存
            tmp_path = os.path.join(job.dir, "outputs", f"tmp_{uuid.uuid4().hex}.png")
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            # 疊 LOGO（參數可選）
            pos = logo.get("pos", "bottom-right")
            scale = float(logo.get("scale", 0.18) or 0.18)
            opacity = float(logo.get("opacity", 0.9) or 0.9)
            try:
                if os.path.exists(LOGO_PATH):
                    add_logo(tmp_path, LOGO_PATH, out_png, logo_ratio=scale, position=pos, opacity=opacity)
                    os.remove(tmp_path)
                else:
                    out_png = tmp_path
            except Exception:
                out_png = tmp_path
            results.append({"style": style, "url": _abs_url(f"/jobs/{job.id}/outputs/{os.path.basename(out_png)}")})
            qc[style] = {"keypoint_error": ""}
        except Exception as e:
            results.append({"style": style, "url": "", "error": str(e)})
            qc[style] = {"keypoint_error": ""}

    return jsonify({"images": results, "qc": qc})

# ==== API: /upload/mask ====
@app.post("/upload/mask")
def upload_mask():
    """
    家具編修遮罩上傳。FormData: jobId, mask(file png)
    回：{ maskUrl: "/jobs/<id>/masks/user/<name>.png" }
    """
    job_id = request.form.get("jobId", "").strip()
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid jobId"}), 400
    if "mask" not in request.files:
        return jsonify({"error": "mask file required"}), 400
    job = JOBS[job_id]
    user_dir = os.path.join(job.dir, "masks", "user")
    os.makedirs(user_dir, exist_ok=True)
    fname = f"user_{uuid.uuid4().hex}.png"
    fpath = os.path.join(user_dir, fname)
    request.files["mask"].save(fpath)
    return jsonify({"maskUrl": _abs_url(f"/jobs/{job.id}/masks/user/{fname}")})

# ==== API: /render/furniture-edit ====
@app.post("/render/furniture-edit")
def furniture_edit():
    """
    JSON: {
      jobId: string,
      baseImageId?: string,      # /jobs/... 或 http(s)://.../jobs/... 或 style token
      mask: string,              # /jobs/... UI 遮罩（白=鎖、透明=可改）或灰階黑白遮罩
      type: "replace"|"recolor",
      spec?: string,             # replace 用
      color?: string             # recolor 用
    }
    """
    data = request.get_json(force=True, silent=False)
    job_id = (data.get("jobId") or "").strip()
    if not job_id:
        return jsonify({"error": "jobId required"}), 400
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    # 解析遮罩路徑
    mask_url = data.get("mask") or ""

    def to_local_path(p: str) -> Optional[str]:
        if not p:
            return None
        if p.startswith("/"):
            p2 = p.lstrip("/")
            return p2 if os.path.exists(p2) else None
        if p.startswith("http"):
            i = p.find("/jobs/")
            if i != -1:
                p2 = p[i+1:]
                return p2 if os.path.exists(p2) else None
        return None

    mask_path = to_local_path(mask_url)
    if not mask_path:
        return jsonify({"error": "mask not found"}), 400

    # 基底圖：baseImageId > original.jpg
    baseImageId = (data.get("baseImageId") or "").strip()
    base_path = None
    if baseImageId:
        base_path = to_local_path(baseImageId)
        if not base_path and os.path.isdir(os.path.join(job.dir, "outputs")):
            outs = sorted(
                [os.path.join(job.dir, "outputs", f) for f in os.listdir(os.path.join(job.dir, "outputs")) if baseImageId in f],
                key=lambda p: os.path.getmtime(p),
                reverse=True
            )
            if outs:
                base_path = outs[0]
    if not base_path:
        base_path = job.original

    # 轉為 OpenAI 用透明遮罩
    im = Image.open(mask_path).convert("RGBA")
    if im.mode == "RGBA" and im.split()[-1].getextrema() != (255, 255):
        # UI PNG → L
        l = ui_png_to_l(mask_path)
    else:
        l = np.array(im.convert("L"))
        l = np.where(l > 127, 255, 0).astype(np.uint8)
    alpha_png = os.path.join(job.dir, "alpha_runtime.png")
    l_to_alpha_png_path(l, base_path, alpha_png)

    # 組提示詞
    t = (data.get("type") or "replace").lower()
    if t == "recolor":
        color = data.get("color", "#1E3A8A")
        prompt = f"Recolor the selected furniture/object to {color}. Keep camera angle, structure, windows/doors positions, and layout exactly the same. Only modify within the mask."
    else:
        spec = data.get("spec", "modern sofa with metal legs")
        prompt = f"Replace the selected object with: {spec}. Keep camera angle, structure, windows/doors positions, and layout exactly the same. Only modify within the mask."

    try:
        img_bytes = edit_image_with_mask(base_path, alpha_png, prompt, size="1024x1024")
    except Exception as e:
        return jsonify({"error": f"OpenAI edit failed: {e}"}), 500

    out_dir = os.path.join(job.dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"edit_{uuid.uuid4().hex}.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)

    return jsonify({"imageUrl": _abs_url(f"/jobs/{job.id}/outputs/{os.path.basename(out_path)}")})

@app.get("/styles")
def styles_endpoint():
    return jsonify(load_styles())

# ==== 舊版 /generate + /status（保留） ====
def _to_alpha(mask_l_path: str, ref_image_path: str) -> str:
    l = cv2.imread(mask_l_path, cv2.IMREAD_GRAYSCALE)
    out = os.path.join(MASK_DIR, f"{uuid.uuid4()}_alpha.png")
    l_to_alpha_png_path(l, ref_image_path, out)
    return out

def _work_task(task_id: str, image_path: str, mask_l_path: str, style: str, colors: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task: return
        task.status = "pending"
    try:
        def _parse_colors(s: str):
            s = (s or "").strip()
            if not s: return {}
            if s.startswith("#"):
                return {"main": s}
            parts = [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]
            keys = ["main","acc1","acc2","acc3"]
            return {k:v for k,v in zip(keys, parts)}

        colors_dict = _parse_colors(colors)
        prompt = build_style_prompt(style, colors=colors_dict, enforce_hard_rules=True)

        alpha_mask = _to_alpha(mask_l_path, image_path)
        tmp_out = os.path.join(RESULT_DIR, f"{uuid.uuid4()}.png")
        img_bytes = edit_image_with_mask(image_path, alpha_mask, prompt, size="1024x1024")
        with open(tmp_out, "wb") as f:
            f.write(img_bytes)
        final_path = os.path.join(RESULT_DIR, f"{uuid.uuid4()}.png")
        try:
            if os.path.exists(LOGO_PATH):
                add_logo(tmp_out, LOGO_PATH, final_path)
                styled_path = final_path
            else:
                styled_path = tmp_out
        except Exception:
            styled_path = tmp_out
        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if not task: return
            task.status = "completed"
            task.original_image_url = f"/uploads/{os.path.basename(image_path)}"
            task.styled_image_url = f"/results/{os.path.basename(styled_path)}"
            task.mask_url = f"/masks/{os.path.basename(alpha_mask)}"
    except Exception as e:
        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if not task: return
            task.status = "failed"
            task.error = str(e)

@app.post("/generate")
def generate_legacy():
    data = request.get_json(force=True, silent=False) or {}
    image_b64 = data.get("image_base64")
    style = (data.get("style") or "").strip()
    colors = (data.get("colors") or "").strip()
    mask_mode = data.get("mask", "smart")
    mask_opts = data.get("mask_options", {})

    if not image_b64:
        return jsonify({"status": "failed", "error": "image_base64 is required"}), 400
    if not style or not colors:
        return jsonify({"status": "failed", "error": "style and colors are required"}), 400

    # 存原圖
    if image_b64.startswith("data:"):
        _, b64 = image_b64.split(",", 1)
    else:
        b64 = image_b64
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = correct_image_orientation(img)
    img_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    img.save(img_path, "JPEG", quality=92)

    # 產 L 遮罩（白=可改）
    l = build_mask_l(img_path, mode=mask_mode, opts=mask_opts)
    l_path = os.path.join(MASK_DIR, f"{uuid.uuid4()}_L.png")
    Image.fromarray(l, mode="L").save(l_path, "PNG")

    task_id = str(uuid.uuid4())
    with TASKS_LOCK:
        TASKS[task_id] = Task(status="queued", original_image_url=f"/uploads/{os.path.basename(img_path)}",
                              mask_url=f"/masks/{os.path.basename(l_path)}")
    threading.Thread(target=_work_task, args=(task_id, img_path, l_path, style, colors), daemon=True).start()
    return jsonify({"status": "queued", "task_id": task_id})

@app.get("/status/<task_id>")
def status_legacy(task_id):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"status": "failed", "error": "task not found"}), 404

        def _abs(v):
            try:
                return _abs_url(v) if v else v
            except Exception:
                return v
        return jsonify({
            "status": task.status,
            "error": task.error,
            "original_image_url": _abs(task.original_image_url),
            "mask_url": _abs(task.mask_url),
            "styled_image_url": _abs(task.styled_image_url),
            "created_at": task.created_at,
        })

# ==== 健康檢查 ====
@app.get("/healthz")
def healthz():
    return ("", 204)

# ==== 靜態檔案 ====
@app.route("/jobs/<job_id>/<path:filename>")
def get_job_file(job_id, filename):
    jdir = _job_dir(job_id)
    if not os.path.isdir(jdir):
        abort(404)
    return send_from_directory(jdir, filename)

@app.route("/uploads/<path:filename>")
def get_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/results/<path:filename>")
def get_result(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/masks/<path:filename>")
def get_mask(filename):
    return send_from_directory(MASK_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)


# -*- coding: utf-8 -*-
"""
app.py — Final (MiDaS-OFF) consolidated backend
- No depth output (HAS_DEPTH=False)
- Unified mask semantics: L mask (white=editable, black=protect) -> OpenAI alpha (transparent=editable)
- /render/furniture-edit returns BOTH imageUrl and image_b64
- /render/batch returns BOTH url and image_b64 for each result
"""

import os, io, uuid, time, base64, json, traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from PIL import Image
import numpy as np
import cv2

# ===== helper imports (robust fallbacks) =====
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
    try:
        from prompt_templates import load_styles, build_style_prompt, make_prompt
    except ModuleNotFoundError:
        def load_styles(): return [{"name":"現代風 Modern"}]
        def build_style_prompt(style, colors=None, enforce_hard_rules=True):
            pal = colors or {}
            return (
                f"室內設計風格：{style}。保持原相機視角與透視，不改變樑柱、門窗與結構。"
                f"僅在遮罩圈選區更新表面材質與家具，統一色系：{pal}。高品質、寫實、無文字。"
            )
        def make_prompt(*args, **kwargs): return build_style_prompt(*args, **kwargs)

# ===== depth OFF =====
HAS_DEPTH = False

# ===== Paths =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MASK_DIR = os.path.join(BASE_DIR, "masks")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo", "LOGO.png")

for d in (JOBS_DIR, UPLOAD_DIR, RESULT_DIR, MASK_DIR, STATIC_DIR, os.path.dirname(LOGO_PATH)):
    os.makedirs(d, exist_ok=True)

# ===== Flask =====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB upload limit
CORS(app,
     resources={r"/*": {"origins": ["*"]}},
     supports_credentials=False,
     expose_headers="*",
     allow_headers=["Content-Type","Authorization"],
     methods=["GET","POST","OPTIONS"])

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

# ===== helpers =====
def correct_image_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img.getexif()
        orientation = exif.get(0x0112)
        if orientation == 3:   return img.rotate(180, expand=True)
        if orientation == 6:   return img.rotate(270, expand=True)
        if orientation == 8:   return img.rotate(90, expand=True)
    except Exception:
        pass
    return img

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

# --- mask conversions ---
def build_mask_l(image_path: str, mode: str = "smart", opts: dict = None) -> np.ndarray:
    """
    Produce initial L mask (white=editable, black=protect). Return uint8 (H,W).
    Simple heuristic: protect edges, bright windows and margins.
    """
    img = cv2.imread(image_path)
    if img is None:
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

    # edges protect (black)
    t1, t2 = edge_thresh, max(edge_thresh * 3, edge_thresh + 1)
    edges = cv2.Canny(gray, t1, t2)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[edges > 0] = 0

    # dilate black region
    k = max(1, int(dilate_px))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask, kernel, iterations=1)

    # bright window protect
    if protect_windows:
        _, bright = cv2.threshold(gray, window_bright_thresh, 255, cv2.THRESH_BINARY)
        win_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), max(3, k)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, win_kernel, iterations=1)
        mask[bright > 0] = 0

    # frame margin protect
    m = int(round(min(h, w) * margin_ratio))
    if m > 0:
        mask[:m, :] = 0; mask[-m:, :] = 0; mask[:, :m] = 0; mask[:, -m:] = 0

    return mask

def l_to_ui_rgba(mask_l: np.ndarray) -> Image.Image:
    """L (white=editable) -> UI overlay (white=lock, transparent=editable)"""
    h, w = mask_l.shape[:2]
    alpha = 255 - mask_l  # white->0 (editable), black->255
    rgba = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba

def ui_png_to_l(ui_png_path: str) -> np.ndarray:
    """UI PNG (white=lock, transparent=editable) -> L (white=editable)"""
    im = Image.open(ui_png_path).convert("RGBA")
    a = np.array(im.split()[-1])  # alpha
    l = np.where(a > 0, 0, 255).astype(np.uint8)
    return l

def b64_to_l(b64png: str, size_wh: Optional[tuple]=None) -> np.ndarray:
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
    """L (white=editable) -> OpenAI alpha (transparent=editable)"""
    with Image.open(ref_image_path) as ref:
        w, h = ref.size
    if mask_l.shape[:2] != (h, w):
        mask_l = cv2.resize(mask_l, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = 255 - mask_l
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    rgba.save(out_path, "PNG")
    return out_path

def pil_to_b64(im: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ===== Job state =====
@dataclass
class Job:
    id: str
    dir: str
    original: str
    mask_l_path: Optional[str] = None
    ui_mask_path: Optional[str] = None

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

# ===== Routes =====
@app.get("/healthz")
def healthz():
    return ("", 204)

@app.post("/analyze")
def analyze():
    """Accept image via multipart (image) or JSON (image_b64/image_data_url).
       Produce initial L mask and UI overlay. Depth disabled.
       Return: { jobId, original, masks.editable_surface, combined_mask_b64 }
    """
    # read input
    if "image" in request.files and request.files["image"].filename:
        job = _new_job()
        image_path = os.path.join(job.dir, "original.jpg")
        save_upload_image(request.files["image"], image_path)
    else:
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

    # initial L mask
    l = build_mask_l(image_path, mode="smart", opts={
        "edge_thresh": 28, "dilate_px": 6, "margin_ratio": 0.04,
        "protect_windows": True, "window_bright_thresh": 215
    })
    l_path = os.path.join(job.dir, "masks", "initial_L.png")
    Image.fromarray(l, mode="L").save(l_path, "PNG")
    job.mask_l_path = l_path

    # UI mask preview
    ui_im = l_to_ui_rgba(l)
    ui_path = os.path.join(job.dir, "masks", "ui_mask.png")
    ui_im.save(ui_path, "PNG")
    job.ui_mask_path = ui_path

    JOBS[job.id] = job

    return jsonify({
        "jobId": job.id,
        "original": _abs_url(f"/jobs/{job.id}/original.jpg"),
        "masks": {"editable_surface": _abs_url(f"/jobs/{job.id}/masks/ui_mask.png")},
        "combined_mask_b64": pil_to_b64(Image.fromarray(l, mode="L"), "PNG")
    })

@app.post("/masks/save")
def save_mask():
    """Save final L mask (white=editable). Accept multipart (jobId+mask) or JSON (mask_final_b64/combined_mask_b64/mask_b64)."""
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
        im = Image.open(tmp).convert("RGBA")
        # if has alpha not all 255 -> UI mask
        if im.split()[-1].getextrema() != (255, 255):
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
        job = JOBS[job_id]
        with Image.open(job.original) as ref:
            W, H = ref.size
        mask_l = b64_to_l(b64, size_wh=(W, H))

    l_path = os.path.join(JOBS[job_id].dir, "masks", "final_L.png")
    Image.fromarray(mask_l, mode="L").save(l_path, "PNG")
    JOBS[job_id].mask_l_path = l_path
    return jsonify({"ok": True, "jobId": job_id})

@app.post("/upload/mask")
def upload_mask():
    """Upload a local mask (PNG) for furniture editing. FormData: jobId, mask(file). Return: maskUrl"""
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

@app.post("/render/batch")
def render_batch():
    """JSON: { jobId, styles:[..<=3], palette:{main,acc1,acc2,acc3}, logo:{pos,scale,opacity} }.
       Return: { images: [{style, url, image_b64}], qc:{} }
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
    # final mask
    if not job.mask_l_path or not os.path.exists(job.mask_l_path):
        init_path = os.path.join(job.dir, "masks", "initial_L.png")
        if not os.path.exists(init_path):
            return jsonify({"error": "mask not ready"}), 400
        job.mask_l_path = init_path

    # prepare OpenAI alpha
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
            # save temp then logo
            tmp_path = os.path.join(job.dir, "outputs", f"tmp_{uuid.uuid4().hex}.png")
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            pos = logo.get("pos", "bottom-right")
            scale = float(logo.get("scale", 0.18) or 0.18)
            opacity = float(logo.get("opacity", 0.9) or 0.9)
            try:
                if os.path.exists(LOGO_PATH):
                    add_logo(tmp_path, LOGO_PATH, out_png, logo_ratio=scale, position=pos, opacity=opacity)
                    os.remove(tmp_path)
                    with open(out_png, "rb") as _f:
                        b64 = base64.b64encode(_f.read()).decode("ascii")
                else:
                    out_png = tmp_path
                    with open(tmp_path, "rb") as _f:
                        b64 = base64.b64encode(_f.read()).decode("ascii")
            except Exception:
                with open(tmp_path, "rb") as _f:
                    b64 = base64.b64encode(_f.read()).decode("ascii")
                out_png = tmp_path

            results.append({
                "style": style,
                "url": _abs_url(f"/jobs/{job.id}/outputs/{os.path.basename(out_png)}"),
                "image_b64": b64
            })
            qc[style] = {"keypoint_error": ""}
        except Exception as e:
            results.append({"style": style, "url": "", "error": str(e)})
            qc[style] = {"keypoint_error": ""}

    return jsonify({"images": results, "qc": qc})

@app.post("/render/furniture-edit")
def furniture_edit():
    """JSON: { jobId, mask: <URL or /jobs/...>, type: 'replace'|'recolor', spec? | color? }"""
    data = request.get_json(force=True, silent=False)
    job_id = (data.get("jobId") or "").strip()
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid jobId"}), 400
    job = JOBS[job_id]

    # resolve mask path
    def to_local_path(p: str) -> Optional[str]:
        if not p: return None
        if p.startswith("/"):
            p2 = p.lstrip("/")
            return p2 if os.path.exists(p2) else None
        if p.startswith("http"):
            i = p.find("/jobs/")
            if i != -1:
                p2 = p[i+1:]
                return p2 if os.path.exists(p2) else None
        return None

    mask_url = data.get("mask") or ""
    mask_path = to_local_path(mask_url)
    if not mask_path:
        return jsonify({"error": "mask not found"}), 400

    # choose base image (latest output or original)
    base_path = job.original

    # convert mask to OpenAI alpha
    im = Image.open(mask_path).convert("RGBA")
    if im.mode == "RGBA" and im.split()[-1].getextrema() != (255, 255):
        l = ui_png_to_l(mask_path)
    else:
        l = np.array(im.convert("L"))
        l = np.where(l > 127, 255, 0).astype(np.uint8)
    alpha_png = os.path.join(job.dir, "masks", f"alpha_{uuid.uuid4().hex}.png")
    l_to_alpha_png_path(l, base_path, alpha_png)

    # prompt
    t = (data.get("type") or "replace").lower()
    if t == "recolor":
        color = data.get("color", "#1E3A8A")
        prompt = f"Recolor the selected furniture/object to {color}. Keep camera angle, structure, and layout exactly the same. Only modify within the mask."
    else:
        spec = data.get("spec", "modern sofa with metal legs")
        prompt = f"Replace the selected object with: {spec}. Keep camera angle, structure, and layout exactly the same. Only modify within the mask."

    try:
        img_bytes = edit_image_with_mask(base_path, alpha_png, prompt, size="1024x1024")
    except Exception as e:
        return jsonify({"error": f"OpenAI edit failed: {e}"}), 500

    out_dir = os.path.join(job.dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"edit_{uuid.uuid4().hex}.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)

    with open(out_path, "rb") as _f:
        b64 = base64.b64encode(_f.read()).decode("ascii")

    return jsonify({
        "imageUrl": _abs_url(f"/jobs/{job.id}/outputs/{os.path.basename(out_path)}"),
        "image_b64": b64
    })

@app.get("/styles")
def styles_endpoint():
    return jsonify(load_styles())

# ===== static / files =====
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

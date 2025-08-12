
# -*- coding: utf-8 -*-
"""
Flask 應用主程式（整合對齊版 v3）
前後端 I/O 對齊 frontend.html：
- POST /analyze           ：接收原圖（FormData: image）→ 建 jobId、產初始遮罩（前端用 UI 遮罩：白=鎖、透明=可改）
- POST /masks/save        ：接收前端修正後的 UI 遮罩（PNG，白=鎖、透明=可改）→ 轉 L 遮罩（白=可改、黑=保護）存檔
- POST /render/batch      ：一次最多 3 種風格（JSON: jobId, styles[], palette{main,acc1,acc2,acc3}, logo{pos,scale,opacity}）→ 逐一生成
- POST /upload/mask       ：上傳局部遮罩（家具編修用）→ 回傳可用 URL
- POST /render/furniture-edit：針對單一遮罩做 replace / recolor（實作為同一張圖的局部 edit）
- GET  /styles            ：回傳可用風格清單
- GET  /jobs/<jobId>/<path:filename> ：提供 jobs 目錄靜態檔案（原圖/遮罩/輸出等）

保留舊接口（給其他客戶端）
- POST /generate          ：（舊）單次生成，參數 image_base64, style, colors, mask, mask_options
- GET  /status/<task_id>  ：（舊）查任務狀態

遮罩規則統一：
- 前端 UI 遮罩： 白=鎖定（不可改）、透明=可改
- OpenAI 遮罩（透明 PNG）： alpha=255（不透明）=保護、alpha=0（透明）=可編輯
- 內部 L 遮罩（單通道）：   L=0（黑）=保護、L=255（白）=可編輯
"""
import os, io, uuid, time, threading, base64, json
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import cv2

# ==== 相依匯入（優先 utils.*；若沒有 utils，再嘗試同層） ====
try:
    from utils.dalle_api import edit_image_with_mask, generate_image
    from utils.image_logo import add_logo
    from utils.prompt_templates import load_styles, build_style_prompt, make_prompt
except ModuleNotFoundError:
    from dalle_api import edit_image_with_mask, generate_image
    from image_logo import add_logo
    from prompt_templates import load_styles, build_style_prompt, make_prompt

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
CORS(app, resources={r"/*": {"origins": "*"}})

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

# ==== 遮罩轉換 ====
def build_mask_l(image_path: str, mode: str = "smart", opts: dict = None) -> np.ndarray:
    """
    產生 L 遮罩（白=可改，黑=保護）；回傳 numpy uint8 (H,W)。
    mode: 'full' → 全白；'smart' → 邊緣/窗戶/邊框保護
    """
    img = cv2.imread(image_path)
    if img is None: raise ValueError("failed to read image for masking")
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
    """
    L 遮罩（白=可改、黑=保護）→ 前端 UI 遮罩（白=鎖、透明=可改）
    黑(0)=保護 → UI 需白(255,255,255,255)
    白(255)=可改 → UI 需透明(alpha=0)
    """
    h, w = mask_l.shape[:2]
    # alpha = (白→0, 黑→255) = invert(L)
    alpha = 255 - mask_l
    rgba = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    rgba.putalpha(alpha)
    # 將可視顏色填為白（便於使用者看見「鎖定區域」）
    rgb = Image.new("RGB", (w, h), (255, 255, 255))
    rgba = Image.merge("RGBA", (*rgb.split(), rgba.split()[-1]))
    return rgba

def ui_png_to_l(ui_png_path: str) -> np.ndarray:
    """
    前端 UI 遮罩 PNG（白=鎖、透明=可改）→ L 遮罩（白=可改、黑=保護）
    以 alpha 為準： alpha>0 → 鎖（黑=0）；alpha=0 → 可改（白=255）
    """
    im = Image.open(ui_png_path).convert("RGBA")
    a = np.array(im.split()[-1])  # alpha
    l = np.where(a > 0, 0, 255).astype(np.uint8)
    return l

def l_to_alpha_png_path(mask_l: np.ndarray, ref_image_path: str, out_path: str) -> str:
    """
    L 遮罩（白=可改、黑=保護）→ OpenAI 透明遮罩（alpha=0 可編輯；alpha=255 保護）
    """
    with Image.open(ref_image_path) as ref:
        w, h = ref.size
    if mask_l.shape[:2] != (h, w):
        mask_l = cv2.resize(mask_l, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = 255 - mask_l  # 白→0、黑→255
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    Image.fromarray(alpha, mode="L")  # 方便除錯時觀察
    Image.fromarray(alpha, mode="L")
    Image.fromarray(alpha, mode="L")
    Image.fromarray(alpha, mode="L")
    rgba.save(out_path, "PNG")
    return out_path

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
    接收 FormData: image
    建立 jobId，儲存 original.jpg，產「初始 L 遮罩」與「UI 遮罩」，回傳：
    { jobId, masks: { editable_surface: "/jobs/<id>/masks/ui_mask.png" } }
    """
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "empty image"}), 400

    job = _new_job()
    orig_path = os.path.join(job.dir, "original.jpg")
    save_upload_image(f, orig_path)
    job.original = orig_path

    # 產生初始智慧遮罩（L）與 UI 遮罩
    l = build_mask_l(orig_path, mode="smart", opts={
        "edge_thresh": 28, "dilate_px": 6, "margin_ratio": 0.04,
        "protect_windows": True, "window_bright_thresh": 215
    })
    l_path = os.path.join(job.dir, "masks", "initial_L.png")
    Image.fromarray(l, mode="L").save(l_path, "PNG")

    ui_im = l_to_ui_rgba(l)
    ui_path = os.path.join(job.dir, "masks", "ui_mask.png")
    ui_im.save(ui_path, "PNG")

    job.mask_l_path = l_path
    job.ui_mask_path = ui_path
    JOBS[job.id] = job

    return jsonify({
        "jobId": job.id,
        "masks": {"editable_surface": f"/jobs/{job.id}/masks/ui_mask.png"}
    })

# ==== API: /masks/save ====
@app.post("/masks/save")
def save_mask():
    """
    接收前端修正後的 UI 遮罩（PNG，白=鎖、透明=可改）→ 轉 L（白=可改）覆蓋 job 的最終遮罩
    FormData: jobId, mask(file)
    """
    job_id = request.form.get("jobId", "").strip()
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid jobId"}), 400
    if "mask" not in request.files:
        return jsonify({"error": "mask file required"}), 400

    job = JOBS[job_id]
    ui_path = os.path.join(job.dir, "masks", "ui_mask.png")
    request.files["mask"].save(ui_path)
    job.ui_mask_path = ui_path

    l = ui_png_to_l(ui_path)
    l_path = os.path.join(job.dir, "masks", "final_L.png")
    Image.fromarray(l, mode="L").save(l_path, "PNG")
    job.mask_l_path = l_path

    return jsonify({"ok": True, "jobId": job_id})

# ==== API: /render/batch ====
@app.post("/render/batch")
def render_batch():
    """
    JSON: { jobId, styles: [..max3], palette:{main,acc1,acc2,acc3}, logo:{pos,scale,opacity} }
    回傳：{ images: [{style,url}], qc: {<style>:{keypoint_error:...}} }
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
        if not os.path.exists(os.path.join(job.dir, "masks", "initial_L.png")):
            return jsonify({"error": "mask not ready"}), 400
        job.mask_l_path = os.path.join(job.dir, "masks", "initial_L.png")

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
                add_logo(tmp_path, LOGO_PATH, out_png, logo_ratio=scale, position=pos, opacity=opacity)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                # 沒 LOGO 或失敗：直接使用原圖
                out_png = tmp_path
            results.append({"style": style, "url": f"/jobs/{job.id}/outputs/{os.path.basename(out_png)}"})
            qc[style] = {"keypoint_error": ""}  # 可擴充：關鍵點比對
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
    return jsonify({"maskUrl": f"/jobs/{job.id}/masks/user/{fname}"})

# ==== API: /render/furniture-edit ====
@app.post("/render/furniture-edit")
def furniture_edit():
    """
    JSON: { jobId, baseImageId, operations:[{type:'replace'|'recolor', target:'object', mask:<url>, spec|color:...}] }
    目前簡化：只吃第一個 operation 與 job.original 作為基底。
    """
    data = request.get_json(force=True, silent=False)
    job_id = (data.get("jobId") or "").strip()
    ops = data.get("operations") or []
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid jobId"}), 400
    if not ops:
        return jsonify({"error": "operations required"}), 400

    job = JOBS[job_id]
    op = ops[0]
    mask_url = op.get("mask") or ""
    if not mask_url.startswith("/jobs/"):
        return jsonify({"error": "mask url must be from /jobs"}), 400
    # 轉 URL → 路徑
    mask_path_ui = os.path.join(BASE_DIR, mask_url.lstrip("/"))
    # 轉為 L → 透明遮罩（OpenAI）
    l = ui_png_to_l(mask_path_ui)
    tmp_l = os.path.join(job.dir, "masks", f"edit_L_{uuid.uuid4().hex}.png")
    Image.fromarray(l, mode="L").save(tmp_l, "PNG")
    alpha_path = os.path.join(job.dir, "masks", f"edit_alpha_{uuid.uuid4().hex}.png")
    l_to_alpha_png_path(l, job.original, alpha_path)

    # 組 prompt
    if op.get("type") == "recolor":
        color = op.get("color", "#1E3A8A")
        prompt = f"Recolor the selected furniture/object to {color}. Keep the camera angle, structure, windows/doors and room layout unchanged."
    else:  # replace
        spec = op.get("spec", "modern sofa with metal legs")
        prompt = f"Replace the selected object with: {spec}. Keep all architecture and perspective exactly the same; only modify the masked area."

    out_png = os.path.join(job.dir, "outputs", f"edit_{uuid.uuid4().hex}.png")
    try:
        img_bytes = edit_image_with_mask(job.original, alpha_path, prompt, size="1024x1024")
        with open(out_png, "wb") as f:
            f.write(img_bytes)
        return jsonify({"imageUrl": f"/jobs/{job.id}/outputs/{os.path.basename(out_png)}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== API: /styles ====
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
        # 解析 colors（可為 "主色,配1,配2,配3" 或 HEX）
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
    # 支援 dataURL / 純 base64
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
        return jsonify({
            "status": task.status,
            "error": task.error,
            "original_image_url": task.original_image_url,
            "mask_url": task.mask_url,
            "styled_image_url": task.styled_image_url,
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

# -*- coding: utf-8 -*-
"""
Flask 應用主程式（最終版）
- /styles        ：回傳可用風格
- /generate      ：接收前端圖片與參數→排成背景任務
- /status/<id>   ：查任務狀態
- /healthz       ：健康檢查（Render）
- /uploads|/results|/masks ：靜態檔案（讓前端能直接載圖）

重點修正：
1) 將 L 模式黑白遮罩（白=可編輯、黑=保護）轉為 RGBA 透明遮罩（透明=可編輯），符合 OpenAI 規範
2) 完整接上 mask_options（邊緣閾值、膨脹、窗戶保護、亮度閾值、外框邊距）
3) 前端 maskMode="full" 與後端判斷對齊（也兼容 "all"）
4) 任務狀態更新加鎖，避免併發寫入競態
"""

import os
import io
import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ExifTags, ImageOps  # 透明遮罩會用到 ImageOps
import base64
import cv2
import numpy as np

# --- 匯入：優先 utils.*；若沒有 utils，再嘗試同層檔案（避免部署目錄差異）---
try:
    from utils.dalle_api import edit_image_with_mask
    from utils.image_logo import add_logo
    from utils.prompt_templates import load_styles, make_prompt
except ModuleNotFoundError:
    from dalle_api import edit_image_with_mask
    from image_logo import add_logo
    from prompt_templates import load_styles, make_prompt

# === 目錄設定 ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MASK_DIR = os.path.join(BASE_DIR, "masks")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo", "LOGO.png")  # 若沒有就不強制

for d in (UPLOAD_DIR, RESULT_DIR, MASK_DIR, STATIC_DIR, os.path.dirname(LOGO_PATH)):
    os.makedirs(d, exist_ok=True)

# === Flask App 與 CORS（正式環境建議改為白名單網域）===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# === 任務狀態 ===
@dataclass
class Task:
    status: str                                  # queued | pending | completed | failed
    error: Optional[str] = None
    original_image_url: Optional[str] = None
    mask_url: Optional[str] = None               # 建議對外回傳透明遮罩路徑（完成後會更新）
    styled_image_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)

TASKS: Dict[str, Task] = {}
TASKS_LOCK = threading.Lock()  # 多執行緒寫入保護

# === EXIF 方向修正（手機拍照常見旋轉問題）===
def correct_image_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img.getexif()
        if not exif:
            return img
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

# === Base64（或 dataURL）寫檔為 PNG ===
def save_base64_image(data_url: str, out_path: str) -> str:
    """
    支援 dataURL 或純 base64；將影像存為 PNG。
    *若需更嚴格的大小/格式限制，可在此處加驗證（例如 2MB、類型白名單）。
    """
    if data_url.startswith("data:"):
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = correct_image_orientation(img)
    img.save(out_path, "PNG")
    return out_path

# ---- 遮罩參數清洗（來自前端的 mask_options）----
def _sanitize_mask_opts(o: dict):
    """
    支援的鍵：
      - margin_ratio:      外框保護邊距占最短邊比例（0~0.2，預設 0.04）
      - edge_thresh:       邊緣偵測閾值（5~80，預設 28；Canny 高閾值=3x）
      - dilate_px:         邊緣保護膨脹像素（0~20，預設 6；讓保護區更寬）
      - protect_windows:   是否保護高亮窗戶區域（True/False，預設 True）
      - window_bright_thresh: 亮區分隔閾值（128~252，預設 215）
    """
    def clamp(v, lo, hi, typ=float):
        try:
            v = typ(v)
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    return {
        "margin_ratio": clamp(o.get("margin_ratio", 0.04), 0.0, 0.2, float),
        "edge_thresh": int(clamp(o.get("edge_thresh", 28), 5, 80, int)),
        "dilate_px": int(clamp(o.get("dilate_px", 6), 0, 20, int)),
        "protect_windows": bool(o.get("protect_windows", True)),
        "window_bright_thresh": int(clamp(o.get("window_bright_thresh", 215), 128, 252, int)),
    }

# === 遮罩建立（輸出 L 模式：白=可編輯、黑=保護）===
def build_mask(image_path: str, mode: str = "smart", opts: dict = None) -> str:
    """
    產生 L 模式遮罩（白=可編輯、黑=保護）。
    - mode in ("all","full"): 全白
    - mode="smart": 依參數做邊緣保護、窗戶保護、外框保護與膨脹
    生成後仍是 L（黑白）遮罩；送模型前會由 to_openai_alpha_mask 轉為 RGBA 透明遮罩。
    """
    opts = _sanitize_mask_opts(opts or {})
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("failed to read image for masking")
    h, w = img.shape[:2]

    # 前端用 'full'，也兼容 'all'
    if mode in ("all", "full"):
        mask = np.full((h, w), 255, dtype=np.uint8)  # 全白＝全可編輯
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) 邊緣保護（黑）：Canny 低/高閾值
        t1 = int(opts["edge_thresh"])
        t2 = max(t1 * 3, t1 + 1)
        edges = cv2.Canny(gray, t1, t2)

        # 2) 可編輯區（白）為底；把邊緣標為黑（保護）
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[edges > 0] = 0

        # 3) 膨脹保護：讓黑色保護區更寬一些（避免邊界被改到）
        k = max(1, int(opts["dilate_px"]))
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.erode(mask, kernel, iterations=1)  # 等價於對黑區做 dilate

        # 4) 高亮窗戶保護（選用）：以亮度閾值估計窗戶，標為黑
        if opts["protect_windows"]:
            _, bright = cv2.threshold(
                gray, opts["window_bright_thresh"], 255, cv2.THRESH_BINARY
            )
            # 清掉雜點，保留較連續的亮區
            win_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), max(3, k)))
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, win_kernel, iterations=1)
            mask[bright > 0] = 0

        # 5) 外框保護：四邊留 margin（黑）
        m = int(round(min(h, w) * float(opts["margin_ratio"])))
        if m > 0:
            mask[:m, :] = 0
            mask[-m:, :] = 0
            mask[:, :m] = 0
            mask[:, -m:] = 0

    out_path = os.path.join(MASK_DIR, f"{uuid.uuid4()}_L.png")
    Image.fromarray(mask, mode="L").save(out_path, "PNG")
    return out_path

# === 將 L 模式黑白遮罩 → 轉為 OpenAI 規範的 RGBA 透明遮罩（透明=可編輯）===
def to_openai_alpha_mask(mask_l_path: str, ref_image_path: str) -> str:
    """
    - 透明 (alpha=0) ：可被編輯
    - 不透明 (alpha=255)：保持原樣
    並確保遮罩尺寸與原圖一致（OpenAI 要求）。
    """
    m = Image.open(mask_l_path).convert("L")

    # 尺寸對齊到原圖
    with Image.open(ref_image_path) as ref_im:
        if m.size != ref_im.size:
            m = m.resize(ref_im.size, Image.NEAREST)

    # 白(255)=可編輯 → 要變透明(alpha=0)，所以把 L 取反當 alpha
    alpha = ImageOps.invert(m)  # 255→0、0→255
    rgba = Image.new("RGBA", m.size, (0, 0, 0, 0))  # 顏色無所謂，重點是 alpha
    rgba.putalpha(alpha)

    out_path = os.path.join(MASK_DIR, f"{uuid.uuid4()}_alpha.png")
    rgba.save(out_path, "PNG")
    return out_path

# === 背景任務：呼叫 OpenAI 編輯 +（可選）疊 LOGO ===
def _work_task(task_id: str, image_path: str, mask_path: str, style: str, colors: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task.status = "pending"

    try:
        # 組 prompt（使用 style + colors）
        prompt = make_prompt(style, colors)

        # 先把 L 遮罩轉為 RGBA 透明遮罩（透明=可編輯）
        mask_alpha = to_openai_alpha_mask(mask_path, image_path)

        # 呼叫 OpenAI（參考圖 + 透明遮罩 + prompt）
        out_tmp = os.path.join(RESULT_DIR, f"{uuid.uuid4()}.png")
        img_bytes = edit_image_with_mask(
            image_path=image_path,
            mask_path=mask_alpha,   # ← 使用透明遮罩
            prompt=prompt,
            size="1024x1024"
        )
        with open(out_tmp, "wb") as f:
            f.write(img_bytes)

        # 疊 LOGO（若失敗不影響主流程）
        final_path = os.path.join(RESULT_DIR, f"{uuid.uuid4()}.png")
        try:
            if os.path.exists(LOGO_PATH):
                add_logo(out_tmp, LOGO_PATH, final_path)
                styled_path = final_path
            else:
                styled_path = out_tmp
        except Exception:
            styled_path = out_tmp

        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if not task:
                return
            task.status = "completed"
            task.original_image_url = f"/uploads/{os.path.basename(image_path)}"
            task.styled_image_url = f"/results/{os.path.basename(styled_path)}"
            task.mask_url = f"/masks/{os.path.basename(mask_alpha)}"  # 對外回傳透明遮罩更直覺
    except Exception as e:
        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if not task:
                return
            task.status = "failed"
            task.error = str(e)

# === APIs ===
@app.get("/styles")
def styles_endpoint():
    """回傳可用風格（從 styles.txt 或預設）。"""
    return jsonify(load_styles())

@app.post("/generate")
def generate_endpoint():
    """
    接收 JSON：
    - image_base64：dataURL 或純 base64
    - style：風格字串
    - colors：主色系文字
    - mask：遮罩模式（例如 "smart" | "full" | "all"）
    - mask_options：遮罩細部參數（可選）
    """
    try:
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

        # 1) 儲存上傳圖
        img_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.png")
        save_base64_image(image_b64, img_path)

        # 2) 產生 L 遮罩（白=可編輯、黑=保護）。轉透明在背景任務進行
        mask_path = build_mask(img_path, mode=mask_mode, opts=mask_opts)

        # 3) 建立任務並丟到背景執行
        task_id = str(uuid.uuid4())
        with TASKS_LOCK:
            TASKS[task_id] = Task(
                status="queued",
                original_image_url=f"/uploads/{os.path.basename(img_path)}",
                mask_url=f"/masks/{os.path.basename(mask_path)}",  # 先回 L 遮罩路徑（背景會換成透明版）
            )

        threading.Thread(
            target=_work_task,
            args=(task_id, img_path, mask_path, style, colors),
            daemon=True
        ).start()

        return jsonify({"status": "queued", "task_id": task_id})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.get("/status/<task_id>")
def status_endpoint(task_id):
    """查詢任務狀態。"""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"status": "failed", "error": "task not found"}), 404
        return jsonify({
            "status": task.status,
            "error": task.error,
            "original_image_url": task.original_image_url,
            "mask_url": task.mask_url,               # 若已完成會是透明版遮罩
            "styled_image_url": task.styled_image_url,
            "created_at": task.created_at,
        })

# === 健康檢查（Render 會定期打這個）===
@app.get("/healthz")
def healthz():
    return ("", 204)

# === 靜態檔案服務（讓前端能直接載圖）===
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
    # 本機開發用；在 Render 會用 gunicorn 綁 $PORT 啟動
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

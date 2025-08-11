# -*- coding: utf-8 -*-
# Flask 後端主程式（修正版）：
# - 與前端一致的 API 介面：/styles、/generate、/status/<id>
# - /generate 接受 JSON（image_base64, style, colors, mask, mask_options）
# - 背景執行緒處理任務，前端用 /status 輪詢結果
# - 內建 3 種遮罩模式：full / safe_edges / smart
# - 自動糾正 EXIF 方向、支援靜態檔案回傳、加入 CORS
# - 產圖後嘗試疊 LOGO（若不存在不會中斷）

import os
import io
import uuid
import json
import threading
import base64
from dataclasses import dataclass
from typing import Dict

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ExifTags
import cv2

from dalle_api import edit_image_with_mask
from image_logo import add_logo
from prompt_templates import load_styles, make_prompt

app = Flask(__name__)
CORS(app)  # 允許跨網域請求，方便前端本機或不同網域呼叫

# 路徑設定
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
MASK_FOLDER = "masks"
STATIC_FOLDER = "static"
LOGO_PATH = os.path.join(STATIC_FOLDER, "logo", "LOGO.png")

# 確保資料夾存在
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, MASK_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)
os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)

# === 簡易任務管理（記憶體）===
@dataclass
class Task:
    status: str = "pending"   # pending | completed | failed
    original_image_url: str = ""
    styled_image_url: str = ""
    mask_url: str = ""
    error: str = ""

TASKS: Dict[str, Task] = {}

# === 工具：修正圖片 EXIF 旋轉 ===
def correct_image_orientation(image: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

# === 工具：儲存 base64 圖片到 PNG 檔 ===
def save_base64_image(data_url: str, out_path: str):
    """接受 dataURL 或純 base64 字串，轉存為 PNG。"""
    if data_url.startswith("data:"):
        b64 = data_url.split(",",1)[1]
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = correct_image_orientation(img)
    img.save(out_path, "PNG")
    return out_path

# === 遮罩生成：白=可編輯；黑=保護 ===
def build_mask(image_path: str, mode: str="smart", opts=None) -> str:
    """依據模式輸出 L 模式遮罩 PNG，供 OpenAI 編輯 API 使用。"""
    opts = opts or {}
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 可調參數（提供預設值）
    margin_ratio = float(opts.get("margin_ratio", 0.04))           # 邊框保護比例（避免邊緣扭曲）
    edge_thresh = int(opts.get("edge_thresh", 28))                  # 邊緣偵測阈值
    window_bright_thresh = int(opts.get("window_bright_thresh", 215))  # 亮窗/玻璃保護阈值
    dilate_px = int(opts.get("dilate_px", 6))                       # 稍微膨脹避免黑縫
    protect_windows = bool(opts.get("protect_windows", True))       # 是否保護高亮區域（常見窗戶）

    if mode == "full":
        # 全可編輯
        editable = np.full((h, w), 255, dtype=np.uint8)
    else:
        # 邊緣 + 框線 + 高亮區域保護
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_thresh, max(2*edge_thresh, edge_thresh+1))
        protect = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=max(1, dilate_px//2))

        if protect_windows:
            bright = cv2.inRange(gray, window_bright_thresh, 255)
            protect = cv2.bitwise_or(protect, bright)

        # 影像四周保護邊框
        m = int(max(1, round(min(h, w) * margin_ratio)))
        protect[:m,:] = 255; protect[-m:,:] = 255; protect[:,:m] = 255; protect[:,-m:] = 255

        # smart 模式：在 safe_edges 基礎上，再保護強垂直線，降低牆面歪斜機率
        if mode == "smart":
            vertical_sum = np.sum(edges, axis=0)
            vertical_cols = np.where(vertical_sum > 255 * 5)[0]  # 可調門檻
            vert_map = np.zeros_like(edges)
            vert_map[:, vertical_cols] = 255
            protect = cv2.bitwise_or(protect, vert_map)

        # 取反：白=可編輯；黑=保護
        editable = cv2.bitwise_not(protect)

    # 微膨脹可編輯區，避免邊界鋸齒或殘留
    if dilate_px > 0:
        editable = cv2.dilate(editable, np.ones((3,3), np.uint8), iterations=max(1, dilate_px//3))

    mask_path = os.path.join(MASK_FOLDER, f"{uuid.uuid4()}.png")
    cv2.imwrite(mask_path, editable)
    return mask_path

# === 背景任務：呼叫 OpenAI 編輯並疊 LOGO ===
def _work_task(task_id: str, image_path: str, mask_path: str, style: str, colors: str):
    task = TASKS[task_id]
    try:
        # 產生嚴格的風格轉換 prompt
        prompt = make_prompt(style, colors)

        # 呼叫 OpenAI 進行影像編輯
        out_tmp = os.path.join(RESULT_FOLDER, f"{uuid.uuid4()}.png")
        img_bytes = edit_image_with_mask(image_path=image_path, mask_path=mask_path, prompt=prompt, size="1024x1024")
        with open(out_tmp, "wb") as f:
            f.write(img_bytes)

        # 疊 LOGO（若失敗不會中斷）
        final_path = os.path.join(RESULT_FOLDER, f"{uuid.uuid4()}.png")
        try:
            add_logo(out_tmp, LOGO_PATH, final_path)
            styled_path = final_path
        except Exception:
            styled_path = out_tmp

        # 更新任務狀態
        task.status = "completed"
        task.original_image_url = f"/{UPLOAD_FOLDER}/{os.path.basename(image_path)}"
        task.styled_image_url = f"/{RESULT_FOLDER}/{os.path.basename(styled_path)}"
        task.mask_url = f"/{MASK_FOLDER}/{os.path.basename(mask_path)}"
    except Exception as e:
        task.status = "failed"
        task.error = str(e)

# === API: 讀取風格清單 ===
@app.route("/styles", methods=["GET"])
def styles_endpoint():
    return jsonify(load_styles())

# === API: 發佈生成任務 ===
@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    需求 JSON 格式：
    {
      "image_base64": "data:image/png;base64,... 或 純 base64",
      "style": "北歐風",
      "colors": "藍＋金",
      "mask": "smart" | "safe_edges" | "full",
      "mask_options": { ... 可選 參數 ... }
    }
    回傳：{"status":"queued","task_id":"..."}，前端用 /status/<id> 輪詢。
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        image_b64 = data.get("image_base64")
        style = data.get("style", "")
        colors = data.get("colors", "")
        mask_mode = data.get("mask", "smart")
        mask_opts = data.get("mask_options", {})

        if not image_b64:
            return jsonify({"status":"failed","error":"image_base64 is required"}), 400
        if not style or not colors:
            return jsonify({"status":"failed","error":"style and colors are required"}), 400

        # 儲存輸入圖片
        img_name = f"{uuid.uuid4()}.png"
        img_path = os.path.join(UPLOAD_FOLDER, img_name)
        save_base64_image(image_b64, img_path)

        # 產生遮罩
        mask_path = build_mask(img_path, mode=mask_mode, opts=mask_opts)

        # 建立任務並在背景執行
        task_id = str(uuid.uuid4())
        TASKS[task_id] = Task(status="pending")
        th = threading.Thread(target=_work_task, args=(task_id, img_path, mask_path, style, colors), daemon=True)
        th.start()

        return jsonify({"status":"queued","task_id":task_id})
    except Exception as e:
        return jsonify({"status":"failed","error":str(e)}), 500

# === API: 查詢任務狀態 ===
@app.route("/status/<task_id>", methods=["GET"])
def status_endpoint(task_id):
    t = TASKS.get(task_id)
    if not t:
        return jsonify({"status":"failed","error":"task not found"}), 404
    return jsonify({
        "status": t.status,
        "original_image_url": t.original_image_url,
        "styled_image_url": t.styled_image_url,
        "mask_url": t.mask_url,
        "error": t.error
    })

# === 靜態檔案服務（讓前端能直接載圖）===
@app.route(f"/{UPLOAD_FOLDER}/<path:filename>")
def get_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route(f"/{RESULT_FOLDER}/<path:filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route(f"/{MASK_FOLDER}/<path:filename>")
def get_mask(filename):
    return send_from_directory(MASK_FOLDER, filename)

if __name__ == "__main__":
    # 預設啟動於 0.0.0.0:5000，開發時 debug=True
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
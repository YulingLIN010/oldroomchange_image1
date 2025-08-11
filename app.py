
import os
import uuid
import base64
import traceback
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
from PIL import ImageFilter, ImageOps

from utils.prompt_templates import make_prompt, load_styles
from utils.dalle_api import edit_image_with_mask
from utils.image_logo import add_logo

# --- Smart mask helpers ---
def _dilate_binary_pil(arr, iterations=2, axis=None):
    """膨脹：用 MaxFilter 疊代。axis='x' 水平加粗；'y' 垂直加粗；None 等向。"""
    from PIL import Image as _PILImage
    img = _PILImage.fromarray(arr).convert("L")
    if axis == 'x':
        for _ in range(max(1, int(iterations))):
            img = img.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.SMOOTH)
    elif axis == 'y':
        for _ in range(max(1, int(iterations))):
            img = img.rotate(90, expand=True).filter(ImageFilter.MaxFilter(size=3)).rotate(-90, expand=True)
    else:
        for _ in range(max(1, int(iterations))):
            img = img.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(img)

def _build_smart_mask(base_img_path, out_mask_path,
                      margin_ratio=0.04,
                      bright_v_thresh=215,
                      low_sat_thresh=40,
                      edge_thresh=28,
                      vert_span_ratio=0.40,
                      top_band_ratio=0.20,
                      dilate_px_edges=6,
                      dilate_px_vert=8,
                      dilate_px_horz=8,
                      protect_windows=True):
    """結構感知遮罩：保護外框、門窗、樑柱、天花近水平強邊。白=可編輯；黑=保護。"""
    with Image.open(base_img_path).convert("RGB") as im_color:
        w, h = im_color.size
        margin = max(8, int(min(w, h) * float(margin_ratio)))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[margin:h-margin, margin:w-margin] = 255

        hsv = np.asarray(im_color.convert("HSV"))
        H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
        bright_bin = (V >= int(bright_v_thresh))
        low_sat_bin = (S <= int(low_sat_thresh))
        window_like = (bright_bin & low_sat_bin).astype(np.uint8) * 255
        window_like = np.asarray(Image.fromarray(window_like).filter(ImageFilter.MedianFilter(size=5)))
        window_like = _dilate_binary_pil(window_like, iterations=2)
        if protect_windows:
            mask = np.where(window_like > 0, 0, mask)

        gray = np.asarray(ImageOps.grayscale(im_color))
        gx = np.zeros_like(gray, dtype=np.int16); gy = np.zeros_like(gray, dtype=np.int16)
        gx[:,1:] = gray[:,1:] - gray[:,:-1]; gy[1:,:] = gray[1:,:] - gray[:-1,:]
        edge_mag = (np.abs(gx) + np.abs(gy)).astype(np.int16)
        edges = (edge_mag >= int(edge_thresh)).astype(np.uint8) * 255

        col_strength = (edges > 0).sum(axis=0) / float(h)
        vertical_cols = (col_strength >= float(vert_span_ratio))
        vert_map = np.zeros_like(edges); vert_map[:, vertical_cols] = 255
        vert_map = _dilate_binary_pil(vert_map, iterations=max(1, dilate_px_vert//2), axis='x')

        top_band_h = int(h * float(top_band_ratio))
        row_strength = (edges[:top_band_h,:] > 0).sum(axis=1) / float(w)
        strong_rows = (row_strength >= 0.10)
        horz_map = np.zeros_like(edges); horz_map[:top_band_h,:][strong_rows] = 255
        horz_map = _dilate_binary_pil(horz_map, iterations=max(1, dilate_px_horz//2), axis='y')

        edges_dil = _dilate_binary_pil(edges, iterations=max(1, dilate_px_edges//2))
        protect = np.maximum.reduce([edges_dil, vert_map, horz_map])
        mask = np.where(protect > 0, 0, mask)

        Image.fromarray(mask, mode="L").save(out_mask_path, "PNG")
    return out_mask_path

def _build_mask(base_img_path, out_mask_path, mode="safe_edges", **opts):
    """統一遮罩接口：safe_edges / full / smart"""
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw
    if mode == "smart":
        # 兼容前端命名（window_bright_thresh / dilate_px）
        bright_v = opts.get("bright_v_thresh", opts.get("window_bright_thresh", 215))
        dil_edges = opts.get("dilate_px_edges", opts.get("dilate_px", 6))
        dil_vert = opts.get("dilate_px_vert", dil_edges)
        dil_horz = opts.get("dilate_px_horz", dil_edges)
        return _build_smart_mask(base_img_path, out_mask_path,
                                 margin_ratio=opts.get("margin_ratio", 0.04),
                                 bright_v_thresh=bright_v,
                                 low_sat_thresh=opts.get("low_sat_thresh", 40),
                                 edge_thresh=opts.get("edge_thresh", 28),
                                 vert_span_ratio=opts.get("vert_span_ratio", 0.40),
                                 top_band_ratio=opts.get("top_band_ratio", 0.20),
                                 dilate_px_edges=dil_edges,
                                 dilate_px_vert=dil_vert,
                                 dilate_px_horz=dil_horz,
                                 protect_windows=opts.get("protect_windows", True))
    with Image.open(base_img_path) as im:
        w, h = im.size
        if mode == "full":
            mask = _PILImage.new("L", (w, h), color=255)  # 全白可編輯
        else:
            margin = max(12, int(min(w, h) * 0.04))
            mask = _PILImage.new("L", (w, h), color=0)
            draw = _ImageDraw.Draw(mask)
            draw.rectangle([margin, margin, w-margin, h-margin], fill=255)
        mask.save(out_mask_path, "PNG")
    return out_mask_path

# --- App setup ---
UPLOAD_DIR = 'static/uploads'
OUTPUT_DIR = 'static/output'
LOGO_PATH = 'static/logo/LOGO.PNG'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # 強化CORS設定

# 用 dict 模擬記憶體快取任務（正式建議用DB或Redis）
tasks = {}

@app.route("/")
def home():
    return render_template("frontend.html", styles=load_styles())

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}
        style = data.get("style")
        colors = data.get("colors", "")
        mask_mode = (data.get("mask") or "smart")
        mask_opts = data.get("mask_options") or {}

        # 讀入上傳圖片
        image_b64_full = data.get("image_base64")
        if not image_b64_full or "," not in image_b64_full:
            return jsonify({"status":"failed","error":"缺少 image_base64 或格式錯誤"}), 400
        image_b64 = image_b64_full.split(",",1)[1]
        img_bytes = base64.b64decode(image_b64)

        # 檔案大小限制
        if len(img_bytes) > 2*1024*1024:
            return jsonify({"status":"failed", "error":"圖片超過2MB"}), 413

        # 存上傳圖片
        filename = f"{uuid.uuid4()}.png"
        up_path = os.path.join(UPLOAD_DIR, filename)
        with open(up_path, "wb") as f:
            f.write(img_bytes)

        # 構建 prompt
        prompt = make_prompt(style, colors)

        # 產生遮罩
        mask_path = os.path.join(OUTPUT_DIR, f"{filename}_mask.png")
        _build_mask(up_path, mask_path, mode=mask_mode, **mask_opts)

        # 使用 image edits + mask
        try:
            img_bytes = edit_image_with_mask(up_path, mask_path, prompt, transparent=False)
        except Exception as e:
            print("[OPENAI API ERROR]", traceback.format_exc())
            return jsonify({"status":"failed", "error":f"OpenAI API failed: {e}"}), 500

        # 存結果
        gen_img_path = os.path.join(OUTPUT_DIR, f"{filename}_gen.png")
        with open(gen_img_path, 'wb') as f:
            f.write(img_bytes)

        # 驗證產生圖
        try:
            with Image.open(gen_img_path) as img:
                img.verify()
        except Exception as ex:
            print(f"[IMAGE VERIFY ERROR] {gen_img_path}", traceback.format_exc())
            return jsonify({"status":"failed", "error": f"產生的圖片內容錯誤或損毀 ({ex})"}), 500

        # 加 LOGO
        logo_img_path = os.path.join(OUTPUT_DIR, f"{filename}_logo.png")
        try:
            add_logo(gen_img_path, LOGO_PATH, logo_img_path)
        except Exception as ex:
            print(f"[ADD LOGO ERROR]", traceback.format_exc())
            return jsonify({"status":"failed", "error": f"加 LOGO 失敗：{ex}"}), 500

        # 回傳任務 id
        task_id = filename.split('.')[0]
        tasks[task_id] = {
            "original": f"/static/uploads/{filename}",
            "styled": f"/static/output/{filename}_logo.png",
            "mask": f"/static/output/{filename}_mask.png"
        }
        return jsonify({"task_id": task_id, "status": "processing"})
    except Exception as e:
        print("[GENERAL ERROR]", traceback.format_exc())
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route("/status/<task_id>")
def status(task_id):
    info = tasks.get(task_id)
    if not info:
        return jsonify({"status":"processing"})
    return jsonify({
        "status": "completed",
        "original_image_url": info["original"],
        "styled_image_url": info["styled"],
        "mask_url": info.get("mask")
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/styles')
def styles():
    return jsonify(load_styles())

@app.route('/healthz')
def healthz():
    return jsonify({'ok': True}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


# -*- coding: utf-8 -*-
"""
完整後端（對齊 index_new.html）
- /upload：上傳圖片（≤2MB）
- /detect/v2：501 讓前端 fallback
- /detect：簡易遮罩與綠色覆疊
- /mask/commit：提交鎖定遮罩（白=鎖定）→ 反推 editable → 更新覆疊
- /meta/styles：讀 styles_brief_table.json 或 styles.txt
- /generate：呼叫 dalle_api 依 style + palette 生成（或編輯）1~3 張
- /select：記錄選擇
- /compare：回原圖＋變體清單
- /furniture：指定區塊編修（新增/更換/改色）
"""
import os, io, uuid, json, time, re, base64
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps

# OpenCV / NumPy
import cv2
import numpy as np

try:
    import dalle_api
except Exception:
    dalle_api = None
try:
    import image_logo
except Exception:
    image_logo = None
try:
    import prompt_templates
except Exception:
    prompt_templates = None

APP = Flask(__name__)
CORS(APP, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}})

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def _ok(**kw):
    kw.setdefault("ok", True)
    return jsonify(kw)

def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code

def _img_size(img: Image.Image):
    return {"w": img.width, "h": img.height}

def _save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def _serve_path(image_id: str, rel: str):
    return f"/files/{image_id}/{rel}"

def _now_ver() -> int:
    return int(time.time())

def _ensure_alpha_mask(lock_gray_path: Path, out_path: Path) -> Path:
    # 白=鎖定 -> alpha=255 保留；黑=可改 -> alpha=0 可編輯
    m = Image.open(lock_gray_path).convert("L")
    rgba = Image.new("RGBA", m.size, (0, 0, 0, 0))
    a = m.point(lambda v: 255 if v > 127 else 0)
    rgba.putalpha(a)
    rgba.save(out_path)
    return out_path

def _data_url_to_png(data_url: str, out_path: Path) -> Path:
    if not data_url or "," not in data_url:
        raise ValueError("invalid data url")
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path

def _meta_path(image_id: str) -> Path:
    return DATA / image_id / "meta.json"

def _load_meta(image_id: str) -> Dict[str, Any]:
    p = _meta_path(image_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_meta(image_id: str, meta: Dict[str, Any]):
    p = _meta_path(image_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- basic routes ----------------
@APP.get("/healthz")
def healthz():
    return ("", 204)

@APP.get("/files/<path:rel>")
def files(rel: str):
    p = (DATA / rel).resolve()
    if not str(p).startswith(str(DATA.resolve())):
        return _bad("invalid path", 404)
    if not p.exists():
        return _bad("not found", 404)
    return send_file(str(p))

# ---------------- 1) upload ----------------
@APP.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return _bad("missing file")
    f.seek(0, io.SEEK_END); size = f.tell(); f.seek(0)
    if size > 2*1024*1024:
        return _bad("file too large (>2MB)", 413)
    image_id = uuid.uuid4().hex[:10]
    img = Image.open(f.stream).convert("RGB")
    out_dir = DATA / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(img, out_dir / "original.png")
    _save_meta(image_id, {"created_at": int(time.time())})
    return _ok(image_id=image_id, **_img_size(img))

# ---------------- 2) detect ----------------

def _opencv_detect_lock(image_path: Path) -> Image.Image:
    """回傳 PIL Image（L）：白=鎖定"""
    img = cv2.imread(str(image_path))  # BGR
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    # 長邊限制 1600，提升速度
    scale = 1600 / max(H, W) if max(H, W) > 1600 else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    enh = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 自動 Canny
    v = np.median(gray_blur)
    lower = int(max(0, 0.67 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_blur, lower, upper)

    # Hough 線段
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(min(H,W)*0.10), maxLineGap=12)
    line_mask = np.zeros((H,W), np.uint8)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dx == 0 or dy == 0 or min(dx,dy)/max(dx,dy) < 0.2:
                cv2.line(line_mask, (x1,y1), (x2,y2), 255, thickness=3)
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)), iterations=1)

    # 幾何區塊候選
    thr = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,5)
    filled = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)), iterations=2)
    geom = cv2.bitwise_or(filled, line_mask)

    contours, _ = cv2.findContours(geom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lock = np.zeros((H,W), np.uint8)
    area_img = H*W

    def touches_border(box):
        x,y,w,h = box
        return (x <= 2) or (y <= 2) or (x+w >= W-3) or (y+h >= H-3)

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_img*0.015:
            continue
        x,y,w,h = cv2.boundingRect(c)
        rect_area = w*h
        rectangularity = area / max(1,rect_area)
        ratio = w / max(1.0, h)

        big_touch = (area > area_img*0.05) and touches_border((x,y,w,h))
        near_rect = (rectangularity > 0.65 and 0.2 < ratio < 5.0)
        if big_touch or near_rect:
            cv2.drawContours(lock, [c], -1, 255, thickness=-1)

    lock = cv2.morphologyEx(lock, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)), iterations=1)

    pil = Image.fromarray(lock, mode="L")
    if scale < 1.0:
        pil = pil.resize((int(W/scale), int(H/scale)), resample=Image.NEAREST)
    return pil



@APP.post("/detect")
def detect():
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    if not image_id:
        return _bad("missing image_id")
    base = DATA / image_id / "original.png"
    if not base.exists():
        return _bad("image not found", 404)
    try:
        lock_img = _opencv_detect_lock(base)
    except Exception as e:
        return _bad(f"opencv detect failed: {e}", 500)

    im = Image.open(base).convert("RGBA")
    w, h = im.size
    ver = _now_ver()
    out_dir = DATA / image_id
    lock_path = out_dir / f"lock_v{ver}.png"
    _save(lock_img, lock_path)

    # merged overlay（綠覆層）
    green = Image.new("RGBA", (w, h), (30, 190, 107, int(0.85*255)))
    green.putalpha(lock_img)
    merged = Image.alpha_composite(im, green)
    merged_path = out_dir / f"merged_v{ver}.png"
    _save(merged, merged_path)

    return _ok(detector="opencv",
               mask_version=ver,
               lock_mask=_serve_path(image_id, lock_path.name),
               merged_overlay=_serve_path(image_id, merged_path.name))

# ---------------- 5) generate ----------------
# ---------------- meta styles ----------------
@APP.get("/meta/styles")
def meta_styles():
    # If you have a file to drive styles, add reading here (e.g., styles.json)
    default = [
        {"code":"北歐風","name":"北歐風","brief":"簡約線條、自然採光、功能導向"},
        {"code":"現代風","name":"現代風","brief":"幾何分明、極簡收納、玻璃金屬感"},
        {"code":"地中海風","name":"地中海風","brief":"拱門、白牆、藍白對比"}
    ]
    return _ok(styles=default)

@APP.post("/generate")
def generate():
    if dalle_api is None or prompt_templates is None:
        return _bad("image backend not configured", 503)
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    styles = (data.get("styles") or [])[:3]
    mask_version = str(data.get("mask_version") or "")
    palette = data.get("palette") or {}
    want_logo = bool(data.get("logo", True))
    if not image_id or not styles:
        return _bad("missing image_id or styles")

    base_path = DATA / image_id / "original.png"
    if not base_path.exists():
        return _bad("image not found", 404)

    out_dir = DATA / image_id
    # 找符合版本的 lock 遮罩，否則取最新
    if mask_version and (out_dir / f"lock_v{mask_version}.png").exists():
        lock_gray = out_dir / f"lock_v{mask_version}.png"
    else:
        locks = sorted(out_dir.glob("lock_v*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        lock_gray = locks[0] if locks else None

    mask_alpha_path = None
    if lock_gray:
        mask_alpha_path = out_dir / "lock_alpha.png"
        _ensure_alpha_mask(lock_gray, mask_alpha_path)

    # palette → colors for prompt template
    colors = {"main": palette.get("main")}
    accs = palette.get("accents") or []
    for i, a in enumerate(accs[:3], 1):
        colors[f"acc{i}"] = a

    size = os.getenv("IMAGE_SIZE", "1024x1024")
    variants = []
    meta = _load_meta(image_id)
    meta.setdefault("results", {})

    for idx, st in enumerate(styles, 1):
        prompt = prompt_templates.make_prompt(st, colors)
        try:
            if mask_alpha_path and mask_alpha_path.exists():
                png_bytes = dalle_api.edit_image_with_mask(str(base_path), str(mask_alpha_path), prompt, size=size, transparent=False)
            else:
                png_bytes = dalle_api.edit_image_no_mask(str(base_path), prompt, size=size, transparent=False)
        except Exception as e:
            return _bad(f"generate failed: {e}", 500)

        ver = _now_ver()
        fname = f"gen_v{ver}_{idx}.png"
        fpath = out_dir / fname
        with open(fpath, "wb") as f:
            f.write(png_bytes)

        url_rel = fname
        if want_logo and image_logo is not None:
            logo_path = os.getenv("LOGO_PATH", str(ROOT / "static/logo/LOGO.png"))
            try:
                out_with_logo = image_logo.add_logo(str(fpath), logo_path=logo_path, margin_ratio=0.02, logo_ratio=0.15)
                url_rel = Path(out_with_logo).name
            except Exception:
                url_rel = fname

        rid = f"v{ver}_{idx}"
        meta["results"][rid] = {"file": url_rel, "style": st, "ts": ver}
        variants.append({"result_id": rid, "style": st, "url": _serve_path(image_id, url_rel), "download_url": _serve_path(image_id, url_rel)})

    _save_meta(image_id, meta)
    return _ok(variants=variants)

# ---------------- 6) select ----------------
@APP.post("/select")
def select():
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    result_id = data.get("result_id")
    if not image_id or not result_id:
        return _bad("missing image_id or result_id")
    meta = _load_meta(image_id)
    if result_id not in meta.get("results", {}):
        return _bad("result not found", 404)
    meta["selected"] = result_id
    _save_meta(image_id, meta)
    return _ok(selected=result_id)

# ---------------- 7) compare ----------------
@APP.get("/compare")
def compare():
    image_id = request.args.get("base")
    ids = [s for s in (request.args.get("vars","") or "").split(",") if s]
    if not image_id:
        return _bad("missing base")
    items = []
    base = DATA / image_id / "original.png"
    if base.exists():
        url = _serve_path(image_id, "original.png")
        items.append({"url": url, "download_url": url})
    meta = _load_meta(image_id)
    for rid in ids:
        info = meta.get("results", {}).get(rid)
        if not info: 
            continue
        url = _serve_path(image_id, info["file"])
        items.append({"url": url, "download_url": url})
    return _ok(items=items)

# ---------------- 8) furniture ----------------

@APP.post("/furniture")
def furniture():
    if dalle_api is None or prompt_templates is None:
        return _bad("image backend not configured", 503)
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    result_id = data.get("result_id")
    action = (data.get("action") or "add").lower()
    obj = data.get("object") or ""
    color = data.get("color") or ""
    name = data.get("name") or ""
    location = data.get("location") or ""
    style_hint = data.get("style_hint") or ""
    data_url = data.get("mask_data_url")
    if not image_id or not result_id:
        return _bad("missing image_id or result_id")

    meta = _load_meta(image_id)
    info = meta.get("results", {}).get(result_id)
    if not info:
        return _bad("result not found", 404)

    base_path = DATA / image_id / info["file"]
    out_dir = DATA / image_id
    mask_path = out_dir / "tmp_furn_mask.png"

    if data_url:
        # 統一最終語意：白=鎖定（alpha=255）、黑=可編輯（alpha=0）
        head, b64 = data_url.split(",", 1)
        import base64 as _b64
        raw = _b64.b64decode(b64)
        with open(mask_path, "wb") as f:
            f.write(raw)
        from PIL import Image as _I
        m = _I.open(mask_path).convert("L")
        # 自動相容：若白像素比例 < 50%（推測舊版白=選區），則反向處理
        hist = m.histogram()
        white_cnt = sum(hist[128:])
        total = m.size[0] * m.size[1]
        white_ratio = white_cnt / max(1,total)
        rgba = _I.new("RGBA", m.size, (0,0,0,0))
        if white_ratio < 0.5:
            # 舊語意（白=選區）→ 白區可編輯(alpha=0)，其他鎖定(alpha=255)
            a = m.point(lambda v: 0 if v>127 else 255)
        else:
            # 新語意（白=鎖定）→ 白區鎖定(alpha=255)，黑區可編輯(alpha=0)
            a = m.point(lambda v: 255 if v>127 else 0)
        rgba.putalpha(a)
        rgba.save(mask_path)
    else:
        # 未提供選區 → 預設全白（全鎖定，避免誤改）
        from PIL import Image as _I
        with _I.open(base_path) as _im:
            w,h = _im.size
        rgba = _I.new("RGBA", (w,h), (0,0,0,0))
        a = _I.new("L", (w,h), 255)  # 255=保留（鎖定）
        rgba.putalpha(a)
        rgba.save(mask_path)

    verb = {"add":"add furniture", "swap":"replace furniture", "recolor":"recolor objects", "remove":"remove furniture and reconstruct background"}.get(action,"edit")
    extra = f"{verb}: {name or obj} {color}".strip()
    if location:
        extra += f"; placement: {location}"
    if style_hint:
        extra += f"; furniture style: {style_hint}"
    base_style = info.get("style") or "modern"
    prompt = prompt_templates.make_prompt(base_style, None)
    if extra:
        prompt += "\\n" + extra

    try:
        png_bytes = dalle_api.edit_image_with_mask(str(base_path), str(mask_path), prompt, size=os.getenv("IMAGE_SIZE","1024x1024"), transparent=False)
    except Exception as e:
        return _bad(f"furniture failed: {e}", 500)

    ver = _now_ver()
    fname = f"furn_v{ver}.png"
    fpath = out_dir / fname
    with open(fpath, "wb") as f:
        f.write(png_bytes)

    url_rel = fname
    if image_logo is not None:
        logo_path = os.getenv("LOGO_PATH", str(ROOT / "static/logo/LOGO.png"))
        try:
            out_with_logo = image_logo.add_logo(str(fpath), logo_path=logo_path, margin_ratio=0.02, logo_ratio=0.15)
            url_rel = Path(out_with_logo).name
        except Exception:
            url_rel = fname

    return _ok(url=_serve_path(image_id, url_rel))
# ---------------- mask commit ----------------
@APP.post("/mask/commit")
def mask_commit():
    data = request.form or {}
    image_id = (data.get("image_id") or "").strip()
    f = request.files.get("lock_mask")
    if not image_id or not f:
        return _bad("missing image_id or lock_mask")
    out_dir = DATA / image_id
    base = out_dir / "original.png"
    if not base.exists():
        return _bad("image not found", 404)

    # Save grayscale (white=lock)
    ver = _now_ver()
    lock_path = out_dir / f"lock_v{ver}.png"
    img = Image.open(f.stream).convert("L")
    _save(img, lock_path)

    # Build green overlay preview
    im = Image.open(base).convert("RGBA")
    w, h = im.size
    green = Image.new("RGBA", (w, h), (30, 190, 107, int(0.85*255)))
    green.putalpha(img)
    merged = Image.alpha_composite(im, green)
    merged_path = out_dir / f"merged_v{ver}.png"
    _save(merged, merged_path)

    return _ok(mask_version=ver,
               lock_mask=_serve_path(image_id, lock_path.name),
               merged_overlay=_serve_path(image_id, merged_path.name))


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")), debug=True)

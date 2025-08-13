
# -*- coding: utf-8 -*-
import os, io, uuid, json, base64, time, math, re, functools, threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file, make_response, g
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageStat

# Optional libs
try:
    import cv2, numpy as np
except Exception:
    cv2, np = None, None

# local modules
import prompt_templates as PT
from dalle_api import edit_image_with_mask
from image_logo import add_logo

APP = Flask(__name__)
CORS(APP, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS","*").split(",")}})

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR = ROOT / "static"

# --- Security / JWT ---
REQUIRE_JWT = os.getenv("REQUIRE_JWT", "0") in ("1","true","True")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-please-change")

# --- Model governance / AB ---
MODEL_A = os.getenv("IMAGE_MODEL_NAME", "gpt-image-1")
MODEL_B = os.getenv("IMAGE_MODEL_B", MODEL_A)
AB_WEIGHT_B = float(os.getenv("AB_WEIGHT_B","0.0"))  # 0.0 ~ 1.0

# --- Quota / Rate limiting (simple in-memory token bucket) ---
RATE_LIMIT_QPS = float(os.getenv("RATE_LIMIT_QPS","3"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST","6"))
_rate_state = {}
_lock = threading.Lock()

def rate_limiter(key: str):
    now = time.time()
    with _lock:
        b = _rate_state.get(key)
        if not b:
            _rate_state[key] = {"tokens": RATE_LIMIT_BURST, "ts": now}
            return True, 0
        tokens = b["tokens"]
        elapsed = now - b["ts"]
        tokens = min(RATE_LIMIT_BURST, tokens + elapsed * RATE_LIMIT_QPS)
        if tokens < 1:
            retry = math.ceil((1 - tokens) / RATE_LIMIT_QPS)
            b["tokens"] = tokens
            b["ts"] = now
            return False, retry
        b["tokens"] = tokens - 1
        b["ts"] = now
        return True, 0

def require_quota(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        ip = request.headers.get("x-forwarded-for", request.remote_addr or "anon").split(",")[0]
        ok, retry = rate_limiter(ip + ":" + request.path)
        if not ok:
            resp = jsonify({"ok": False, "error": "Too Many Requests", "retry_after": retry})
            return (resp, 429, {"Retry-After": str(retry)})
        return fn(*a, **kw)
    return wrapper

# --- Observability: simple metrics counters/timers ---
_metrics = {
    "requests_total": {},
    "request_seconds_sum": {},
    "request_seconds_count": {},
}
def _metric_key(path, method, code):
    return f'{method.lower()}_{path}_{code}'
def observe(path, method, code, seconds):
    k = _metric_key(path, method, code)
    _metrics["requests_total"][k] = _metrics["requests_total"].get(k, 0) + 1
    _metrics["request_seconds_sum"][k] = _metrics["request_seconds_sum"].get(k, 0.0) + seconds
    _metrics["request_seconds_count"][k] = _metrics["request_seconds_count"].get(k, 0) + 1

@APP.after_request
def add_csp_headers(resp):
    # basic CSP and cache headers for static images
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
    return resp

def _measure(fn):
    @functools.wraps(fn)
    def inner(*a, **kw):
        t0 = time.time()
        try:
            resp = fn(*a, **kw)
            code = 200
            if isinstance(resp, tuple) and len(resp) > 1:
                code = resp[1]
            elif hasattr(resp, "status_code"):
                code = resp.status_code
            return resp
        finally:
            dt = time.time() - t0
            observe(request.path, request.method, code, dt)
    return inner

# --- helpers ---
def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code

def _ok(**kwargs):
    d = {"ok": True}
    d.update(kwargs)
    return jsonify(d)

def _image_dir(image_id: str) -> Path:
    d = DATA_DIR / image_id
    d.mkdir(exist_ok=True, parents=True)
    (d/"masks").mkdir(exist_ok=True)
    (d/"results").mkdir(exist_ok=True)
    return d

def _serve_path(image_id: str, sub: str) -> str:
    return f"/files/{image_id}/{sub}"

def _ensure_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")

def _normalize_mask_L(img: Image.Image, size)->Image.Image:
    m = img.convert("L").resize(size, Image.NEAREST)
    return m

def _make_openai_alpha_mask_from_white_editable(L_mask: Image.Image) -> Image.Image:
    inv = ImageOps.invert(L_mask)  # white(255)->0 transparent (editable)
    rgba = Image.new("RGBA", L_mask.size, (0,0,0,0))
    rgba.putalpha(inv)
    return rgba

def _auth_required():
    if not REQUIRE_JWT:
        return True
    auth = request.headers.get("Authorization","")
    m = re.match(r"^Bearer\s+(.+)$", auth)
    if not m:
        return False
    token = m.group(1)
    try:
        import jwt
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False

# ---------- routing ----------
@APP.get("/healthz")
def healthz():
    return ("", 204)

@APP.get("/metrics")
def metrics():
    # simple Prometheus-like exposition
    lines = []
    for k,v in _metrics["requests_total"].items():
        lines.append(f'requests_total{{route="{k}"}} {v}')
    for k,v in _metrics["request_seconds_sum"].items():
        lines.append(f'request_seconds_sum{{route="{k}"}} {v}')
    for k,v in _metrics["request_seconds_count"].items():
        lines.append(f'request_seconds_count{{route="{k}"}} {v}')
    return ("\n".join(lines)+"\n", 200, {"Content-Type":"text/plain; version=0.0.4"})

@APP.get("/meta/styles")
@_measure
def meta_styles():
    try:
        styles = PT.load_styles()
        names = [{"code": s["name"], "name": s["name"]} for s in styles]
    except Exception:
        names = [{"code":"現代風","name":"現代風"}]
    kb_v = int(os.path.getmtime((ROOT/"styles_brief_table.json"))) if (ROOT/"styles_brief_table.json").exists() else int(time.time())
    return _ok(styles=names, kb_version=kb_v)

@APP.post("/upload")
@require_quota
@_measure
def upload():
    f = request.files.get("file")
    if not f: return _bad("缺少欄位 file")
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".png",".jpg",".jpeg"): return _bad("僅接受 .png .jpg .jpeg")
    f.seek(0, os.SEEK_END); size = f.tell(); f.seek(0)
    if size > 2*1024*1024: return _bad("檔案超過 2MB", 413)

    image_id = uuid.uuid4().hex
    d = _image_dir(image_id)
    Image.open(f.stream).convert("RGBA").save(d/"original.png", "PNG")

    im = Image.open(d/"original.png")
    w,h = im.size
    meta = {"image_id": image_id, "w": w, "h": h, "created_at": time.time(), "selected_result": None}
    (d/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return _ok(image_id=image_id, w=w, h=h)

@APP.get("/files/<image_id>/<path:subpath>")
def files(image_id, subpath):
    base = _image_dir(image_id).resolve()
    p = (base / subpath).resolve()
    if not str(p).startswith(str(base)) or not p.exists():
        return _bad("找不到檔案", 404)
    # cache headers already set
    return send_file(p)

@APP.delete("/image/<image_id>")
@_measure
def delete_image(image_id):
    d = _image_dir(image_id)
    try:
        for p in d.rglob("*"):
            if p.is_file(): p.unlink()
        d.rmdir()
    except Exception:
        pass
    return _ok(deleted=True)

@APP.post("/detect")
@require_quota
@_measure
def detect():
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
    data = request.get_json(silent=True) or {}
    img_id = data.get("image_id")
    if not img_id: return _bad("缺少 image_id")
    d = _image_dir(img_id)
    raw = d/"original.png"
    if not raw.exists(): return _bad("原始圖不存在",404)
    im = Image.open(raw).convert("RGBA")
    w,h = im.size

    # version
    vfile = d/"masks"/"version.txt"
    ver = int(vfile.read_text())+1 if vfile.exists() else 1
    vfile.write_text(str(ver))

    # Attempt CV detection if cv2 available
    if cv2 is not None:
        img_bgr = cv2.imdecode(np.frombuffer((raw).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        # dilate to thicken edges (lock them)
        kernel = np.ones((3,3), np.uint8)
        lock_mask = cv2.dilate(edges, kernel, iterations=1)
        lock_img = Image.fromarray(lock_mask).convert("L")
        # editable = inverse (areas not locked)
        editable = ImageOps.invert(lock_img).filter(ImageFilter.GaussianBlur(radius=1)).point(lambda p: 255 if p>32 else 0)
        # rudimentary depth: Laplacian variance -> approximate distance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        depth = cv2.normalize(cv2.GaussianBlur(np.abs(lap), (0,0), 3), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        depth_img = Image.fromarray(depth).convert("L")
    else:
        lock_img = Image.new("L",(w,h),0)
        editable = Image.new("L",(w,h),255)
        # basic gradient depth
        depth_img = Image.linear_gradient("L").resize((w,h))

    # save masks
    lock_path = d/"masks"/f"lock_v{ver}.png"
    editable_path = d/"masks"/f"editable_v{ver}.png"
    depth_path = d/"masks"/f"depth_v{ver}.png"
    lock_img.save(lock_path, "PNG")
    editable.save(editable_path, "PNG")
    depth_img.save(depth_path, "PNG")

    # merged overlay
    base = im.copy()
    green = Image.new("RGBA",(w,h),(80,220,120,0)); green.putalpha(editable)
    red   = Image.new("RGBA",(w,h),(239,68,68,0)); red.putalpha(lock_img)
    merged = Image.alpha_composite(base, green)
    merged = Image.alpha_composite(merged, red)
    merged_path = d/"masks"/f"merged_v{ver}.png"
    merged.save(merged_path,"PNG")

    return _ok(mask_version=ver,
               lock_mask=_serve_path(img_id, f"masks/lock_v{ver}.png"),
               editable_mask=_serve_path(img_id, f"masks/editable_v{ver}.png"),
               merged_overlay=_serve_path(img_id, f"masks/merged_v{ver}.png"),
               depth_map=_serve_path(img_id, f"masks/depth_v{ver}.png"))

@APP.post("/mask/commit")
@require_quota
@_measure
def mask_commit():
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
    img_id = request.form.get("image_id")
    if not img_id: return _bad("缺少 image_id")
    d = _image_dir(img_id)
    raw = d/"original.png"
    if not raw.exists(): return _bad("原始圖不存在",404)
    w,h = Image.open(raw).size

    vfile = d/"masks"/"version.txt"
    ver = int(vfile.read_text())+1 if vfile.exists() else 1
    vfile.write_text(str(ver))

    lock_f = request.files.get("lock_mask")
    editable_f = request.files.get("editable_mask")
    if not editable_f: return _bad("缺少 editable_mask")

    editable = _normalize_mask_L(Image.open(editable_f.stream), (w,h))
    if lock_f:
        lock_img = _normalize_mask_L(Image.open(lock_f.stream), (w,h))
    else:
        # derive lock from inverse editable for safety
        lock_img = ImageOps.invert(editable)

    lock_path = d/"masks"/f"lock_v{ver}.png"
    editable_path = d/"masks"/f"editable_v{ver}.png"
    lock_img.save(lock_path,"PNG"); editable.save(editable_path,"PNG")

    base = Image.open(raw).convert("RGBA")
    green = Image.new("RGBA",(w,h),(80,220,120,0)); green.putalpha(editable.point(lambda p:int(p*0.35)))
    red   = Image.new("RGBA",(w,h),(239,68,68,0)); red.putalpha(lock_img.point(lambda p:int(p*0.45)))
    merged = Image.alpha_composite(base, green); merged = Image.alpha_composite(merged, red)
    merged.save(d/"masks"/f"merged_v{ver}.png","PNG")

    return _ok(mask_version=ver)

def _choose_model():
    if AB_WEIGHT_B <= 0.0 or MODEL_A == MODEL_B:
        return MODEL_A
    # hash per request/image to stable assign
    key = request.json.get("image_id","") if request.is_json else str(time.time())
    h = (sum(ord(c) for c in key) % 100) / 100.0
    return MODEL_B if h < AB_WEIGHT_B else MODEL_A

@APP.post("/generate")

@APP.get("/compare")
@_measure
def compare_grid():
    """
    回傳四格比對牆所需的項目清單（原圖 + 指定變體們），皆為可直接顯示/下載的 URL。
    Query:
      - base:  image_id
      - vars:  以逗號分隔的 result_id 清單（可 0~3 個）
      - logo:  1/0，若為 1 則原圖會動態疊上 LOGO 輸出於 compare/ 資料夾
    回傳：{ok, items:[{url, download_url}]}
    """
    img_id = request.args.get("base")
    var_str = request.args.get("vars") or ""
    want_logo = request.args.get("logo","1") in ("1","true","True")
    if not img_id:
        return _bad("缺少 base(image_id)")
    d = _image_dir(img_id)
    if not (d/"original.png").exists():
        return _bad("找不到原圖", 404)

    items = []
    # 原圖處理（是否加 LOGO）
    if want_logo:
        (d/"compare").mkdir(exist_ok=True)
        base_logo = d/"compare"/"original_logo.png"
        if not base_logo.exists():
            try:
                add_logo(str(d/"original.png"), str(STATIC_DIR/"logo/LOGO.png"), str(base_logo))
            except Exception:
                # 若 LOGO 檔缺失，退回原圖
                base_logo = d/"original.png"
        base_url = f"/files/{img_id}/compare/{base_logo.name}" if base_logo.name != "original.png" else f"/files/{img_id}/original.png"
    else:
        base_url = f"/files/{img_id}/original.png"
    items.append({"url": base_url, "download_url": base_url})

    # 變體處理
    rid_list = [x.strip() for x in var_str.split(",") if x.strip()]
    rp = d/"results"
    if rp.exists():
        for rid in rid_list:
            # 找最符合 rid_*.png
            found = None
            for p in rp.glob(f"{rid}_*.png"):
                found = p; break
            if found:
                url = f"/files/{img_id}/results/{found.name}"
                items.append({"url": url, "download_url": url})
    return _ok(items=items)

@require_quota
@_measure
def generate():
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
    data = request.get_json(silent=True) or {}
    img_id = data.get("image_id")
    styles = data.get("styles") or []
    palette = data.get("palette") or {}
    mask_ver = data.get("mask_version")
    do_logo = bool(data.get("logo", True))
    if not img_id: return _bad("缺少 image_id")
    if not styles: return _bad("至少 1 種風格")
    if len(styles) > 3: styles = styles[:3]

    d = _image_dir(img_id)
    raw = d/"original.png"
    if not raw.exists(): return _bad("原始圖不存在",404)
    w,h = Image.open(raw).size

    mv_file = d/"masks"/"version.txt"
    if not mask_ver and mv_file.exists():
        mask_ver = int(mv_file.read_text())
    if not mask_ver:
        return _bad("尚未建立遮罩")

    editable_path = d/"masks"/f"editable_v{mask_ver}.png"
    if not editable_path.exists(): return _bad("找不到遮罩",404)

    openai_mask = _make_openai_alpha_mask_from_white_editable(Image.open(editable_path).convert("L"))
    mask_buf = io.BytesIO(); openai_mask.save(mask_buf, "PNG"); mask_buf.seek(0)

    model = _choose_model()
    variants = []
    for s in styles:
        prompt = PT.make_prompt(s, {
            "main": palette.get("main"),
            "acc1": (palette.get("accents") or [None,None,None])[0],
            "acc2": (palette.get("accents") or [None,None,None])[1] if len(palette.get("accents",[]))>1 else None,
            "acc3": (palette.get("accents") or [None,None,None])[2] if len(palette.get("accents",[]))>2 else None,
        })
        out_bytes = edit_image_with_mask(str(raw), mask_buf, prompt, size="1024x1024", model=model)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rid = uuid.uuid4().hex[:10]
        out_path = d/"results"/f"{rid}_{s}_{ts}.png"
        with open(out_path, "wb") as f: f.write(out_bytes)
        if do_logo:
            add_logo(str(out_path), str(STATIC_DIR/"logo/LOGO.png"), str(out_path))
        variants.append({"result_id": rid, "style": s,
                         "url": _serve_path(img_id, f"results/{out_path.name}"),
                         "download_url": _serve_path(img_id, f"results/{out_path.name}")})

    meta_p = d/"meta.json"
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    m["last_mask_version"] = int(mask_ver)
    m["last_variants"] = [v["result_id"] for v in variants]
    m["model_used"] = model
    meta_p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    return _ok(variants=variants)

@APP.post("/select")
@_measure
def select():
    data = request.get_json(silent=True) or {}
    rid = data.get("result_id"); img_id = data.get("image_id")
    if not img_id or not rid: return _bad("缺少 image_id 或 result_id")
    d = _image_dir(img_id)
    rp = d/"results"
    if not list(rp.glob(f"{rid}_*.png")): return _bad("找不到 result",404)
    meta_p = d/"meta.json"; m = json.loads(meta_p.read_text(encoding="utf-8"))
    m["selected_result"] = rid
    meta_p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return _ok(image_id=img_id, result_id=rid)

@APP.post("/furniture")
@require_quota
@_measure
def furniture():
    """
    家具 Add/Swap/Recolor：使用 images.edits + 區域 mask
    需要：image_id、result_id（必須已 select）、action、prompt/object、color、mask_data_url（或 bbox）
    """
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
    data = request.get_json(silent=True) or {}
    img_id = data.get("image_id")
    result_id = data.get("result_id")
    action = data.get("action")  # add | swap | recolor
    obj = data.get("object") or data.get("prompt") or ""
    color = data.get("color")
    mask_b64 = data.get("mask_data_url")

    if not (img_id and result_id and action and mask_b64):
        return _bad("缺少必要參數 (image_id/result_id/action/mask_data_url)")

    d = _image_dir(img_id)
    meta = json.loads((d/"meta.json").read_text(encoding="utf-8"))
    if meta.get("selected_result") != result_id:
        return _bad("請先 select 該變體", 403)

    # base image is the selected result
    cand = list((d/"results").glob(f"{result_id}_*.png"))
    if not cand: return _bad("找不到 result",404)
    base_img = cand[0]

    # build prompt
    verb = {"add":"add","swap":"replace","recolor":"recolor"}[action]
    composed = f"{verb} {obj}" + (f" in {color}" if color else "")
    # convert mask_data_url -> RGBA mask
    try:
        import base64, re, io
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", mask_b64)
        mask_bytes = base64.b64decode(m.group(1))
        L = Image.open(io.BytesIO(mask_bytes)).convert("L")
        rgba = _make_openai_alpha_mask_from_white_editable(L)
        buf = io.BytesIO(); rgba.save(buf, "PNG"); buf.seek(0)
    except Exception as e:
        return _bad("mask_data_url 解析失敗")

    out_bytes = edit_image_with_mask(str(base_img), buf, PT.make_furniture_prompt(composed))
    rid = uuid.uuid4().hex[:10]
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = d/"results"/f"{rid}_furniture_{ts}.png"
    with open(out_path, "wb") as f: f.write(out_bytes)
    add_logo(str(out_path), str(STATIC_DIR/"logo/LOGO.png"), str(out_path))

    return _ok(result_id=rid, url=_serve_path(img_id, f"results/{out_path.name}"))

@APP.get("/download")
@_measure
def download():
    """
    下載結果：?image_id=...&result_id=...&fmt=webp|jpg|png&size=standard|hires&logo=1|0
    - 重新壓縮，移除 EXIF
    """
    img_id = request.args.get("image_id")
    rid = request.args.get("result_id")
    fmt = (request.args.get("fmt") or "png").lower()
    size = request.args.get("size","standard")
    addlogo = request.args.get("logo","1") in ("1","true","True")

    if not img_id or not rid: return _bad("缺少 image_id 或 result_id")

    d = _image_dir(img_id)
    cands = list((d/"results").glob(f"{rid}_*.png"))
    if not cands: return _bad("找不到 result",404)
    p = cands[0]
    im = Image.open(p).convert("RGBA")
    if size == "standard":
        im = im.resize((min(1024, im.width), int(im.height*min(1024/im.width,1))), Image.LANCZOS)
    if addlogo:
        add_logo(p, str(STATIC_DIR/"logo/LOGO.png"), None, target_img=im)  # use in-memory option supported by helper
    # strip EXIF by re-encode
    out = io.BytesIO()
    if fmt == "webp":
        im.save(out, "WEBP", quality=92, method=6)
        mimetype = "image/webp"
    elif fmt == "jpg" or fmt == "jpeg":
        im = im.convert("RGB"); im.save(out, "JPEG", quality=92, optimize=True)
        mimetype = "image/jpeg"
    else:
        im.save(out, "PNG")
        mimetype = "image/png"
    out.seek(0)
    return send_file(out, as_attachment=True, download_name=f"{rid}.{fmt}", mimetype=mimetype)

@APP.get("/quota")
def quota():
    # simple values + AB config hint
    reset = int(time.time()) + 300
    return _ok(remaining=9999, reset=reset, ab={"modelA":MODEL_A,"modelB":MODEL_B,"wB":AB_WEIGHT_B})

@APP.post("/events")
def events():
    payload = request.get_json(silent=True) or {}
    print("[EVENT]", json.dumps(payload, ensure_ascii=False))
    return _ok(logged=True)

@APP.get("/meta/model")
def meta_model():
    return _ok(image_model=MODEL_A, ab={"modelB":MODEL_B,"wB":AB_WEIGHT_B})

@APP.patch("/meta/model")
def meta_model_patch():
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
    data = request.get_json(silent=True) or {}
    global MODEL_A, MODEL_B, AB_WEIGHT_B
    if "image_model" in data:
        MODEL_A = data["image_model"]
    if "image_model_b" in data:
        MODEL_B = data["image_model_b"]
    if "ab_weight_b" in data:
        AB_WEIGHT_B = float(data["ab_weight_b"])
    return _ok(image_model=MODEL_A, ab={"modelB":MODEL_B,"wB":AB_WEIGHT_B})

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    APP.run(host="0.0.0.0", port=port, debug=False)



# 嘗試載入 GPT Vision 偵測模組；若缺少檔案或相依，則以 HAS_VISION=False 回覆 501
try:
    from vision_detect import detect_structures as _detect_structures  # GPT Vision v2
    HAS_VISION = True
except Exception:
    _detect_structures = None
    HAS_VISION = False

@APP.post("/detect/v2")
@require_quota
@_measure
def detect_v2():
    """
    GPT Vision 版結構/深度偵測：輸出多邊形分層並產生遮罩與合併疊圖。
    回傳同 /detect 基本欄位，另附 layers_json。
    需設 OPENAI_API_KEY；可設 VISION_MODEL_NAME=gpt-4o|gpt-4o-mini。
    """
    if not _auth_required() and REQUIRE_JWT:
        return _bad("Unauthorized", 401)
            # 若未安裝 vision_detect，回 501，前端會自動回退到 /detect
        if not HAS_VISION or _detect_structures is None:
            return _bad("detect_v2 not available", 501)
    data = request.get_json(silent=True) or {

    img_id = data.get("image_id")
    if not img_id: return _bad("缺少 image_id")
    d = _image_dir(img_id)
    raw = d/"original.png"
    if not raw.exists(): return _bad("原始圖不存在",404)

    # versioning for v2 as well
    vfile = d/"masks"/"version.txt"
    ver = int(vfile.read_text())+1 if vfile.exists() else 1
    vfile.write_text(str(ver))

    out = _detect_structures(str(raw), str(d/"masks"), version=ver, model=os.getenv("VISION_MODEL_NAME","gpt-4o-mini"))
    # Convert local paths to served URLs
    def rel(p): return str(Path(p).relative_to(d))
    return _ok(
        mask_version=ver,
        lock_mask=_serve_path(img_id, rel(out["lock_mask"])),
        editable_mask=_serve_path(img_id, rel(out["editable_mask"])),
        merged_overlay=_serve_path(img_id, rel(out["merged_overlay"])),
        depth_map=_serve_path(img_id, rel(out["depth_map"])),
        layers_json=_serve_path(img_id, rel(out["layers_json"])),
        layers_count=out["layers_count"]
    )

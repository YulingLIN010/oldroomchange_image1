
# -*- coding: utf-8 -*-
import os, io, uuid, json, time, math, re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw

APP = Flask(__name__)
CORS(APP, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS","*").split(",")}})

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- utils ----------------
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

# ---------------- basic routes ----------------
@APP.get("/healthz")
def healthz():
    return ("", 204)

@APP.get("/files/<path:rel>")
def files(rel: str):
    # only serve inside DATA
    p = (DATA / rel).resolve()
    if not str(p).startswith(str(DATA.resolve())):
        return _bad("invalid path", 404)
    if not p.exists():
        return _bad("not found", 404)
    return send_file(str(p))

# ---------------- upload ----------------
@APP.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return _bad("missing file")
    # size check (<= 2 MiB)
    f.seek(0, io.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > 2*1024*1024:
        return _bad("file too large (>2MB)", 413)
    image_id = uuid.uuid4().hex[:10]
    img = Image.open(f.stream).convert("RGB")
    out_dir = DATA / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(img, out_dir/"original.png")
    meta = {"created_at": int(time.time())}
    (out_dir/"meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return _ok(image_id=image_id, **_img_size(img))

# ---------------- detect (v2 -> 501 so frontend will fallback) ----------------
@APP.post("/detect/v2")
def detect_v2():
    return _bad("vision not configured", 501)

# ---------------- detect (naive PIL version) ----------------
@APP.post("/detect")
def detect():
    data = request.get_json(silent=True) or {}
    image_id = data.get("image_id")
    if not image_id:
        return _bad("missing image_id")
    base = DATA / image_id / "original.png"
    if not base.exists():
        return _bad("image not found", 404)
    im = Image.open(base).convert("RGB")
    w, h = im.size
    # naive rule: bright regions (likely窗/燈) -> lock (white)
    gray = ImageOps.grayscale(im)
    lock = gray.point(lambda v: 255 if v > 210 else 0).convert("L")
    # save lock mask v1
    out_dir = DATA / image_id
    ver = int(time.time())
    lock_path = out_dir / f"lock_v{ver}.png"
    lock.save(lock_path)
    # merged overlay: tint green where locked
    overlay = Image.new("RGBA", (w,h), (0,0,0,0))
    g = Image.new("RGBA", (w,h), (30,190,107, int(0.85*255)))
    overlay.paste(g, (0,0), lock)
    merged = Image.alpha_composite(im.convert("RGBA"), overlay)
    merged_path = out_dir / f"merged_v{ver}.png"
    _save(merged, merged_path)
    return _ok(
        detector="naive", detector_detail="pil",
        mask_version=ver,
        lock_mask=_serve_path(image_id, lock_path.name),
        merged_overlay=_serve_path(image_id, merged_path.name),
    )

# ---------------- mask commit ----------------
@APP.post("/mask/commit")
def commit_mask():
    image_id = request.form.get("image_id")
    if not image_id:
        return _bad("missing image_id")
    f = request.files.get("lock_mask")
    if not f:
        return _bad("missing lock_mask")
    out_dir = DATA / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # version
    ver = int(time.time())
    lock_path = out_dir / f"lock_v{ver}.png"
    img = Image.open(f.stream).convert("L")
    # normalize to strict 0/255
    lock = img.point(lambda v: 255 if v > 127 else 0)
    _save(lock, lock_path)
    # editable = invert(lock)
    editable = ImageOps.invert(lock)
    _save(editable, out_dir / f"editable_v{ver}.png")
    # update merged
    base = Image.open(out_dir/"original.png").convert("RGBA")
    g = Image.new("RGBA", base.size, (30,190,107, int(0.85*255)))
    merged = Image.alpha_composite(base, g.putalpha(lock) or g)
    # Note: PIL's putalpha returns None; so compose differently
    g = Image.new("RGBA", base.size, (30,190,107, int(0.85*255)))
    g.putalpha(lock)
    merged = Image.alpha_composite(base, g)
    merged_path = out_dir / f"merged_v{ver}.png"
    _save(merged, merged_path)
    return _ok(mask_version=ver, merged_overlay=_serve_path(image_id, merged_path.name))

# ---------------- styles ----------------
@APP.get("/meta/styles")
def meta_styles():
    table = (Path(__file__).parent/"styles_brief_table.json")
    txt = (Path(__file__).parent/"styles.txt")
    styles = []
    kb_version = int(time.time())
    try:
        if table.exists():
            data = json.loads(table.read_text("utf-8"))
            for it in data:
                name = (it.get("name") or "").strip()
                brief = (it.get("core") or "").strip()
                if name:
                    styles.append({"code": name, "name": name, "brief": brief})
            kb_version = int(table.stat().st_mtime)
        elif txt.exists():
            raw = txt.read_text("utf-8", errors="ignore")
            names = re.findall(r'^\*(\S+)', raw, flags=re.M)
            for nm in names:
                styles.append({"code": nm, "name": nm})
            kb_version = int(txt.stat().st_mtime)
        else:
            styles = [
                {"code":"北歐風","name":"北歐風","brief":"簡約、木質、自然光"},
                {"code":"輕奢古典風","name":"輕奢古典風","brief":"線板＋金屬點綴"},
                {"code":"現代風","name":"現代風","brief":"俐落線條、功能導向"},
            ]
    except Exception as e:
        return _bad(f"styles error: {e}")
    return _ok(styles=styles, kb_version=kb_version)

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)

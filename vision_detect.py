
# -*- coding: utf-8 -*-
"""
vision_detect.py
以 GPT Vision (gpt-4o-mini / gpt-4o) 進行室內結構分層偵測。
回傳多邊形（相對座標 0~1），並產生 lock / editable / depth / merged overlay。
"""
import io, os, json, base64, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageOps

from openai import OpenAI

STRUCT_TYPES = ["beam","column","wall","door","window","ceiling_edge","floor_edge","opening","arch"]
LOCK_TYPES = ["beam","column","wall","door","window","ceiling_edge","floor_edge"]

def _b64_of_image(path: str) -> str:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + b64

PROMPT = """You are an architectural vision model. Detect structural features in the interior photo.
Return STRICT JSON with keys:
{
 "layers": [
   {"type":"beam|column|wall|door|window|ceiling_edge|floor_edge|opening|arch",
    "polygons":[ [ {"x":<0..1>,"y":<0..1>}, ... ], ... ],
    "depth": <0..1>  // optional, relative depth where 0=nearer to camera, 1=farther
   },
   ...
 ],
 "notes": "short reasoning (<=20 words)"
}
Rules:
- Polygons must be closed (first and last point must be the same).
- Use normalized coordinates (0..1) relative to image width/height.
- Prefer few, clean polygons.
- Focus on beams, columns, walls, doors, windows, ceiling/floor edges.
- Do not include furniture as structural layers.
- Output ONLY JSON, nothing else.
"""

def call_gpt_detect(image_path: str, model: str = None) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    image_url = _b64_of_image(image_path)
    model = model or os.getenv("VISION_MODEL_NAME","gpt-4o-mini")
    res = client.responses.create(
        model=model,
        input=[{
            "role":"user",
            "content":[
                {"type":"input_text","text":PROMPT},
                {"type":"input_image","image_url":image_url}
            ]
        }],
        temperature=0.0,
        max_output_tokens=2000
    )
    txt = res.output_text  # SDK 1.40+
    try:
        data = json.loads(txt)
    except Exception:
        # Best-effort: try to extract JSON substring
        import re
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            raise ValueError("Vision model did not return JSON")
        data = json.loads(m.group(0))
    return data

def _denorm(poly: List[Dict[str,float]], size: Tuple[int,int]) -> List[Tuple[int,int]]:
    W,H = size
    return [(max(0,min(W-1,int(p["x"]*W))), max(0,min(H-1,int(p["y"]*H)))) for p in poly]

def rasterize_layers(size: Tuple[int,int], layers: List[Dict[str,Any]]) -> Dict[str,Image.Image]:
    W,H = size
    lock = Image.new("L",(W,H),0)
    depth = Image.new("L",(W,H),0)
    merged = Image.new("RGBA",(W,H),(0,0,0,0))
    draw_lock = ImageDraw.Draw(lock)
    draw_depth = ImageDraw.Draw(depth)
    draw_ov = ImageDraw.Draw(merged)

    # color map for overlay
    colors = {
        "beam":  (220, 80, 80, 110),
        "column":(220, 120, 60, 110),
        "wall":  (220, 60, 120, 80),
        "door":  (60, 120, 220, 110),
        "window":(60, 200, 240, 110),
        "ceiling_edge":(180,180,60,110),
        "floor_edge":(120,180,60,110),
        "opening":(200,120,200,90),
        "arch":(200,160,80,90),
    }

    # Determine depth mapping: use provided "depth" or infer by average y
    # Prepare list with fallback depth
    items = []
    for layer in layers:
        t = (layer.get("type") or "wall").lower()
        for poly in layer.get("polygons",[]):
            d = layer.get("depth")
            if d is None:
                # infer: average y
                ay = sum(p["y"] for p in poly)/max(1,len(poly))
                d = min(1.0,max(0.0, ay))
            items.append({"type":t,"poly":poly,"depth":float(d)})
    # normalize depth to 0..1
    if items:
        ds = [it["depth"] for it in items]
        lo, hi = min(ds), max(ds)
        span = max(1e-6, hi-lo)
        for it in items:
            it["depth"] = (it["depth"]-lo)/span

    # rasterize
    for it in items:
        t = it["type"]
        poly_xy = _denorm(it["poly"], (W,H))
        if len(poly_xy) < 3: continue
        if t in LOCK_TYPES:
            draw_lock.polygon(poly_xy, fill=255)
        col = colors.get(t,(160,160,160,90))
        draw_ov.polygon(poly_xy, fill=col, outline=(255,255,255,130))
        g = int(255 * it["depth"])
        draw_depth.polygon(poly_xy, fill=g)

    editable = ImageOps.invert(lock)  # white=editable
    return {"lock": lock, "editable": editable, "depth": depth, "overlay": merged}

def detect_structures(image_path: str, out_dir: str, version: int, model: str = None) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGBA")
    W,H = img.size
    # Call GPT Vision
    data = call_gpt_detect(image_path, model=model)
    layers = data.get("layers",[])

    ras = rasterize_layers((W,H), layers)
    lock_p = Path(out_dir) / f"lock_v{version}.png"
    edit_p = Path(out_dir) / f"editable_v{version}.png"
    depth_p= Path(out_dir) / f"depth_v{version}.png"
    over_p = Path(out_dir) / f"merged_v{version}.png"
    ras["lock"].save(lock_p,"PNG")
    ras["editable"].save(edit_p,"PNG")
    ras["depth"].save(depth_p,"PNG")

    # blend overlay on top of original for preview
    merged_preview = img.copy()
    merged_preview = Image.alpha_composite(merged_preview, ras["overlay"])
    merged_preview.save(over_p, "PNG")

    # also store raw layers json
    jpath = Path(out_dir) / f"layers_v{version}.json"
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mask_version": version,
        "lock_mask": str(lock_p),
        "editable_mask": str(edit_p),
        "depth_map": str(depth_p),
        "merged_overlay": str(over_p),
        "layers_json": str(jpath),
        "layers_count": len(layers)
    }

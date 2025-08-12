
# -*- coding: utf-8 -*-
# image_logo.py — v2
# 預設 LOGO 路徑：static/logo/LOGO.png
# 支援：位置（四角/置中）、不透明度、邊距比例；自動正規化反斜線

import os
from PIL import Image, ImageEnhance

def _norm(p: str) -> str:
    if not p:
        return p
    return p.replace("\\\\", "/").replace("\\", "/")

def add_logo(input_img_path: str,
             logo_path: str = "static/logo/LOGO.png",
             out_img_path: str = None,
             logo_ratio: float = 0.18,
             position: str = "bottom-right",
             opacity: float = 1.0,
             margin_ratio: float = 0.03) -> str:
    """
    參數：
    - input_img_path：輸入圖檔路徑
    - logo_path     ：LOGO 圖檔路徑（建議 PNG 透明底），預設 static/logo/LOGO.png
    - out_img_path  ：輸出圖檔路徑；若 None 則輸出到 <input>_logo.png
    - logo_ratio    ：LOGO 寬度 / 底圖寬度（0.05~0.5）
    - position      ：top-left/top-right/bottom-left/bottom-right/center
    - opacity       ：LOGO 不透明度 0~1
    - margin_ratio  ：相對於寬度的邊距比例（0~0.2）
    回傳：輸出檔路徑
    """
    input_img_path = _norm(input_img_path)
    logo_path = _norm(logo_path)
    out_img_path = _norm(out_img_path) if out_img_path else None

    base = Image.open(input_img_path).convert("RGBA")
    if not os.path.exists(logo_path):
        # 找不到 LOGO 則直接回存原圖
        out = out_img_path or (os.path.splitext(input_img_path)[0] + "_logo.png")
        base.save(out, "PNG")
        return out

    logo = Image.open(logo_path).convert("RGBA")
    w, h = base.size
    logo_w = max(1, int(w * max(0.05, min(0.5, float(logo_ratio)))))
    logo = logo.resize((logo_w, int(logo_w * logo.height / logo.width)), Image.LANCZOS)

    if opacity < 1.0:
        alpha = logo.split()[-1]
        alpha = ImageEnhance.Brightness(alpha).enhance(max(0.0, min(1.0, opacity)))
        logo.putalpha(alpha)

    pad = int(w * max(0.0, min(0.2, float(margin_ratio))))
    pos_map = {
        "bottom-right": (w - logo.size[0] - pad, h - logo.size[1] - pad),
        "bottom-left": (pad, h - logo.size[1] - pad),
        "top-right": (w - logo.size[0] - pad, pad),
        "top-left": (pad, pad),
        "center": ((w - logo.size[0]) // 2, (h - logo.size[1]) // 2),
    }
    x, y = pos_map.get(position, pos_map["bottom-right"])

    result = base.copy()
    result.alpha_composite(logo, (x, y))

    out = out_img_path or (os.path.splitext(input_img_path)[0] + "_logo.png")
    result.save(out, "PNG")
    return out

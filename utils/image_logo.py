# -*- coding: utf-8 -*-
# 這個模組將公司 LOGO 疊加到底圖右下角，輸出 PNG。
# - 若 LOGO 檔案不存在或無法讀取，會直接回存原圖避免流程中斷。

from PIL import Image

def add_logo(input_img_path, logo_path, out_img_path, logo_ratio=0.18):
    """
    參數：
    - input_img_path：輸入圖檔路徑
    - logo_path：LOGO 圖檔路徑（建議 PNG 透明底）
    - out_img_path：輸出圖檔路徑（PNG）
    - logo_ratio：LOGO 寬度相對於底圖寬度的比例（預設 18%）
    """
    base = Image.open(input_img_path).convert("RGBA")
    try:
        logo = Image.open(logo_path).convert("RGBA")
    except Exception:
        # 找不到 LOGO 就直接輸出原圖
        base.save(out_img_path, "PNG")
        return out_img_path

    # 按比例縮放 LOGO，並放置在右下角，留一點邊距
    w, h = base.size
    logo_w = int(w * logo_ratio)
    logo = logo.resize((logo_w, int(logo_w * logo.height / logo.width)), Image.LANCZOS)

    pad = int(w * 0.03)  # 3% 邊距
    pos = (w - logo.size[0] - pad, h - logo.size[1] - pad)

    result = base.copy()
    result.paste(logo, pos, logo)  # 以 LOGO 作為遮罩疊圖（保留透明度）
    result.save(out_img_path, "PNG")
    return out_img_path
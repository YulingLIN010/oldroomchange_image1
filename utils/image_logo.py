from PIL import Image

def add_logo(input_img_path, logo_path, out_img_path, logo_ratio=0.18):
    """將 logo PNG 貼到右下角（依原圖自動縮放），存出 PNG。"""
    base = Image.open(input_img_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    w, h = base.size
    logo_w = int(w * logo_ratio)
    logo = logo.resize((logo_w, int(logo_w * logo.height / logo.width)), Image.LANCZOS)

    pad = int(w * 0.03)
    pos = (w - logo.size[0] - pad, h - logo.size[1] - pad)

    result = base.copy()
    result.paste(logo, pos, logo)
    result.save(out_img_path, 'PNG')
    return out_img_path


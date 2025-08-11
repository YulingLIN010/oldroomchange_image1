# -*- coding: utf-8 -*-
"""
prompt_templates.py
- 將常見「風格名稱」與「顏色中文」對應到英文描述，便於餵給模型；
- 從 styles.txt 讀取風格清單（支援 * 區塊格式），找不到則 fallback；
- 產生嚴格的室內風格轉換 prompt，並與「透明遮罩 = 可編輯、實心遮罩 = 保留」規則對齊。
"""

import os
from typing import List, Dict

# 常見室內風格（中文 => 英文）
STYLE_EN_MAP = {
    "北歐風": "Scandinavian style",
    "工業風": "Industrial style",
    "現代風": "Modern style",
    "簡約風": "Minimalist style",
    "混搭風": "Eclectic style",
    "鄉村風": "Country style",
    "日式侘寂風": "Wabi-sabi Japanese style",
    "日式禪風": "Zen Japanese style",
    "日式無印風": "MUJI Japanese style",
    "古典風": "Classical style",
    "新古典風格": "Neo-classical style",
    "地中海風": "Mediterranean style",
    "美式風": "American style",
}

# 常見顏色中文 => 英文
COLOR_EN_MAP = {
    "白": "white", "奶油白": "cream white", "灰": "gray", "藍": "blue",
    "金": "gold", "米灰": "beige", "木色": "wood", "淺藍": "light blue",
    "黑": "black", "粉紅": "pink", "米色": "ivory", "綠": "green", "黃色": "yellow",
    "深藍": "navy blue", "淺灰": "light gray", "棕": "brown", "紅": "red",
    "橘": "orange", "銀": "silver", "紫": "purple"
}

def cn_color_to_en(color_str: str) -> str:
    """
    將中文色票字串轉成英文，並把「＋ / + / 、 / ，」等分隔符標準化為逗號，利於模型解析多色系。
    例如：「藍＋金」=> "blue, gold"
    """
    result = color_str or ""
    # 先比對長詞，避免子字串誤替換（如「淺藍」與「藍」）
    color_items = sorted(COLOR_EN_MAP.items(), key=lambda x: -len(x[0]))
    for cn, en in color_items:
        result = result.replace(cn, en)
    # 標準化分隔
    for sep in ("＋", "+", "、", "，", " "):
        result = result.replace(sep, ", ")
    # 收斂多餘逗號與空白
    result = ", ".join([p.strip() for p in result.split(",") if p.strip()])
    return result

def load_styles() -> List[Dict[str, str]]:
    """
    讀取 styles.txt，輸出 [{name, desc}] 清單。
    - 若找不到檔案，就用 STYLE_EN_MAP 的 key 當作預設風格清單。
    - styles.txt 的格式：以星號 * 開頭為一個條目，內含名稱與描述段落。
    """
    path = "styles.txt"
    if not os.path.exists(path):
        return [{"name": k, "desc": k} for k in STYLE_EN_MAP.keys()]
    styles: List[str] = []
    with open(path, encoding="utf-8") as f:
        block: List[str] = []
        for line in f:
            if line.strip().startswith("#"):
                continue
            if line.startswith("*"):
                if block:
                    styles.append("".join(block))
                    block = []
            block.append(line)
        if block:
            styles.append("".join(block))
    result: List[Dict[str, str]] = []
    for s in styles:
        name = s.split("\n")[0].replace("*", "").strip()
        # 嘗試從段落中撈一行當作簡短描述
        desc = ""
        for l in s.split("\n"):
            if "核心" in l or "設計理念" in l or "特色" in l:
                desc = (
                    l.replace("設計理念", "")
                    .replace("特色", "")
                    .replace("核心：", "")
                    .replace("：", "")
                    .strip()
                )
                break
        result.append({"name": name, "desc": (desc[:10] if desc else name)})
    return result

def make_prompt(style: str, colors: str) -> str:
    """
    組合最終給模型的指令文字：
    - 強調「保留原圖視角、布局、深度與結構」
    - 僅以表面材質/家具/燈具等做風格轉換
    - 使用者選的風格與色系會轉換成英文以便模型理解
    - 與「透明遮罩 = 可編輯；實心遮罩 = 保留」規則對齊
    """
    style = style or ""
    colors = colors or ""
    style_en = STYLE_EN_MAP.get(style, style)
    colors_en = cn_color_to_en(colors)

    style_list = load_styles()
    style_info = next((s for s in style_list if s["name"] == style), {"desc": ""})
    style_desc = style_info["desc"] or style

    prompt = f"""
You are an interior design image editor. Modify the provided image in-place to match the target style. Do NOT generate a new scene. Preserve exact geometry, perspective and camera framing from the input photo.

Strictly follow all of the following rules for style conversion.

1. DO NOT change or reinterpret:
   - camera angle, lens perspective, or field of view
   - room size, proportions, or depth
   - wall positions, ceiling height, or layout
   - window and door positions, sizes, or shapes
   - hallways, visible background rooms, or any fixed architectural features

2. DO NOT crop, shift, rotate, or reframe the original image.
3. DO NOT remove, redraw, or reinterpret the spatial boundaries or depth.
4. ONLY overlay and replace the following surface-level elements to match the "{style_en}" interior design style with the color palette: {colors_en}.
    - furniture (consistent with {style_en} aesthetics; use built-in units where appropriate)
    - wall finishes (texture, paint, decorative panels)
    - ceiling treatments (trim, lighting layout)
    - flooring materials (wood, tile, or concrete matching the style)
    - lighting fixtures (ceiling, wall-mounted, or floor types that suit the style)
    - door panel surface finish (but NEVER change door location or dimensions)

5. Remove all existing elements that do not conform to "{style_en}".
6. Apply a coherent visual identity based on "{style_en}" style.
7. Strictly use "{colors_en}" as the dominant visual theme in all replaceable areas.
8. Ensure a clean, unified appearance with no mixed-style clutter.
9. Built-in cabinetry is mandatory for all major furniture pieces.
10. Maintain spatial realism, lighting accuracy, and natural shadows.

MASK SCOPE (IMPORTANT)
- Only modify areas where the mask is transparent (alpha = 0).
- Do NOT alter any opaque (alpha = 255) regions of the mask.

DO NOT VIOLATE STRUCTURAL RULES. The spatial layout and perspective must MATCH EXACTLY the original photograph.

Style description: {style_desc}

Output a single interior design image with the above constraints.
""".strip()
    return prompt

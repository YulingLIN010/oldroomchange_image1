
# -*- coding: utf-8 -*-
"""
prompt_templates.py — 風格提示詞生成（加強版）
- 明確限制：只在「透明遮罩區域」inpainting、更換材質/顏色/風格，不改視角/透視/構圖
- 中英雙語硬性規則，要求輸入輸出尺寸一致且像素對齊
- 新增固定式家具規則：電視櫃、衣櫃、床頭櫃、書櫃、收納櫃、展示櫃、矮櫃、書桌、餐桌、屏風 → built-in/fixed furniture
"""
import json, re
from pathlib import Path

BASE_DIR = Path("/mnt/data").resolve()
STYLES_TABLE = BASE_DIR / "styles_brief_table.json"
STYLES_TXT   = BASE_DIR / "styles.txt"

ZH_COLOR_MAP = {
    "白": "white", "米白": "off-white", "黑": "black", "灰": "gray",
    "淺灰": "light gray", "深灰": "charcoal gray",
    "藍": "blue", "海軍藍": "navy blue", "粉藍": "pastel blue", "靛藍": "indigo",
    "綠": "green", "橄欖綠": "olive green", "鼠尾草綠": "sage green",
    "紅": "red", "酒紅": "burgundy", "磚紅": "brick red",
    "黃": "yellow", "金色": "gold", "琥珀": "amber",
    "棕": "brown", "胡桃木": "walnut", "橡木": "oak", "柚木": "teak",
    "米色": "beige", "奶油色": "cream", "駝色": "camel",
    "粉": "pink", "粉紅": "pink", "珊瑚": "coral",
    "紫": "purple", "薰衣草": "lavender",
    "橘": "orange", "陶土": "terracotta",
}

def _load_styles_table() -> list:
    if STYLES_TABLE.exists():
        try:
            data = json.loads(STYLES_TABLE.read_text(encoding="utf-8"))
            names = []
            for it in data:
                if isinstance(it, str):
                    names.append(it)
                elif isinstance(it, dict) and it.get("name"):
                    names.append(it["name"])
            return names
        except Exception:
            pass
    return []

def _load_styles_txt_names() -> list:
    names = []
    if STYLES_TXT.exists():
        txt = STYLES_TXT.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^\*([^\n\r]+)", txt, flags=re.M):
            name = m.group(1).strip()
            if name:
                names.append(name)
    return names

def load_styles() -> list:
    names = _load_styles_table()
    txt_names = _load_styles_txt_names()
    for n in txt_names:
        if n not in names:
            names.append(n)
    return [{"name": n} for n in names]

def _color_piece(c: str) -> str:
    if not c: return ""
    c = c.strip()
    if c.startswith("#") and (len(c) in (4,7)):
        return c
    return ZH_COLOR_MAP.get(c, c)

def _colors_to_text(colors: dict) -> str:
    if not colors: return ""
    main = _color_piece(colors.get("main",""))
    accs = [_color_piece(colors.get(k,"")) for k in ("acc1","acc2","acc3")]
    accs = [a for a in accs if a]
    parts = []
    if main: parts.append(f"Primary color: {main}")
    if accs: parts.append("Accent colors: " + ", ".join(accs))
    return "; ".join(parts)

def build_style_prompt(style_name: str, colors: dict=None, enforce_hard_rules: bool=True, extra: str="") -> str:
    style_name = (style_name or "").strip()
    colors_text = _colors_to_text(colors or {})

    # 讀資料源（簡表 + 長文）
    brief_dict = {}
    if STYLES_TABLE.exists():
        try:
            for it in json.loads(STYLES_TABLE.read_text(encoding="utf-8")):
                if isinstance(it, dict) and it.get("name"):
                    brief_dict[it["name"]] = it
        except Exception:
            pass

    longform = {}
    if STYLES_TXT.exists():
        txt = STYLES_TXT.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n(?=\*[^\n\r]+)", txt)
        for block in blocks:
            block = block.strip()
            if not block: continue
            m = re.match(r"^\*([^\n\r]+)\s*(.*)$", block, flags=re.S)
            if not m: continue
            name, body = m.group(1).strip(), m.group(2).strip()
            longform[name] = body

    style_core = []
    brief = brief_dict.get(style_name)
    if brief:
        key_bits = []
        for k in ("core","materials","space_feel","colors"):
            v = brief.get(k)
            if isinstance(v, (list,tuple)):
                key_bits.extend([str(x) for x in v])
            elif isinstance(v, str):
                key_bits.append(v)
        if key_bits:
            style_core.append("Style essence: " + ", ".join(key_bits))

    long_desc = longform.get(style_name, "")
    if long_desc:
        clean = re.sub(r"\s+", " ", long_desc)[:900]
        style_core.append("Style notes (zh): " + clean)

    if colors_text:
        style_core.append(colors_text)

    hard_rules = []
    if enforce_hard_rules:
        hard_rules = [
            # Camera & composition locks (zh)
            "【嚴禁變動】相機參數（視角/焦距/透視/地平線/消失點）、構圖框架與畫面邊界的位置。",
            "【嚴禁】裁切、旋轉、縮放、平移整張圖片；所有固定建築構件（樑、柱、牆、門窗框、天花/地坪交界線）之『像素座標』必須保持不變。",
            # Mask behavior (zh)
            "僅允許在『透明遮罩區域』內進行 inpainting／材質與物件替換；不得越界到不透明區域。",
            # Object/layout locks (zh)
            "家具佈局錨點（落地位置/接觸邊）不可改變；允許換材質/款式/顏色，但不得改變位置、尺寸與姿態。",
            "指定家具：電視櫃、衣櫃、床頭櫃、書櫃、收納櫃、展示櫃、矮櫃、書桌、餐桌、屏風等皆採用『固定式家具』（built-in / fixed furniture），避免可移動或鬆散式家具。",
            # Lighting (zh)
            "維持原始採光方向、陰影方向/長度與光比；不得新增或移動窗/門、燈具位置。",
            # Output alignment (zh)
            "輸出影像尺寸與輸入相同並像素對齊；不得新增視野或改變房間尺寸。",
            # English duplicates
            "Do NOT change camera parameters, field of view, horizon line, vanishing points, or overall composition.",
            "Do NOT crop, rotate, scale, or translate the image. Keep all architectural edges in the same pixel positions.",
            "Edit ONLY inside transparent mask regions (inpainting). Do NOT spill edits outside opaque areas.",
            "Keep furniture anchors (contact points) fixed; change material/style/color only, not position/size/pose.",
            "For the following, use built-in (fixed) furniture: TV cabinet, wardrobe, bedside cabinet, bookcase, storage cabinet, display cabinet, low cabinet, desk, dining table, and room divider; avoid movable/loose furniture.",
            "Preserve original lighting direction and shadow behavior; do not add/move windows/doors or luminaires.",
            "Output must match input resolution and align pixel-to-pixel; do not expand field of view or room size.",
        ]

    segments = [
        f"室內設計風格：{style_name} / Interior style: {style_name}",
        *style_core,
        "輸出要求：高真實度材質、正確光影、自然陰影、乾淨收納、避免雜亂與重複圖樣。",
    ]
    if hard_rules:
        segments.append("硬規則 / Hard Constraints: " + " ".join(hard_rules))
    if extra:
        segments.append(extra)

    return "\n".join(segments)

def make_prompt(style, colors):
    """
    相容舊呼叫：colors 可為字串（'#112233' 或 '主,配1,配2,配3'）或 dict。
    """
    if isinstance(colors, dict):
        colors_dict = colors
    else:
        s = (colors or "").strip()
        if not s:
            colors_dict = {}
        elif s.startswith("#"):
            colors_dict = {"main": s}
        else:
            parts = [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]
            keys = ["main","acc1","acc2","acc3"]
            colors_dict = {k:v for k,v in zip(keys, parts)}
    return build_style_prompt(style, colors=colors_dict, enforce_hard_rules=True)


# -*- coding: utf-8 -*-
"""
prompt_templates.py — 混合版風格提示詞生成（對齊 app.py 與前端）
新增：
- load_styles()：讀取 styles_brief_table.json 的 name 欄位；若沒有，從 styles.txt 解析星號開頭的風格名
- make_prompt(style, colors)：相容字串或 dict，呼叫 build_style_prompt

核心：build_style_prompt(style_name, colors:dict=None, enforce_hard_rules=True, extra="")
"""
import json, re, os
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
            # 支援直接列名或物件
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
    """
    styles.txt 解析：取每段以「*風格名」開頭的行
    """
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
    # 合併 txt 風格名（避免重複）
    txt_names = _load_styles_txt_names()
    for n in txt_names:
        if n not in names:
            names.append(n)
    # 輸出格式相容前端：[{name: "..."}]
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
            # 也把整筆存下來，後面可用 core/materials 補字
            for it in json.loads(STYLES_TABLE.read_text(encoding="utf-8")):
                if isinstance(it, dict) and it.get("name"):
                    brief_dict[it["name"]] = it
        except Exception:
            pass

    longform = {}
    if STYLES_TXT.exists():
        txt = STYLES_TXT.read_text(encoding="utf-8", errors="ignore")
        # 以「*風格名」切段
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
        # 中英混寫更穩；限制長度避免提示爆長
        clean = re.sub(r"\s+", " ", long_desc)[:900]
        style_core.append("Style notes (zh): " + clean)

    if colors_text:
        style_core.append(colors_text)

    hard_rules = []
    if enforce_hard_rules:
        hard_rules = [
            "嚴禁變動：相機角度、透視、房間尺寸、牆面/天花/開口（門窗）位置與尺寸、走廊與背景空間。",
            "不可裁切、旋轉、位移或改變構圖框架；維持原始透視與深度關係。",
            "僅允許在表面層作業：牆面飾材、天花、地坪、燈具與可移動家具；固定建築構件（樑柱、門窗框）不得改動。",
            "所有新增家具需為 built-in 或合理尺寸，遵守風格一致性與真實光影。",
        ]

    segments = [
        f"室內設計風格：{style_name} / Interior style: {style_name}",
        *style_core,
        "輸出要求：高真實度材質、正確光影、自然陰影、清爽收納、避免雜亂和重複圖樣。",
    ]
    if hard_rules:
        segments.append("硬規則 / Hard Constraints: " + " ".join(hard_rules))
    if extra:
        segments.append(extra)

    return "\n".join(segments)

def make_prompt(style, colors):
    """
    相容 app 舊呼叫：colors 可為字串（'#112233' 或 '主,配1,配2,配3'）或 dict。
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

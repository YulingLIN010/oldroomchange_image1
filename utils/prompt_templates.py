
# -*- coding: utf-8 -*-
"""
prompt_templates.py — 混合版風格提示詞生成
- 同時參考 styles_brief_table.json（精簡重點）與 styles.txt（長描述）
- 支援中文風格/顏色 → 英文化片語（基礎），或直接用 HEX
- 參數：
    build_style_prompt(style_name, colors_dict, enforce_hard_rules=True, extra="")
  colors_dict: {"main":"#112233或中文/英文", "acc1":..., "acc2":..., "acc3":...}
- enforce_hard_rules=True：加入「嚴禁改變相機視角/結構/開口位置」等硬規則（照片上傳版要開）
"""
import json, re
from pathlib import Path

BASE_DIR = Path("/mnt/data").resolve()
STYLES_TABLE = BASE_DIR / "styles_brief_table.json"
STYLES_TXT   = BASE_DIR / "styles.txt"

# --- 基礎詞庫（可擴充） ---
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

def _load_styles_table() -> dict:
    if STYLES_TABLE.exists():
        try:
            return json.loads(STYLES_TABLE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _load_styles_txt() -> dict:
    """
    styles.txt 建議格式：每個風格用 ==== 分隔，標頭一行風格名，以下為長描述。
    允許自由格式；本函式以最寬鬆方式切段。
    """
    result = {}
    if STYLES_TXT.exists():
        txt = STYLES_TXT.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n={2,}\n", txt)  # 以多個等號分段（若無也能兼容）
        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            if not lines: continue
            name = lines[0]
            body = "\n".join(lines[1:]).strip()
            if name:
                result[name] = body
    return result

def _norm_style_name(name: str) -> str:
    return (name or "").strip()

def _color_piece(c: str) -> str:
    if not c: return ""
    c = c.strip()
    if c.startswith("#") and (len(c) in (4,7)):  # #RGB 或 #RRGGBB
        return c
    # 映射中文到英文
    return ZH_COLOR_MAP.get(c, c)

def _colors_to_text(colors: dict) -> str:
    # 將 main+acc1+acc2+acc3 組為短語：Primary ...; Accents ...
    if not colors: return ""
    main = _color_piece(colors.get("main",""))
    accs = [ _color_piece(colors.get(k,"")) for k in ("acc1","acc2","acc3") ]
    accs = [a for a in accs if a]
    parts = []
    if main:
        parts.append(f"Primary color: {main}")
    if accs:
        parts.append("Accent colors: " + ", ".join(accs))
    return "; ".join(parts)

def build_style_prompt(style_name: str, colors: dict=None, enforce_hard_rules: bool=True, extra: str="") -> str:
    """
    回傳最終提示詞（中英混合更穩定）。
    - 來自 styles_brief_table.json 的關鍵詞會作為「風格核心」
    - 來自 styles.txt 的長敘述會作為「細節補充」
    - colors 轉成英文化段落；若給的是 HEX 也會保留
    - enforce_hard_rules=True 時插入「嚴禁變動結構/視角/開口位置/比例」等規則（照片上傳版用）
    - extra 可補充任務特定說明（如空間/構圖）
    """
    style_name = _norm_style_name(style_name)
    colors_text = _colors_to_text(colors or {})

    # 讀資料源
    table = _load_styles_table()
    longform = _load_styles_txt()

    brief = table.get(style_name) or table.get(style_name.replace("風","")) or ""
    long_desc = longform.get(style_name, "")

    # 核心風格段（英文化關鍵詞＋中文補充更穩）
    style_core = []
    if isinstance(brief, dict):
        # 若表格給的是 dict，可彙整主要欄位
        key_bits = []
        for k,v in brief.items():
            if isinstance(v, (list, tuple)):
                key_bits.extend([str(x) for x in v])
            elif isinstance(v, str):
                key_bits.append(v)
        if key_bits:
            style_core.append("Style essence: " + ", ".join(key_bits))
    elif isinstance(brief, str) and brief.strip():
        style_core.append("Style essence: " + brief.strip())

    if long_desc:
        style_core.append("Style notes (zh): " + re.sub(r"\s+", " ", long_desc)[:800])

    if colors_text:
        style_core.append(colors_text)

    # 硬規則（照片上傳版必須開）
    hard_rules = []
    if enforce_hard_rules:
        hard_rules = [
            "嚴禁變動：相機角度、透視、房間尺寸、牆面/天花/開口（門窗）位置與尺寸、走廊與背景空間。",
            "不可裁切、旋轉、位移或改變構圖框架；維持原始透視與深度關係。",
            "僅允許在表面層作業：牆面飾材、天花、地坪、燈具與可移動家具；固定建築構件（樑柱、門窗框）不得改動。",
            "所有新增家具需為 built-in 或合理尺寸，遵守風格一致性與真實光影。",
        ]

    if extra:
        style_core.append(extra)

    # 合成最終提示詞
    segments = [
        f"室內設計風格：{style_name} / Interior style: {style_name}",
        *style_core,
        "輸出要求：高真實度材質、正確光影、自然陰影、清爽收納、避免雜亂和重複圖樣。",
    ]
    if hard_rules:
        segments.append("硬規則 / Hard Constraints: " + " ".join(hard_rules))

    final_prompt = "\n".join(segments)
    return final_prompt

# -*- coding: utf-8 -*-
"""
prompt_templates.py — 風格提示詞生成（2025-08-16 r4）
- 支援 room_type 與 flags（HAS_TV_ZONE, ALLOW_WET_DRY, ALLOW_NEW_FURNITURE）
- 無遮罩也需鎖定視角/幾何/像素對齊（硬規則三明治，雙語）
- 毛胚屋/空房：可選擇是否置入少量基礎家具（依空間類型）
- 已移除「勿新增結構開孔；開關插座不得改」的限制，可依風格自行變化
"""
import json, re
from pathlib import Path
from typing import Dict, Any, Optional, List

BASE_DIR = Path(__file__).parent.resolve()
STYLES_TABLE = BASE_DIR / "styles_brief_table.json"
STYLES_TXT   = BASE_DIR / "styles.txt"

# --- color helpers ---
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

def _load_styles_table() -> List[str]:
    if STYLES_TABLE.exists():
        try:
            data = json.loads(STYLES_TABLE.read_text(encoding="utf-8"))
            names: List[str] = []
            for it in data:
                if isinstance(it, str):
                    names.append(it)
                elif isinstance(it, dict) and it.get("name"):
                    names.append(it["name"])
            return names
        except Exception:
            pass
    return []

def _load_styles_txt_names() -> List[str]:
    names: List[str] = []
    if STYLES_TXT.exists():
        txt = STYLES_TXT.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^\*([^\n\r]+)", txt, flags=re.M):
            name = m.group(1).strip()
            if name:
                names.append(name)
    return names

def load_styles() -> List[Dict[str,str]]:
    names = _load_styles_table()
    for n in _load_styles_txt_names():
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

# --- Hard rules ---
def _hard_rules_header() -> str:
    return " ".join([
        "【嚴禁】更動相機參數/視角/焦距/透視/地平線與消失點。",
        "【嚴禁】裁切、旋轉、縮放或平移整張圖片；輸出解析度必須與輸入完全一致且逐像素對齊（pixel-aligned）。",
        "【嚴禁】改變任何建築構件（樑、柱、牆、門、窗、天花/地坪交界）的像素座標；",
        "即使在『無遮罩模式』也必須保留所有建築邊界與消失點一致。",
        "僅允許在透明遮罩內 inpainting；不透明區域一律不可改。",
        "家具錨點/位置/尺寸/姿態固定；僅可換材質/款式/顏色。例外：若 flags.ALLOW_NEW_FURNITURE 為真且偵測不到家具錨點（空房/毛胚屋），允許依空間類型置入少量基礎家具，但不得遮擋門窗或改變建築尺度。",
        "保持原始採光方向與陰影長度/光比不變。",
        "(EN) Do NOT change camera/FOV/composition. Do NOT crop/rotate/scale/translate; output = input size, pixel-aligned.",
        "(EN) Keep all architectural edges at identical pixel coordinates even in no-mask mode.",
        "(EN) Edit ONLY inside the transparent mask; opaque regions must remain untouched.",
        "(EN) Keep furniture anchors fixed; change materials/style/colors only. Exception: if flags.ALLOW_NEW_FURNITURE is true and no furniture anchors are detected (empty/shell), you may place a minimal baseline set per room type without blocking doors/windows or altering architecture.",
        "(EN) Preserve original daylight direction and shadow length/ratio."
    ])

def _hard_rules_footer() -> str:
    return _hard_rules_header()

# --- Room modules ---
def _room_module(room_type: Optional[str], flags: Optional[Dict[str, Any]]) -> str:
    rt = (room_type or "").strip().lower()
    flags = flags or {}
    has_tv   = bool(flags.get("HAS_TV_ZONE"))
    allow_wd = bool(flags.get("ALLOW_WET_DRY"))
    allow_new = bool(flags.get("ALLOW_NEW_FURNITURE"))

    if rt == "living":
        seg = [
            "【客廳規則】若需要 TV 牆（旗標為真），可建立或優化 built-in 電視牆與 TV 櫃（牆體尺度不變）。",
            "（條件）需要 TV 牆：在主視牆中線置中配置壁掛 ≥65\"（建議 75\"），視高合理、線材隱藏；若原圖已有 TV 區則沿用其位置。",
            "沙發、茶几與動線保持原位；可換為該風格代表性家具（比例相近）。",
            "天花：依風格做造型＋間接燈；可加主燈/線性燈（表面層）。",
            "(EN) Living: if needed, create or refine a built-in TV wall/cabinet without changing wall dimensions; center a ≥65\" wall-mount on the main wall, or reuse the existing TV zone. Keep sofa/circulation anchors. Feature ceiling with indirect light."
        ]
        if not has_tv:
            seg[1] = "（條件未啟用）略過 TV 牆段落：僅做牆面/櫃體表面材質與色系優化。"
        if allow_new:
            seg += [
                "【空房配置（若為空房）】三人沙發置於合理觀影距離，前方茶几＋地毯；邊几/立燈作閱讀光；保持主要動線淨寬，不遮擋門窗。"
            ]
        return "\n".join(seg)

    if rt == "bedroom":
        seg = [
            "【臥室規則】床頭牆與床向固定；床頭櫃/壁燈可換款但位置不動。",
            "衣櫃採 built-in，門片樣式與把手依風格；鏡面僅作門片嵌鏡。",
            "天花：造型＋柔性間接光；不移動窗簾盒位置。",
            "(EN) Bedroom: keep headboard wall & bed orientation. Built-in wardrobe. Soft indirect ceiling; curtain box stays."
        ]
        if allow_new:
            seg += ["【空房配置】床置中於床頭牆；成對床頭櫃＋燈；沿整面實牆配置 built-in 衣櫃；可選貼牆書桌。"]
        return "\n".join(seg)

    if rt == "study":
        seg = [
            "【書房規則】書桌貼牆 built-in（原位不動），線槽/收納隱藏；書櫃為內嵌式，層板節奏符合風格。",
            "增加工作照明（桌燈/洗牆線性光），避免眩光；允許織物牆板/窗簾作聲學軟裝（表面層）。",
            "(EN) Study: built-in desk/shelving; task lighting & wall-wash; acoustic textiles allowed."
        ]
        if allow_new:
            seg += ["【空房配置】貼牆 L 形或直線型書桌＋層架；人體工學椅＋工作燈；保留走道淨寬。"]
        return "\n".join(seg)

    if rt == "dining":
        seg = [
            "【餐廳規則】餐桌位置不動；吊燈置中於餐桌（可換款/數量），垂直投影對準桌面中心。",
            "餐邊櫃為 built-in；保持動線淨寬。",
            "(EN) Dining: keep table anchor; center pendant(s); built-in sideboard; preserve circulation."
        ]
        if allow_new:
            seg += ["【空房配置】餐桌置中、對位吊燈、貼牆餐邊櫃；椅數依尺度為 4–6。"]
        return "\n".join(seg)

    if rt == "kitchen":
        seg = [
            "【廚房規則】流理檯/水槽/爐具/冰箱/洗碗機位置固定；僅換門片/把手/檯面/防濺牆材質與色系。",
            "上/下櫃為內嵌式；把手造型、踢腳/收邊、檯面前緣符合風格；照明允許檯面下洗牆與天花主光（表面層）。",
            "(EN) Kitchen: keep appliance & sink/hob anchors; fronts/handles/countertop/backsplash change only; built-in cabinetry."
        ]
        if allow_new:
            seg += ["【空房配置】直線或 L 形貼牆櫥櫃，上/下櫃＋防濺牆；空間足夠可加窄中島（不影響動線）。"]
        return "\n".join(seg)

    if rt == "bathroom":
        seg = [
            "【衛浴規則】馬桶、洗手台、浴缸/淋浴間位置不變；僅換牆/地磚、檯面、鏡櫃與五金。",
            "（條件）若空間足夠且允許，做『乾濕分離』：玻璃隔間或擋水條；門向與排水坡度不得改。",
            "照明：防潮等級＋鏡前燈，演色性佳；通風配置不變。",
            "(EN) Bathroom: plumbing anchors fixed; tiles/vanity/fixtures may change. Optional wet/dry partition if feasible; keep drainage falls."
        ]
        if not allow_wd:
            seg[1] = "（條件未啟用）不執行乾濕分離；僅進行表面材質與五金優化。"
        if allow_new:
            seg += ["【空房配置】盥洗台＋鏡櫃、壁掛層板與五金；示意性置入，不影響建築尺度與排水。"]
        return "\n".join(seg)

    if rt == "balcony":
        seg = [
            "【陽台規則】欄杆/女兒牆/外窗像素邊界不變；僅換戶外耐候地坪、牆面塗料與洗衣機/收納櫃外觀（位置不動）。",
            "允許 built-in 收納櫃與簡易植栽槽（表層件），避免遮擋採光與窗啟閉。",
            "(EN) Balcony: keep railing & external windows; outdoor-rated finishes; built-in utility storage."
        ]
        if allow_new:
            seg += ["【空房配置】立式或貼牆收納櫃、簡易植栽槽或曬衣桿（不遮擋採光/窗啟閉）。"]
        return "\n".join(seg)

    if rt == "entry":
        seg = [
            "【玄關規則】鞋櫃 built-in（含坐墊位），鏡面作門片或壁飾；地坪導入線性分色，呼應主色。",
            "照明：洗牆或腳燈導引；門邊器件位置不動。",
            "(EN) Entry: built-in shoe cabinet/bench; subtle floor delineation; guidance lighting."
        ]
        if allow_new:
            seg += ["【空房配置】內嵌鞋櫃＋坐墊位；壁面鏡飾。"]
        return "\n".join(seg)

    if rt == "corridor":
        seg = [
            "【走廊規則】連續天花與線性間接光；牆面可加護牆板或節奏線條（表面層）。",
            "門套/門片僅改表面材與色；位置與尺寸不變。",
            "(EN) Corridor: continuous ceiling & linear cove; wall paneling allowed; door leaf surfaces may change."
        ]
        if allow_new:
            seg += ["【空房配置】狹長案幾或壁掛層板（不縮小通道淨寬），線性洗牆燈導引。"]
        return "\n".join(seg)

    return ""

# --- Builder ---
def build_style_prompt(style_name: str, colors: dict=None,
                       room_type: Optional[str]=None, flags: Optional[Dict[str, Any]]=None,
                       enforce_hard_rules: bool=True, extra: str="") -> str:
    style_name = (style_name or "").strip()
    colors_text = _colors_to_text(colors or {})
    rt = (room_type or "").strip().lower()

    # 風格資訊（簡短 from styles_brief_table.json + 長文 from styles.txt）
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
            longform[name] = re.sub(r"\s+", " ", body)[:900]

    style_palette = [
        f"室內設計風格：{style_name} / Interior style: {style_name}",
        colors_text if colors_text else "",
        "單一風格純粹（Do not mix styles）。"
    ]

    task_focus = [
        "只做『表面換材/換色』與『家具換款/換材』；櫃體一律 built-in 固定式（TV 櫃、衣櫃、床頭櫃、書櫃、收納/展示櫃、矮櫃、餐邊櫃、屏風…以內嵌/貼牆優先）。",
        "兩段式生成：A. 建築表面（牆/天花/地坪/壁面）統一風格與色系；B. 家具/燈具/軟裝改款與配色（保留錨點）。",
        "若偵測不到任何家具錨點（毛胚屋/空房）：flags.ALLOW_NEW_FURNITURE=false → 僅處理建築表面；flags.ALLOW_NEW_FURNITURE=true → 依空間類型置入少量基礎家具。",
        "輸出要求：高真實度材質、正確光影、自然陰影、乾淨收納、避免雜亂與重複圖樣。"
    ]

    room_seg = _room_module(rt, flags or {})

    style_core = []
    if brief_dict.get(style_name):
        brief = brief_dict[style_name]
        key_bits = []
        for k in ("core","materials","space_feel","colors"):
            v = brief.get(k)
            if isinstance(v, (list,tuple)):
                key_bits.extend([str(x) for x in v])
            elif isinstance(v, str):
                key_bits.append(v)
        if key_bits:
            style_core.append("Style essence: " + ", ".join(key_bits))
    if longform.get(style_name):
        style_core.append("Style notes (zh): " + longform[style_name])

    segments = []
    if enforce_hard_rules:
        segments.append("[HARD RULES — HEADER]\n" + _hard_rules_header())
    segments += [s for s in style_palette if s]
    segments += style_core
    segments += task_focus
    if room_seg:
        segments.append(room_seg)
    if extra:
        segments.append(extra)
    if enforce_hard_rules:
        segments.append("[HARD RULES — FOOTER]\n" + _hard_rules_footer())

    return "\n".join(segments)

def make_prompt(style, colors, room_type: Optional[str]=None, flags: Optional[Dict[str, Any]]=None):
    """Back-compat：舊簽名 (style, colors) 仍可用；若提供 room_type/flags 則注入。"""
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
    return build_style_prompt(style, colors=colors_dict, room_type=room_type, flags=flags, enforce_hard_rules=True)

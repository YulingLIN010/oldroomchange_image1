import os

# 風格中→英文對照（可依你的13種需求擴充）
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

COLOR_EN_MAP = {
    "白": "white", "奶油白": "cream white", "灰": "gray", "藍": "blue",
    "金": "gold", "米灰": "beige", "木色": "wood", "淺藍": "light blue",
    "黑": "black", "粉紅": "pink", "米色": "ivory", "綠": "green", "黃色": "yellow",
    "深藍": "navy blue", "淺灰": "light gray", "棕": "brown", "紅": "red",
    "橘": "orange", "銀": "silver", "紫": "purple"
}

def cn_color_to_en(color_str):
    """將中文色系描述轉為英文，用英文逗號分隔"""
    result = color_str
    color_items = sorted(COLOR_EN_MAP.items(), key=lambda x: -len(x[0]))
    for cn, en in color_items:
        result = result.replace(cn, en)
    result = result.replace("＋", ", ").replace("+", ", ")
    return result

def load_styles():
    """解析 styles.txt，跳過#註解，取得風格中文名稱與簡介（供前端用）"""
    styles = []
    with open('styles.txt', encoding='utf-8') as f:
        block = []
        for line in f:
            if line.strip().startswith("#"):
                continue  # 跳過註解行
            if line.startswith('*'):
                if block:
                    styles.append(''.join(block))
                    block = []
            block.append(line)
        if block:
            styles.append(''.join(block))
    result = []
    for s in styles:
        name = s.split('\n')[0].replace('*', '').strip()
        desc = ""
        for l in s.split('\n'):
            if "核心" in l or "設計理念" in l or "特色" in l:
                desc = l.replace('設計理念', '').replace('特色', '').replace('核心：', '').replace('：', '').strip()
                break
        result.append({
            'name': name,
            'desc': desc[:10] if desc else name
        })
    return result

def make_prompt(style, colors):
    """自動中翻英，並輸出條列英文 prompt，DALL·E 最佳化用"""
    style_en = STYLE_EN_MAP.get(style, style)
    colors_en = cn_color_to_en(colors)
    style_list = load_styles()
    style_info = next((s for s in style_list if s['name'] == style), {'desc': ''})
    style_desc = style_info['desc'] or style

    prompt = f"""
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
    - furniture (must be consistent with {style_en} aesthetics; use built-in units where appropriate)
    - wall finishes (e.g., texture, paint, decorative panels)
    - ceiling treatments (e.g., trim, lighting layout)
    - flooring materials (e.g., wood, tile, or concrete matching the style)
    - lighting fixtures (ceiling, wall-mounted, or floor types that suit the style)
    - door panel surface finish (but NEVER change door location or dimensions)

5. Remove all existing elements that do not conform to "{style_en}".
6. Apply a coherent visual identity based on "{style_en}" style.
7. Strictly use "{colors_en}" as the dominant visual theme in all replaceable areas.
8. Ensure a clean, unified appearance with no mixed-style clutter.
9. Built-in cabinetry is mandatory for all major furniture pieces.
10. Maintain spatial realism, lighting accuracy, and natural shadows.

DO NOT VIOLATE STRUCTURAL RULES. All spatial layout and perspective must MATCH EXACTLY the original photograph.

Style description: {style_desc}

Output a single interior design image with the above constraints.
"""
    return prompt.strip()


# -*- coding: utf-8 -*-
"""
dalle_api.py
- 封裝 OpenAI 的影像生圖（generate）與影像編輯（edit）
- 重點：支援「參考圖 + 透明遮罩 + prompt」的編輯流程
  * 透明遮罩規則：alpha=0 為可編輯區；alpha=255 為保留區
- 採用 lazy client，避免在模組匯入時就因環境或相依問題中斷
"""

import os
import base64
import time
from typing import Optional

from openai import OpenAI

# 模型名稱可由環境變數覆蓋，預設 gpt-image-1
MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gpt-image-1")

# ---- Lazy Client（第一次用到才建立）----
_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

# ---- 小工具：重試與 b64 轉 bytes ----
def _with_retry(fn, retries=3, backoff=2.0):
    """
    指數退避重試：backoff * (2**attempt)
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise last_err

def _decode_b64_image(res) -> bytes:
    """
    OpenAI 圖像 API 回傳物件中的第一張圖 b64 → bytes
    """
    b64 = res.data[0].b64_json
    return base64.b64decode(b64)

# ---- Public APIs ----
def generate_image(prompt: str, size: str = "1024x1024", transparent: bool = False,
                   retries: int = 3, backoff: float = 2.0) -> bytes:
    """
    純文字生圖（無參考圖/遮罩）
    - prompt：描述
    - size  ："1024x1024" / "512x512" / ...
    - transparent=True 時會設定 background="transparent"
    回傳：PNG 圖片位元組（bytes）
    """
    def _call():
        client = _get_client()
        params = dict(model=MODEL_NAME, prompt=prompt, size=size, n=1)
        if transparent:
            params["background"] = "transparent"
        res = client.images.generate(**params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_with_mask(image_path: str, mask_path: str, prompt: str, size: str = "1024x1024",
                         transparent: bool = False, retries: int = 3, backoff: float = 2.0) -> bytes:
    """
    影像編輯（參考圖 + 透明遮罩 + prompt）
    - image_path：原始參考圖（檔案路徑）
    - mask_path ：透明 PNG 遮罩（alpha=0 可編輯；alpha=255 保留）。尺寸必須與 image 相同
    - prompt    ：風格/顏色等描述
    - size/transparent 同上
    回傳：編輯後 PNG 圖片位元組（bytes）
    """
    def _call():
        client = _get_client()
        params = dict(model=MODEL_NAME, prompt=prompt, size=size, n=1)
        if transparent:
            params["background"] = "transparent"

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            # SDK 版本差異：有的叫 images.edits、有的叫 images.edit
            try:
                res = getattr(client.images, "edits")(image=img_f, mask=mask_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, mask=mask_f, **params)
        return _decode_b64_image(res)

    return _with_retry(_call, retries=retries, backoff=backoff)

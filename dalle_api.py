
# -*- coding: utf-8 -*-
"""
dalle_api.py (merged clean version)
- generate_image：純文字生圖（仍保留）
- edit_image_with_mask：原圖 + 遮罩 + prompt（inpainting）
- edit_image_no_mask：原圖 + prompt（無遮罩整張編輯）
"""
import os, base64, time
from typing import Optional
from openai import OpenAI

MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gpt-image-1")
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "low").strip().lower()

_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def _with_retry(fn, retries=3, backoff=2.0):
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
    b64 = res.data[0].b64_json
    return base64.b64decode(b64)

def generate_image(prompt: str, size: str = "1024x1024", transparent: bool = False,
                   retries: int = 3, backoff: float = 2.0) -> bytes:
    def _call():
        client = _get_client()
        params = dict(model=MODEL_NAME, prompt=prompt, size=size, n=1)
        if IMAGE_QUALITY in ("low","medium","high"):
            params["quality"] = IMAGE_QUALITY
        if transparent:
            params["background"] = "transparent"
        res = client.images.generate(**params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_with_mask(image_path: str, mask_path: str, prompt: str, size: str = "1024x1024",
                         transparent: bool = False, retries: int = 3, backoff: float = 2.0) -> bytes:
    def _call():
        client = _get_client()
        params = dict(model=MODEL_NAME, prompt=prompt, size=size, n=1)
        if IMAGE_QUALITY in ("low","medium","high"):
            params["quality"] = IMAGE_QUALITY
        if transparent:
            params["background"] = "transparent"
        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            # 新版 SDK 是 .edits；舊版也可能存在 .edit，故做兼容
            try:
                res = getattr(client.images, "edits")(image=img_f, mask=mask_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, mask=mask_f, **params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_no_mask(image_path: str, prompt: str, size: str = "1024x1024",
                       transparent: bool = False, retries: int = 3, backoff: float = 2.0) -> bytes:
    """
    無遮罩整張編輯：參考原圖 + prompt，但允許全畫面重繪（模型通常保留大構圖）
    """
    def _call():
        client = _get_client()
        params = dict(model=MODEL_NAME, prompt=prompt, size=size, n=1)
        if IMAGE_QUALITY in ("low","medium","high"):
            params["quality"] = IMAGE_QUALITY
        if transparent:
            params["background"] = "transparent"
        with open(image_path, "rb") as img_f:
            try:
                res = getattr(client.images, "edits")(image=img_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, **params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

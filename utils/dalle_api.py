
# utils/dalle_api.py
from openai import OpenAI
import os
import base64
import time

# 建議 requirements.txt：openai>=1.40.0
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-image-3"  # ← 這裡換成 gpt-image-3

def _decode_b64_image(res):
    b64 = res.data[0].b64_json
    return base64.b64decode(b64)

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

def generate_image(prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    """
    純生成：不參考原圖。透明背景可用 transparent=True。
    """
    def _call():
        params = dict(model=MODEL_NAME, prompt=prompt, size=size)
        if transparent:
            params["background"] = "transparent"
        res = client.images.generate(**params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_with_style(image_path, prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    """
    基於單一原圖做整體風格化（不使用遮罩）。
    """
    def _call():
        params = dict(model=MODEL_NAME, prompt=prompt, size=size)
        if transparent:
            params["background"] = "transparent"
        with open(image_path, "rb") as img_f:
            # 先嘗試新版 .edits；沒有就 fallback 到 .edit
            try:
                res = getattr(client.images, "edits")(image=img_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, **params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_with_mask(image_path, mask_path, prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    """
    基於原圖 + 遮罩（白=可編輯，黑=保護）做局部風格化。
    """
    def _call():
        params = dict(model=MODEL_NAME, prompt=prompt, size=size)
        if transparent:
            params["background"] = "transparent"
        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            # 先嘗試新版 .edits；沒有就 fallback 到 .edit
            try:
                res = getattr(client.images, "edits")(image=img_f, mask=mask_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, mask=mask_f, **params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

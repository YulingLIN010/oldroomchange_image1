# -*- coding: utf-8 -*-
# 這個模組封裝了 OpenAI 影像生成功能（文字轉圖）與影像編輯功能（基底圖 + 遮罩微調）。
# 目的：讓主程式以簡潔方式呼叫 generate 與 edit，並統一錯誤重試與回傳格式。

from openai import OpenAI
import os
import base64
import time
from PIL import Image

# 建議在系統環境變數設定 OPENAI_API_KEY；模型名稱可用 IMAGE_MODEL_NAME 覆蓋，預設 gpt-image-1
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gpt-image-1")

def _decode_b64_image(res):
    """
    將 OpenAI 回傳的 base64 圖片轉成位元組（bytes）。
    res.data[0].b64_json 為 base64 字串，需先解碼後才能寫入檔案。
    """
    b64 = res.data[0].b64_json
    return base64.b64decode(b64)

def _with_retry(fn, retries=3, backoff=2.0):
    """
    簡單的重試包裝器：
    - retries：重試次數
    - backoff：指數退避基礎秒數，重試間隔會乘以 2^attempt
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
                # 最終仍失敗，將錯誤往外丟
                raise last_err

def generate_image(prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    """
    純文字生圖：輸入 prompt 與尺寸，回傳 PNG 圖片位元組。
    transparent=True 時，背景透明（需要 PNG 輸出）。
    """
    def _call():
        params = dict(model=MODEL_NAME, prompt=prompt, size=size)
        if transparent:
            params["background"] = "transparent"
        res = client.images.generate(**params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)

def edit_image_with_mask(image_path, mask_path, prompt, size="1024x1024",
                         transparent=False, retries=3, backoff=2.0):
    """
    影像編輯：提供基底圖 image + 遮罩 mask（白=可編輯、黑=保護）以及文字描述 prompt。
    回傳編輯後圖片的位元組。
    """
    def _call():
        params = dict(model=MODEL_NAME, prompt=prompt, size=size)
        if transparent:
            params["background"] = "transparent"

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            # SDK 有的版本方法名是 images.edits，有的是 images.edit，做一次相容處理
            try:
                res = getattr(client.images, "edits")(image=img_f, mask=mask_f, **params)
            except AttributeError:
                res = getattr(client.images, "edit")(image=img_f, mask=mask_f, **params)
        return _decode_b64_image(res)
    return _with_retry(_call, retries=retries, backoff=backoff)
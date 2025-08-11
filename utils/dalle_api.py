
from openai import OpenAI
import os
import base64
import time

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_image(prompt, size="1024x1024"):
    """
    使用 OpenAI gpt-image-1 產生圖片，回傳 PNG 位元組。
    """
    if not client:
        raise Exception("OpenAI client not initialized")
    res = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        # background="transparent",  # 若需透明背景可開啟
    )
    b64 = res.data[0].b64_json
    return base64.b64decode(b64)

def edit_image_with_style(image_path, prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    last_err=None
    for attempt in range(retries):
        try:
            params=dict(model="gpt-image-1", prompt=prompt, size=size)
            if transparent: params['background']='transparent'
            with open(image_path,'rb') as f:
                res=client.images.edits(image=f, **params)
            b64=res.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            last_err=e
            if attempt<retries-1: time.sleep(backoff*(2**attempt))
            else: raise last_err

def edit_image_with_mask(image_path, mask_path, prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    last_err=None
    for attempt in range(retries):
        try:
            params=dict(model="gpt-image-1", prompt=prompt, size=size)
            if transparent: params['background']='transparent'
            with open(image_path,'rb') as img_f, open(mask_path,'rb') as mask_f:
                res=client.images.edits(image=img_f, mask=mask_f, **params)
            b64=res.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            last_err=e
            if attempt<retries-1: time.sleep(backoff*(2**attempt))
            else: raise last_err

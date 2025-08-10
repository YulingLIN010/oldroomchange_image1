from openai import OpenAI
import os
import base64
import time  # ← 必加

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=60.0))
def generate_image(prompt, size="1024x1024", transparent=False, retries=3, backoff=2.0):
    """
    使用 OpenAI gpt-image-1 產生 PNG 位元組。
    transparent=True → 背景透明
    """
    last_err = None
    for attempt in range(retries):
        try:
            params = dict(model="gpt-image-1", prompt=prompt, size=size)
            if transparent:
                params["background"] = "transparent"
            res = client.images.generate(**params)
            b64 = res.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise e





# 舊屋換新風格 — 完整功能（含偵測、家具、下載、配額、治理、安全、觀測）

## 主要功能與端點
- `POST /upload` 上傳 ≤2MB
- `POST /detect` 自動偵測（OpenCV 若可，用邊緣/拉普拉斯；否則安全預設）
- `POST /mask/commit` 最終遮罩
- `POST /generate` gpt-image-1 編輯（支援 AB 測試）
- `POST /select` 選定變體（家具 Gate）
- `POST /furniture` 新增/更換/改色（需選定變體；區域以 mask_data_url）
- `GET /compare` 四格比對牆
- `GET /download` 下載（webp/jpg/png、hires/standard、去 EXIF）
- `DELETE /image/<id>` 刪除影像資料
- `GET /meta/styles` 風格清單
- `GET /meta/model`、`PATCH /meta/model` 模型治理（需 JWT 時才開）
- `GET /quota` 配額資訊（示範）
- `POST /events` 事件記錄
- `GET /metrics` Prometheus 監控輸出
- `GET /healthz` 健康檢查
- `GET /files/<id>/<path>` 檔案輸出（Cache-Control 已設）

## 前端
- `frontend/index.html`（預設 BASE_URL = `https://oldroomchange-image3.onrender.com`；可用 `window.API_BASE` 覆蓋）
- 支援：上傳即預覽、Canvas 遮罩、Auto 偵測、風格≤3＋色卡、生成、四格牆、選定、矩形框選家具區域、家具操作、下載格式選擇

## 部署
- `gunicorn app:APP --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
- Env：`OPENAI_API_KEY`、`ALLOWED_ORIGINS`、`REQUIRE_JWT`(0/1)、`JWT_SECRET`、`IMAGE_MODEL_NAME`、`IMAGE_MODEL_B`、`AB_WEIGHT_B`
- `requirements.txt` 已含：Flask, OpenAI, Pillow, opencv-python-headless, numpy, PyJWT, prometheus-client 等

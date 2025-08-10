
# Deploy Guide

- Set `OPENAI_API_KEY` in environment
- `pip install -r requirements.txt`
- Dev: `python app.py`
- Prod: `gunicorn app:app --workers 3 --threads 2 --timeout 180 --bind 0.0.0.0:5000`
- Optional: POST `/generate` JSON 可含 `transparent: true`
- Health check: GET `/healthz`

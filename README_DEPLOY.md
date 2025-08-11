
# Deploy Guide
- Set `OPENAI_API_KEY` in environment
- `pip install -r requirements.txt`
- Dev: `python app.py`
- Prod: `gunicorn app:app --workers 3 --threads 2 --timeout 180 --bind 0.0.0.0:5000`
- Health check: GET `/healthz`
- Frontend sends `mask` = `smart|safe_edges|full` and optional `mask_options`.

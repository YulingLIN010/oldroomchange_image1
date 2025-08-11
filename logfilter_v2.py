# -*- coding: utf-8 -*-
# logfilter_v2.py
# 目的：過濾 Gunicorn access log 中的 Render 健康檢查請求（/healthz 或 UA: Render/*）

import logging

class HealthzFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = getattr(record, "args", {}) or {}
        ua = args.get("a")
        req = args.get("r")

        # 兼容 bytes/None
        try:
            ua = (ua.decode("latin1") if isinstance(ua, (bytes, bytearray)) else (ua or "")).strip()
        except Exception:
            ua = str(ua)
        try:
            req = (req.decode("latin1") if isinstance(req, (bytes, bytearray)) else (req or "")).strip()
        except Exception:
            req = str(req)

        # 丟棄：Render 健康檢查
        if ua.startswith("Render/") or "GET /healthz" in req or " /healthz " in req:
            return False
        return True

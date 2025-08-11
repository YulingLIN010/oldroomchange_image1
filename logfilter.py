# -*- coding: utf-8 -*-
# logfilter.py
# 目的：過濾 Gunicorn access log 中的 Render 健康檢查請求（/healthz 或 UA: Render/*）

import logging

class HealthzFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # gunicorn.access 的 record.args 是一個 dict，包含：
        # h(遠端IP) r(請求行) s(狀態) a(User-Agent) 等鍵。
        args = getattr(record, "args", {}) or {}

        ua = args.get("a")
        req = args.get("r")

        # 防禦：可能是 bytes，轉成字串避免比較失敗
        if isinstance(ua, (bytes, bytearray)):
            try:
                ua = ua.decode("latin1")
            except Exception:
                ua = str(ua)
        if isinstance(req, (bytes, bytearray)):
            try:
                req = req.decode("latin1")
            except Exception:
                req = str(req)

        ua = (ua or "").strip()
        req = (req or "").strip()

        # 丟棄 Render 健康檢查：/healthz 或 User-Agent: Render/*
        if ua.startswith("Render/"):
            return False
        if " /healthz " in f" {req} ":
            return False

        return True

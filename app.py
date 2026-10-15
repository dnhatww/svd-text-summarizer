# -*- coding: utf-8 -*-
"""
app.py — Flask server cho SVD Text Summarizer.

Routes:
    GET  /                  Giao diện web
    GET  /api/health        Kiểm tra trạng thái + bộ tách từ đang dùng
    POST /api/summarize     Tóm tắt văn bản (JSON)
    POST /api/upload        Đọc file .txt (multipart/form-data, UTF-8, ≤ 2 MB)
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from summarizer import (MAX_TEXT_CHARS, SummarizerError, get_tokenizer_name,
                        summarize)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 64 * 1024  # dư cho header multipart
app.config["JSON_AS_ASCII"] = False         # Flask < 2.3
app.json.ensure_ascii = False               # Flask ≥ 2.3: giữ nguyên tiếng Việt trong JSON
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("svd-summarizer")


def error_response(message: str, status: int, code: str = "error"):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status


@app.get("/")
def index():
    return render_template("index.html", max_chars=MAX_TEXT_CHARS)


@app.get("/api/health")
def health():
    return jsonify({"success": True, "status": "ok", "tokenizer": get_tokenizer_name()})


def _parse_bool(value, default=True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/summarize")
def api_summarize():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response("Dữ liệu gửi lên phải là JSON object hợp lệ.", 400, "invalid_json")
    if "text" not in payload:
        return error_response("Thiếu trường 'text'.", 400, "missing_text")
    try:
        result = summarize(
            text=payload.get("text"),
            k=payload.get("k", 2),
            ratio=payload.get("ratio", 30),
            weighting=payload.get("weighting", "bow"),
            remove_stopwords=_parse_bool(payload.get("remove_stopwords"), True),
        )
    except SummarizerError as exc:
        return error_response(exc.message, exc.status_code, exc.code)
    except Exception:  # lỗi không lường trước -> không lộ stack trace
        log.exception("Unexpected error in /api/summarize")
        return error_response("Đã xảy ra lỗi máy chủ khi xử lý văn bản.", 500, "internal_error")
    return jsonify(result), 200


@app.post("/api/upload")
def api_upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return error_response("Chưa chọn tệp.", 400, "no_file")
    if not file.filename.lower().endswith(".txt"):
        return error_response("Chỉ chấp nhận tệp .txt.", 400, "invalid_file_type")
    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return error_response("Tệp vượt quá dung lượng tối đa 2 MB.", 413, "file_too_large")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return error_response("Tệp không phải mã hóa UTF-8 hợp lệ.", 400, "invalid_encoding")
    if not text.strip():
        return error_response("Tệp rỗng.", 400, "empty_file")
    return jsonify({"success": True, "filename": file.filename, "text": text,
                    "chars": len(text)}), 200


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return error_response("Dữ liệu gửi lên quá lớn (tối đa 2 MB).", 413, "payload_too_large")


@app.errorhandler(HTTPException)
def http_error(exc: HTTPException):
    if request.path.startswith("/api/"):
        return error_response(exc.description or exc.name, exc.code or 500, exc.name.lower().replace(" ", "_"))
    return exc


if __name__ == "__main__":
    import sys
    # Buộc stdout dùng UTF-8 để in ký tự tiếng Việt trên Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    tokenizer = get_tokenizer_name()
    print(f"[SVD Summarizer] Tokenizer: {tokenizer}")
    print(f"[SVD Summarizer] Running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)

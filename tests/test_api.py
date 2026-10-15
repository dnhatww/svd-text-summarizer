"""Kiểm thử Flask API."""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

SAMPLE = "AI phát triển công nghệ. AI ứng dụng trong giáo dục. Giáo dục sử dụng công nghệ AI."


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "SVD Text Summarizer" in html
    assert "Tóm tắt văn bản" in html


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["tokenizer"] in {"underthesea", "pyvi", "regex"}


def test_summarize_success_schema(client):
    res = client.post("/api/summarize", json={"text": SAMPLE, "k": 2, "ratio": 30, "weighting": "bow"})
    assert res.status_code == 200
    assert res.content_type.startswith("application/json")
    data = res.get_json()
    assert data["success"] is True
    for key in ["summary", "statistics", "sentences", "singular_values", "matrix",
                "selected_indices", "config", "warnings"]:
        assert key in data
    st = data["statistics"]
    for key in ["original_sentences", "selected_sentences", "actual_ratio", "k_used",
                "unique_terms", "matrix_rank"]:
        assert key in st
    assert st["original_sentences"] == 3
    assert st["actual_ratio"] == pytest.approx(33.33)
    for s in data["sentences"]:
        assert set(["id", "text", "score", "rank", "selected"]) <= set(s)
    m = data["matrix"]
    assert len(m["values"]) == len(m["words"])
    assert all(len(row) == len(m["columns"]) for row in m["values"])
    assert data["singular_values"] == sorted(data["singular_values"], reverse=True)
    # JSON giữ nguyên tiếng Việt (không escape \uXXXX)
    assert "phát triển" in res.get_data(as_text=True)


def test_summarize_defaults(client):
    res = client.post("/api/summarize", json={"text": SAMPLE})
    assert res.status_code == 200
    assert res.get_json()["config"]["k_requested"] == 2


@pytest.mark.parametrize("payload,status,code", [
    ({"text": ""}, 400, "empty_text"),
    ({"text": "   "}, 400, "empty_text"),
    ({"k": 2}, 400, "missing_text"),
    ({"text": SAMPLE, "k": 0}, 400, "invalid_k"),
    ({"text": SAMPLE, "k": "abc"}, 400, "invalid_k"),
    ({"text": SAMPLE, "ratio": 0}, 400, "invalid_ratio"),
    ({"text": SAMPLE, "ratio": 150}, 400, "invalid_ratio"),
    ({"text": SAMPLE, "weighting": "abc"}, 400, "invalid_weighting"),
    ({"text": "!!! ... ???"}, 422, "no_valid_sentences"),
    ({"text": "a" * 100_001}, 413, "text_too_long"),
])
def test_summarize_errors(client, payload, status, code):
    res = client.post("/api/summarize", json=payload)
    assert res.status_code == status
    data = res.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == code
    assert data["error"]["message"]


def test_summarize_invalid_json(client):
    res = client.post("/api/summarize", data="not json", content_type="application/json")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "invalid_json"
    res = client.post("/api/summarize", json=[1, 2, 3])
    assert res.status_code == 400


def test_k_too_large_is_clamped(client):
    res = client.post("/api/summarize", json={"text": SAMPLE, "k": 50})
    assert res.status_code == 200
    st = res.get_json()["statistics"]
    assert st["k_used"] == st["matrix_rank"] < 50


def test_unknown_api_route_returns_json(client):
    res = client.get("/api/unknown")
    assert res.status_code == 404
    assert res.get_json()["success"] is False


def test_method_not_allowed(client):
    res = client.get("/api/summarize")
    assert res.status_code == 405
    assert res.get_json()["success"] is False


# ----- Upload -----
def _upload(client, content: bytes, name: str):
    return client.post("/api/upload", data={"file": (io.BytesIO(content), name)},
                       content_type="multipart/form-data")


def test_upload_ok(client):
    res = _upload(client, "\ufeffXin chào Việt Nam.".encode("utf-8"), "a.txt")
    assert res.status_code == 200
    assert res.get_json()["text"] == "Xin chào Việt Nam."


def test_upload_wrong_extension(client):
    assert _upload(client, b"abc", "a.pdf").status_code == 400


def test_upload_bad_encoding(client):
    res = _upload(client, "Tiếng Việt".encode("utf-16"), "a.txt")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "invalid_encoding"


def test_upload_too_large(client):
    res = _upload(client, b"a" * (2 * 1024 * 1024 + 10), "a.txt")
    assert res.status_code == 413

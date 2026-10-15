"""Kiểm thử thuật toán SVD/LSA trong summarizer.py."""

import math
import os
import sys
import unicodedata

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarizer import (SummarizerError, build_term_sentence_matrix,  # noqa: E402
                        compute_svd, lsa_sentence_scores,
                        number_of_sentences_to_select, preprocess,
                        rank_sentences, split_sentences, summarize)

SAMPLE = "AI phát triển công nghệ. AI ứng dụng trong giáo dục. Giáo dục sử dụng công nghệ AI."

TEN_SENTENCES = " ".join([
    "Trí tuệ nhân tạo đang phát triển rất nhanh.",
    "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.",
    "Dữ liệu lớn giúp mô hình học máy chính xác hơn.",
    "Giáo dục ứng dụng trí tuệ nhân tạo để cá nhân hóa việc học.",
    "Y tế sử dụng học máy để chẩn đoán bệnh.",
    "Nông nghiệp thông minh dùng cảm biến và dữ liệu.",
    "Giao thông thông minh giảm ùn tắc đô thị.",
    "Ngân hàng phát hiện gian lận nhờ học máy.",
    "Đạo đức dữ liệu là thách thức lớn.",
    "Pháp luật cần theo kịp sự phát triển công nghệ.",
])


# ---------- Test 1: văn bản mẫu ----------
def test_sample_text_full_pipeline():
    sentences = split_sentences(SAMPLE)
    assert len(sentences) == 3

    docs, _, _, _ = preprocess(sentences)
    A, vocab = build_term_sentence_matrix(docs, "bow")
    assert A.shape == (len(vocab), 3)            # m × n: hàng = từ, cột = câu
    assert "ai" in vocab
    ai_row = A[vocab.index("ai")]
    assert ai_row.tolist() == [1, 1, 1]          # "AI" xuất hiện ở cả 3 câu

    U, S, Vt, rank = compute_svd(A)
    assert rank >= 1

    result = summarize(SAMPLE, k=2, ratio=30)
    assert result["success"] is True
    assert result["statistics"]["original_sentences"] == 3
    assert result["statistics"]["selected_sentences"] == 1
    assert result["statistics"]["k_used"] == 2
    assert len(result["sentences"]) == 3
    assert all(s["score"] >= 0 for s in result["sentences"])
    assert sorted(s["rank"] for s in result["sentences"]) == [1, 2, 3]
    assert result["summary"] in sentences
    # Câu được chọn chính là câu hạng 1
    top = next(s for s in result["sentences"] if s["rank"] == 1)
    assert top["selected"] and result["summary"] == top["text"]


def test_scores_match_formula():
    """Điểm câu phải đúng công thức Steinberger–Ježek."""
    res = summarize(TEN_SENTENCES, k=3, ratio=30)
    sentences = split_sentences(TEN_SENTENCES)
    docs, _, _, _ = preprocess(sentences)
    A, _ = build_term_sentence_matrix(docs, "bow")
    _, S, Vt, _ = compute_svd(A)
    for j, s in enumerate(res["sentences"]):
        expected = math.sqrt(sum((S[i] ** 2) * (Vt[i, j] ** 2) for i in range(3)))
        assert s["score"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_summary_preserves_original_order():
    res = summarize(TEN_SENTENCES, k=3, ratio=50)
    ids = res["selected_indices"]
    assert ids == sorted(ids)
    sentences = split_sentences(TEN_SENTENCES)
    assert res["summary"] == " ".join(sentences[i - 1] for i in ids)
    # Các câu được chọn đúng là N câu có hạng cao nhất
    top_ranked = {s["id"] for s in res["sentences"] if s["rank"] <= len(ids)}
    assert top_ranked == set(ids)


def test_tfidf_weighting():
    res = summarize(SAMPLE, k=2, ratio=30, weighting="tfidf")
    assert res["matrix"]["weighting"] == "tfidf"
    assert all(0 <= v <= 1 for row in res["matrix"]["values"] for v in row)


# ---------- Test 2: một câu ----------
def test_single_sentence():
    res = summarize("Đây là một câu duy nhất về thuật toán SVD.", k=5, ratio=10)
    assert res["statistics"]["original_sentences"] == 1
    assert res["statistics"]["selected_sentences"] == 1
    assert res["statistics"]["k_used"] == 1
    assert res["summary"] == "Đây là một câu duy nhất về thuật toán SVD."


# ---------- Test 3: văn bản trống ----------
@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_text(text):
    with pytest.raises(SummarizerError) as exc:
        summarize(text)
    assert exc.value.status_code == 400
    assert "nhập văn bản" in exc.value.message


# ---------- Test 4: các câu giống nhau ----------
def test_identical_sentences():
    text = "Học máy rất thú vị. Học máy rất thú vị. Học máy rất thú vị."
    res = summarize(text, k=3, ratio=30)
    assert res["statistics"]["matrix_rank"] == 1
    assert res["statistics"]["k_used"] == 1
    scores = [s["score"] for s in res["sentences"]]
    assert all(np.isfinite(scores))
    # Điểm bằng nhau -> câu xuất hiện trước xếp hạng cao hơn
    assert [s["rank"] for s in res["sentences"]] == [1, 2, 3]
    assert res["selected_indices"] == [1]


def test_tie_break_prefers_earlier_sentence():
    assert rank_sentences(np.array([0.5, 0.9, 0.9, 0.1])) == [1, 2, 0, 3]
    # Sai số dấu phẩy động nhỏ vẫn được coi là bằng nhau
    assert rank_sentences(np.array([1.0 + 1e-15, 1.0])) == [0, 1]
    assert rank_sentences(np.array([1.0, 1.0 + 1e-15])) == [0, 1]


# ---------- Test 5: k vượt giới hạn ----------
def test_k_exceeds_rank():
    res = summarize(SAMPLE, k=100, ratio=30)
    rank = res["statistics"]["matrix_rank"]
    assert res["statistics"]["k_requested"] == 100
    assert res["statistics"]["k_used"] == rank
    assert any("vượt quá hạng" in w for w in res["warnings"])


@pytest.mark.parametrize("k", [0, -1, 1.5, "abc", None, True])
def test_invalid_k(k):
    with pytest.raises(SummarizerError) as exc:
        summarize(SAMPLE, k=k)
    assert exc.value.code == "invalid_k"


@pytest.mark.parametrize("ratio", [0, -10, 101, "x", None, float("nan")])
def test_invalid_ratio(ratio):
    with pytest.raises(SummarizerError) as exc:
        summarize(SAMPLE, ratio=ratio)
    assert exc.value.code == "invalid_ratio"


def test_invalid_weighting():
    with pytest.raises(SummarizerError):
        summarize(SAMPLE, weighting="word2vec")


# ---------- Test 6: tỷ lệ 10/20/30/50% ----------
@pytest.mark.parametrize("ratio,expected", [(10, 1), (20, 2), (30, 3), (50, 5)])
def test_ratios_ten_sentences(ratio, expected):
    res = summarize(TEN_SENTENCES, k=3, ratio=ratio)
    assert res["statistics"]["original_sentences"] == 10
    assert res["statistics"]["selected_sentences"] == expected
    assert len(res["selected_indices"]) == expected


@pytest.mark.parametrize("n,ratio,expected", [
    (3, 10, 1), (3, 20, 1), (3, 30, 1), (3, 50, 2),
    (7, 10, 1), (7, 20, 2), (7, 30, 3), (7, 50, 4),
    (1, 50, 1), (100, 30, 30), (10, 100, 10),
])
def test_number_of_sentences_rule(n, ratio, expected):
    assert number_of_sentences_to_select(n, ratio) == expected


# ---------- Test 7: Unicode tiếng Việt ----------
def test_vietnamese_unicode_preserved():
    # Chuỗi dạng NFD (tổ hợp) phải được chuẩn hóa NFC, không mất dấu
    nfd = unicodedata.normalize("NFD", "Người Việt yêu tiếng Việt. Tiếng Việt giàu đẹp và phong phú.")
    res = summarize(nfd, k=1, ratio=50)
    assert res["summary"] == unicodedata.normalize("NFC", res["summary"])
    for s in res["sentences"]:
        assert unicodedata.is_normalized("NFC", s["text"])
    joined = " ".join(s["text"] for s in res["sentences"])
    for word in ["Người", "Việt", "tiếng", "giàu", "đẹp", "phú"]:
        assert word in joined


def test_negation_words_kept():
    sentences = ["Sinh viên không đi học.", "Bài tập chưa hoàn thành."]
    docs, _, _, _ = preprocess(sentences)
    flat = " ".join(t for d in docs for t in d)
    assert "không" in flat and "chưa" in flat


def test_sentence_split_abbreviations_and_numbers():
    text = "TP. Hồ Chí Minh có GDP tăng 7.5% năm nay. PGS.TS. Nguyễn Văn A phát biểu! Số dân là 1.000.000 người?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].startswith("TP. Hồ Chí Minh")
    assert "1.000.000" in sents[2]


def test_multi_paragraph():
    text = "Đoạn một nói về AI\nĐoạn hai nói về dữ liệu.\n\nĐoạn ba nói về AI và dữ liệu."
    assert len(split_sentences(text)) == 3


# ---------- Test 8: không có từ hợp lệ ----------
@pytest.mark.parametrize("text", ["... !!! ???", "—— … ;;; ,,,", "🙂🙂 !!"])
def test_no_valid_words(text):
    with pytest.raises(SummarizerError) as exc:
        summarize(text)
    assert exc.value.status_code == 422


def test_too_long_text():
    with pytest.raises(SummarizerError) as exc:
        summarize("a" * 100_001)
    assert exc.value.status_code == 413


# ---------- Test 9: SVD ----------
def test_svd_reconstruction_and_order():
    sentences = split_sentences(TEN_SENTENCES)
    docs, _, _, _ = preprocess(sentences)
    for weighting in ("bow", "tfidf"):
        A, _ = build_term_sentence_matrix(docs, weighting)
        U, S, Vt, rank = compute_svd(A)
        m, n = A.shape
        r0 = min(m, n)
        assert U.shape == (m, r0) and S.shape == (r0,) and Vt.shape == (r0, n)
        # Tái tạo: A = U Σ Vᵀ
        assert np.allclose(U @ np.diag(S) @ Vt, A, atol=1e-10)
        # σ giảm dần và không âm
        assert np.all(np.diff(S) <= 1e-12) and np.all(S >= 0)
        # Trực chuẩn
        assert np.allclose(U.T @ U, np.eye(r0), atol=1e-10)
        assert np.allclose(Vt @ Vt.T, np.eye(r0), atol=1e-10)
        assert rank == np.linalg.matrix_rank(A)


def test_truncated_svd_full_k_equals_column_norm():
    """Với k = rank(A): Score(j) = ||A[:, j]||₂ (kiểm chứng toán học)."""
    sentences = split_sentences(TEN_SENTENCES)
    docs, _, _, _ = preprocess(sentences)
    A, _ = build_term_sentence_matrix(docs, "bow")
    _, S, Vt, rank = compute_svd(A)
    scores = lsa_sentence_scores(S, Vt, rank)
    assert np.allclose(scores, np.linalg.norm(A, axis=0), atol=1e-10)


def test_svd_rejects_empty_matrix():
    with pytest.raises(SummarizerError):
        compute_svd(np.zeros((0, 0)))
    with pytest.raises(SummarizerError):
        compute_svd(np.array([[np.nan, 1.0]]))

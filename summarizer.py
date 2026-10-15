"""
summarizer.py
=============

Tóm tắt văn bản tiếng Việt theo hướng trích xuất (Extractive Summarization)
bằng Singular Value Decomposition (SVD) kết hợp Latent Semantic Analysis (LSA).

Quy trình:
    1. Tiền xử lý: chuẩn hóa Unicode (NFC), loại khoảng trắng thừa, tách câu,
       tách từ tiếng Việt (underthesea -> PyVi -> regex dự phòng), chữ thường,
       bỏ dấu câu, bỏ stopwords (giữ lại từ phủ định).
    2. Xây dựng ma trận từ–câu A ∈ R^(m×n) (BoW hoặc TF-IDF).
       scikit-learn trả về ma trận câu×từ (n×m) nên cần CHUYỂN VỊ.
    3. Phân rã SVD:  A = U Σ Vᵀ   (numpy.linalg.svd, full_matrices=False).
    4. Truncated SVD: giữ k thành phần lớn nhất, 1 ≤ k ≤ rank(A).
    5. Điểm câu (Steinberger & Ježek, 2004):
            Score(j) = sqrt( Σ_{i=1..k} σ_i² · v_ij² )
    6. Chọn N = max(1, ceil(n·r/100)) câu điểm cao nhất, sắp xếp lại theo
       thứ tự xuất hiện trong văn bản gốc rồi ghép thành bản tóm tắt.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from fractions import Fraction
from functools import lru_cache
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------------------------------- #
# Cấu hình / giới hạn
# --------------------------------------------------------------------------- #
MAX_TEXT_CHARS = 100_000      # Giới hạn độ dài văn bản đầu vào (ký tự)
MAX_SENTENCES = 2_000         # Giới hạn số câu để SVD chạy nhanh
MATRIX_MAX_ROWS = 15          # Số hàng (từ) tối đa của ma trận hiển thị
MATRIX_MAX_COLS = 12          # Số cột (câu) tối đa của ma trận hiển thị
SCORE_TIE_DECIMALS = 12       # Hai điểm trùng nhau đến 12 chữ số thập phân
                              # được coi là bằng nhau (khử nhiễu dấu phẩy động)
VALID_WEIGHTINGS = ("bow", "tfidf")

# Từ viết tắt phổ biến: dấu "." sau các từ này KHÔNG phải ranh giới câu.
ABBREVIATIONS = {
    "tp", "ts", "ths", "pgs", "gs", "bs", "ks", "kts", "cn", "ncs", "th.s",
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc",
    "v.v", "vv", "tr", "nxb", "q", "p", "sđt", "đt", "no", "e.g", "i.e",
    "co", "ltd", "inc", "corp", "ft", "tt", "ubnd",
}

# Stopwords tiếng Việt (hư từ phổ biến). KHÔNG chứa từ phủ định như
# "không", "chưa", "chẳng", "đừng"... vì chúng đảo nghĩa câu.
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "các", "những", "được", "bị", "cho", "với", "trong",
    "thì", "mà", "này", "kia", "đó", "ấy", "nên", "nhưng", "hoặc", "vì",
    "do", "bởi", "để", "từ", "tại", "về", "theo", "như", "khi", "đã", "đang",
    "sẽ", "rất", "cũng", "vẫn", "lại", "ra", "vào", "một", "mỗi", "nào",
    "gì", "ở", "thế", "vậy", "nữa", "rằng", "còn", "đến", "tới", "qua",
    "trên", "dưới", "cùng", "nếu", "thế nhưng", "tuy nhiên", "ngoài ra",
    "bên cạnh đó", "do đó", "vì vậy", "vì thế", "trong khi", "mặc dù",
    "tuy", "song", "hay là", "chính là", "có thể", "đây", "kìa", "nhé",
    "à", "ạ", "ơi", "nhỉ", "thôi", "chứ", "luôn", "ngay", "hơn", "nhất",
}
NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đừng", "chớ",
                  "không thể", "không phải", "chưa từng"}
VIETNAMESE_STOPWORDS -= NEGATION_WORDS  # đảm bảo an toàn

# Biểu thức chính quy dùng chung
_HAS_WORD_RE = re.compile(r"[^\W_]", re.UNICODE)            # có chữ/số
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)  # dấu câu ở biên
_SYLLABLE_RE = re.compile(r"[^\W_]+", re.UNICODE)           # âm tiết/từ đơn
# Ranh giới câu: . ! ? … (có thể lặp) + dấu đóng ngoặc/nháy, theo sau là
# khoảng trắng hoặc kết thúc chuỗi. "3.14" hay "1.000.000" không bị tách.
_BOUNDARY_RE = re.compile(r"([.!?…]+)([\"'”’»)\]]*)(?=\s|$)")


class SummarizerError(ValueError):
    """Lỗi nghiệp vụ có kèm HTTP status code và mã lỗi để API trả về."""

    def __init__(self, message: str, status_code: int = 400,
                 code: str = "invalid_input") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


# --------------------------------------------------------------------------- #
# 1. TIỀN XỬ LÝ
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và loại bỏ khoảng trắng thừa (giữ xuống dòng)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)     # ký tự vô hình
    text = re.sub(r"[ \t\f\v\u00a0]+", " ", text)               # gộp khoảng trắng
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _ends_with_abbreviation(before: str) -> bool:
    """Kiểm tra đoạn văn bản (trước dấu chấm) có kết thúc bằng từ viết tắt."""
    tokens = before.split()
    if not tokens:
        return False
    last = tokens[-1].lower().strip("(\"'“‘[")
    if last in ABBREVIATIONS:
        return True
    if "." in last:                       # v.v / U.S / PGS.TS / T.Ư
        parts = [p for p in last.split(".") if p]
        if parts and (parts[-1] in ABBREVIATIONS
                      or (len(parts) >= 2 and all(len(p) == 1 for p in parts))):
            return True
    return False


def _split_paragraph(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(paragraph):
        punct = match.group(1)
        before = paragraph[start:match.start(1)]
        # Dấu chấm sau từ viết tắt -> không tách
        if punct == "." and _ends_with_abbreviation(before):
            continue
        # Dấu ba chấm mà câu tiếp theo viết thường -> vẫn cùng một câu
        if set(punct) <= {".", "…"} and (len(punct) > 1 or punct == "…"):
            nxt = paragraph[match.end():].lstrip()
            if nxt and nxt[0].islower():
                continue
        piece = paragraph[start:match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[str]:
    """
    Tách văn bản thành danh sách câu GỐC (giữ nguyên chữ hoa, dấu tiếng Việt,
    dấu câu) để dùng khi xuất bản tóm tắt. Mỗi dòng/đoạn được xử lý riêng.
    Câu không chứa chữ cái/chữ số nào (vd. "...", "!!!") bị loại.
    """
    text = normalize_text(text)
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(_split_paragraph(paragraph))
    return [s for s in sentences if _HAS_WORD_RE.search(s)]


def _regex_tokenize(sentence: str) -> list[str]:
    return _SYLLABLE_RE.findall(sentence)


@lru_cache(maxsize=1)
def _load_word_tokenizer() -> tuple[str, Callable[[str], list[str]]]:
    """
    Nạp bộ tách từ tiếng Việt theo thứ tự ưu tiên:
    underthesea -> PyVi -> regex (tách theo âm tiết, dự phòng).
    """
    try:
        from underthesea import word_tokenize  # type: ignore

        word_tokenize("Kiểm tra")  # chạy thử để bắt lỗi nạp mô hình sớm
        return "underthesea", lambda s: list(word_tokenize(s))
    except Exception:  # pragma: no cover - phụ thuộc môi trường
        pass
    try:
        from pyvi import ViTokenizer  # type: ignore

        return "pyvi", lambda s: ViTokenizer.tokenize(s).split()
    except Exception:  # pragma: no cover - phụ thuộc môi trường
        pass
    return "regex", _regex_tokenize


def get_tokenizer_name() -> str:
    return _load_word_tokenizer()[0]


def tokenize_sentence(sentence: str,
                      tokenizer: Callable[[str], list[str]]) -> list[str]:
    """Tách từ một câu, bỏ dấu câu/ký tự đặc biệt, giữ nguyên chữ hoa/thường."""
    tokens: list[str] = []
    for raw in tokenizer(sentence):
        tok = raw.replace("_", " ")                 # PyVi nối từ ghép bằng "_"
        tok = _EDGE_PUNCT_RE.sub("", tok)
        tok = re.sub(r"\s+", " ", tok).strip()
        if tok and _HAS_WORD_RE.search(tok):
            tokens.append(tok)
    return tokens


def preprocess(sentences: list[str], remove_stopwords: bool = True
               ) -> tuple[list[list[str]], dict[str, str], str, list[str]]:
    """
    Trả về:
        docs      : danh sách token (chữ thường) cho từng câu
        display   : ánh xạ term -> dạng hiển thị (giữ "AI", "SVD" viết hoa)
        tokenizer : tên bộ tách từ đã dùng
        warnings  : cảnh báo phát sinh
    """
    name, tokenizer = _load_word_tokenizer()
    warnings: list[str] = []
    surface_docs: list[list[str]] = []
    for sentence in sentences:
        try:
            surface_docs.append(tokenize_sentence(sentence, tokenizer))
        except Exception:  # Lỗi thư viện NLP -> dùng regex cho câu này
            surface_docs.append(tokenize_sentence(sentence, _regex_tokenize))
            msg = f"Bộ tách từ '{name}' lỗi ở một số câu, đã dùng tách âm tiết dự phòng."
            if msg not in warnings:
                warnings.append(msg)

    display: dict[str, str] = {}
    lower_docs: list[list[str]] = []
    for doc in surface_docs:
        lowered = []
        for tok in doc:
            term = tok.lower()
            # Từ viết hoa toàn bộ (AI, SVD, LSA) giữ nguyên khi hiển thị
            if term not in display or (tok.isupper() and len(tok) > 1):
                display[term] = tok if (tok.isupper() and len(tok) > 1) else term
            lowered.append(term)
        lower_docs.append(lowered)

    if remove_stopwords:
        filtered = [[t for t in d if t not in VIETNAMESE_STOPWORDS] for d in lower_docs]
        if any(filtered):
            return filtered, display, name, warnings
        if any(lower_docs):
            warnings.append("Văn bản chỉ gồm hư từ; đã giữ lại stopwords để có thể phân tích.")
    return lower_docs, display, name, warnings


# --------------------------------------------------------------------------- #
# 2. MA TRẬN TỪ–CÂU
# --------------------------------------------------------------------------- #
def _identity(doc):
    """Analyzer cho scikit-learn: dữ liệu đã được tách từ sẵn."""
    return doc


def build_term_sentence_matrix(docs: list[list[str]], weighting: str = "bow"
                               ) -> tuple[np.ndarray, list[str]]:
    """
    Xây dựng A ∈ R^(m×n): hàng = từ, cột = câu.
    scikit-learn tạo ma trận câu×từ (n×m) => chuyển vị .T để đúng lý thuyết.
    """
    if weighting == "tfidf":
        vectorizer = TfidfVectorizer(analyzer=_identity, lowercase=False,
                                     norm="l2", smooth_idf=True)
    else:
        vectorizer = CountVectorizer(analyzer=_identity, lowercase=False)
    try:
        sentence_term = vectorizer.fit_transform(docs)          # n × m
    except ValueError as exc:  # empty vocabulary
        raise SummarizerError("Không có từ hợp lệ nào sau tiền xử lý để xây dựng ma trận.",
                              422, "empty_vocabulary") from exc
    A = sentence_term.T.toarray().astype(float)                  # m × n
    vocab = vectorizer.get_feature_names_out().tolist()
    return A, vocab


# --------------------------------------------------------------------------- #
# 3–5. SVD VÀ ĐIỂM LSA
# --------------------------------------------------------------------------- #
def compute_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Phân rã A = U Σ Vᵀ (reduced SVD). Trả về U (m×r0), S (r0), Vt (r0×n), rank.
    r0 = min(m, n). Giá trị suy biến được đảm bảo giảm dần.
    rank(A) = số σ > tol, tol = σ_max · max(m, n) · eps (giống numpy.matrix_rank).
    """
    if A.ndim != 2 or A.size == 0:
        raise SummarizerError("Ma trận từ–câu rỗng, không đủ dữ liệu để phân rã SVD.",
                              422, "empty_matrix")
    if not np.all(np.isfinite(A)):
        raise SummarizerError("Ma trận chứa giá trị không hợp lệ (NaN/Inf).",
                              422, "invalid_matrix")
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SummarizerError("Phân rã SVD không hội tụ.", 500, "svd_failed") from exc

    order = np.argsort(-S, kind="stable")          # phòng thủ: đảm bảo giảm dần
    U, S, Vt = U[:, order], S[order], Vt[order, :]
    tol = (S.max() * max(A.shape) * np.finfo(float).eps) if S.size else 0.0
    rank = int(np.sum(S > tol))
    return U, S, Vt, rank


def lsa_sentence_scores(S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Steinberger–Ježek:  Score(j) = sqrt( Σ_{i=1..k} σ_i² · v_ij² )
    Vt[i, j] là thành phần của câu j trên chủ đề ẩn i. Bình phương nên dấu
    (tùy ý) của vector suy biến không ảnh hưởng; điểm luôn ≥ 0.
    """
    sigma_k = S[:k, None]                      # k × 1
    v_k = Vt[:k, :]                            # k × n
    return np.sqrt(np.sum((sigma_k ** 2) * (v_k ** 2), axis=0))


def rank_sentences(scores: np.ndarray) -> list[int]:
    """
    Trả về chỉ số câu theo thứ tự hạng (điểm giảm dần).
    Điểm bằng nhau (tới SCORE_TIE_DECIMALS chữ số) -> câu xuất hiện trước xếp trên.
    """
    return sorted(range(len(scores)),
                  key=lambda j: (-round(float(scores[j]), SCORE_TIE_DECIMALS), j))


def number_of_sentences_to_select(n: int, ratio: float) -> int:
    """N = max(1, ceil(n · r / 100)), không vượt quá n. Dùng Fraction để tránh sai số."""
    exact = Fraction(str(ratio)) * n / 100
    return max(1, min(n, math.ceil(exact)))


# --------------------------------------------------------------------------- #
# Kiểm tra tham số
# --------------------------------------------------------------------------- #
def _parse_k(k) -> int:
    err = SummarizerError("Số thành phần k phải là số nguyên lớn hơn hoặc bằng 1.",
                          400, "invalid_k")
    if isinstance(k, bool) or k is None:
        raise err
    if isinstance(k, str):
        if not re.fullmatch(r"\s*\d+\s*", k):
            raise err
        k = int(k)
    if isinstance(k, float):
        if not k.is_integer():
            raise err
        k = int(k)
    if not isinstance(k, int) or k < 1:
        raise err
    return k


def _parse_ratio(ratio) -> float:
    err = SummarizerError("Tỷ lệ tóm tắt phải là số trong khoảng (0, 100].",
                          400, "invalid_ratio")
    if isinstance(ratio, bool) or ratio is None:
        raise err
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        raise err from None
    if not math.isfinite(value) or value <= 0 or value > 100:
        raise err
    return int(value) if value.is_integer() else value


def _matrix_preview(A: np.ndarray, vocab: list[str], display: dict[str, str],
                    weighting: str) -> dict:
    """Ma trận thu nhỏ để hiển thị: ưu tiên từ xuất hiện ở nhiều câu nhất."""
    m, n = A.shape
    df = (A > 0).sum(axis=1)
    total = A.sum(axis=1)
    first = [int(np.argmax(A[i] > 0)) for i in range(m)]
    row_order = sorted(range(m), key=lambda i: (-df[i], -total[i], first[i], vocab[i]))
    rows = row_order[:MATRIX_MAX_ROWS]
    cols = list(range(min(n, MATRIX_MAX_COLS)))
    values = []
    for i in rows:
        if weighting == "bow":
            values.append([int(round(A[i, j])) for j in cols])
        else:
            values.append([round(float(A[i, j]), 4) for j in cols])
    return {
        "weighting": weighting,
        "words": [display.get(vocab[i], vocab[i]) for i in rows],
        "columns": [f"C{j + 1}" for j in cols],
        "values": values,
        "full_shape": [m, n],
        "shown_shape": [len(rows), len(cols)],
        "truncated": m > len(rows) or n > len(cols),
    }


# --------------------------------------------------------------------------- #
# HÀM CHÍNH
# --------------------------------------------------------------------------- #
def summarize(text, k=2, ratio=30, weighting: str = "bow",
              remove_stopwords: bool = True) -> dict:
    """Tóm tắt văn bản; trả về dict sẵn sàng serialize JSON."""
    started = time.perf_counter()

    # ---- Kiểm tra đầu vào ----
    if not isinstance(text, str) or not text.strip():
        raise SummarizerError("Vui lòng nhập văn bản cần tóm tắt.", 400, "empty_text")
    if len(text) > MAX_TEXT_CHARS:
        raise SummarizerError(
            f"Văn bản quá dài ({len(text):,} ký tự). Giới hạn là {MAX_TEXT_CHARS:,} ký tự.",
            413, "text_too_long")
    k_requested = _parse_k(k)
    ratio = _parse_ratio(ratio)
    weighting = (weighting or "bow").lower() if isinstance(weighting, str) else weighting
    if weighting not in VALID_WEIGHTINGS:
        raise SummarizerError("Phương pháp biểu diễn phải là 'bow' hoặc 'tfidf'.",
                              400, "invalid_weighting")

    # ---- Bước 1: Tiền xử lý ----
    sentences = split_sentences(text)
    if not sentences:
        raise SummarizerError(
            "Văn bản không chứa câu hợp lệ (chỉ gồm dấu câu hoặc ký tự đặc biệt).",
            422, "no_valid_sentences")
    if len(sentences) > MAX_SENTENCES:
        raise SummarizerError(
            f"Văn bản có {len(sentences)} câu, vượt giới hạn {MAX_SENTENCES} câu.",
            413, "too_many_sentences")
    docs, display, tokenizer_name, warnings = preprocess(sentences, remove_stopwords)
    if not any(docs):
        raise SummarizerError("Không có từ hợp lệ nào sau tiền xử lý.", 422, "empty_vocabulary")

    # ---- Bước 2: Ma trận từ–câu A (m × n) ----
    A, vocab = build_term_sentence_matrix(docs, weighting)
    m, n = A.shape

    # ---- Bước 3: SVD ----
    U, S, Vt, rank = compute_svd(A)
    if rank == 0:
        raise SummarizerError("Ma trận có hạng bằng 0, không đủ thông tin để tóm tắt.",
                              422, "zero_rank")

    # ---- Bước 4: Truncated SVD — giới hạn 1 ≤ k ≤ rank(A) ----
    k_used = min(k_requested, rank)
    if k_used != k_requested:
        warnings.append(f"k = {k_requested} vượt quá hạng của ma trận (rank = {rank}). "
                        f"Hệ thống đã tự động dùng k = {k_used}.")

    # ---- Bước 5: Điểm LSA ----
    scores = lsa_sentence_scores(S, Vt, k_used)

    # ---- Bước 6: Chọn câu ----
    order = rank_sentences(scores)
    n_select = number_of_sentences_to_select(n, ratio)
    selected = sorted(order[:n_select])          # trả về thứ tự gốc trong văn bản
    selected_set = set(selected)
    rank_of = {j: r + 1 for r, j in enumerate(order)}
    summary = " ".join(sentences[j] for j in selected)

    # ---- Bước 7: Thống kê ----
    energy_total = float(np.sum(S ** 2))
    energy_kept = float(np.sum(S[:k_used] ** 2)) / energy_total * 100 if energy_total else 0.0
    original_words = sum(len(_SYLLABLE_RE.findall(s)) for s in sentences)
    summary_words = sum(len(_SYLLABLE_RE.findall(sentences[j])) for j in selected)

    return {
        "success": True,
        "summary": summary,
        "config": {
            "k_requested": k_requested,
            "k_used": k_used,
            "ratio": ratio,
            "weighting": weighting,
            "remove_stopwords": bool(remove_stopwords),
        },
        "statistics": {
            "original_sentences": n,
            "selected_sentences": n_select,
            "actual_ratio": round(n_select / n * 100, 2),
            "requested_ratio": ratio,
            "unique_terms": m,
            "k_requested": k_requested,
            "k_used": k_used,
            "matrix_rank": rank,
            "matrix_shape": [m, n],
            "energy_retained": round(energy_kept, 2),
            "original_words": original_words,
            "summary_words": summary_words,
            "original_chars": sum(len(s) for s in sentences),
            "summary_chars": len(summary),
        },
        "sentences": [
            {
                "id": j + 1,
                "text": sentences[j],
                "score": float(scores[j]),
                "rank": rank_of[j],
                "selected": j in selected_set,
                "terms": docs[j],
            }
            for j in range(n)
        ],
        "selected_indices": [j + 1 for j in selected],
        "singular_values": [float(s) for s in S],
        "matrix": _matrix_preview(A, vocab, display, weighting),
        "tokenizer": tokenizer_name,
        "warnings": warnings,
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
    }

# SVD Text Summarizer

**Đề tài:** Xây dựng hệ thống tóm tắt văn bản tiếng Việt sử dụng thuật toán
**Singular Value Decomposition (SVD)** kết hợp **Latent Semantic Analysis (LSA)**.

Hệ thống tóm tắt theo hướng **trích xuất** (extractive): chọn ra các câu quan trọng nhất
trong văn bản gốc, giữ nguyên nội dung câu, không dùng AI tạo sinh và không cần API key.

---

## 1. Cài đặt và chạy (Windows / VS Code / PowerShell)

Yêu cầu: **Python 3.11+**.

```powershell
cd svd-text-summarizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Mở trình duyệt: <http://127.0.0.1:5000>

> Nếu PowerShell chặn script kích hoạt venv, chạy một lần:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Chạy kiểm thử:

```powershell
python -m pytest -v
```

> Chart.js và font Be Vietnam Pro được tải từ CDN — cần Internet để biểu đồ hiển thị.
> Thuật toán (backend) chạy hoàn toàn offline.

---

## 2. Cấu trúc dự án

```text
svd-text-summarizer/
├── app.py                 # Flask: routes, kiểm tra đầu vào, trả JSON
├── summarizer.py          # Tiền xử lý + ma trận từ–câu + SVD + LSA + chọn câu
├── requirements.txt
├── README.md
├── .gitignore
├── templates/index.html   # Giao diện
├── static/css/style.css   # Giao diện, responsive
├── static/js/main.js      # Gọi API, vẽ bảng & biểu đồ (Chart.js)
└── tests/
    ├── test_summarizer.py # Kiểm thử thuật toán
    └── test_api.py        # Kiểm thử API
```

---

## 3. Thuật toán

| Bước | Nội dung |
|---|---|
| 1. Tiền xử lý | Chuẩn hóa Unicode NFC, gộp khoảng trắng, tách câu (xử lý viết tắt `TP.`, `PGS.TS.`, số `1.000.000`), tách từ bằng **underthesea** (dự phòng PyVi → regex), chữ thường, bỏ dấu câu, bỏ stopwords (**giữ** từ phủ định *không, chưa, chẳng…*). |
| 2. Ma trận | `A ∈ ℝ^(m×n)`, hàng = từ, cột = câu. `CountVectorizer` (BoW) hoặc `TfidfVectorizer`. scikit-learn trả về câu×từ nên **chuyển vị**. |
| 3. SVD | `U, Σ, Vᵀ = numpy.linalg.svd(A, full_matrices=False)`; σ giảm dần; `rank(A)` = số σ > tol. |
| 4. Truncated SVD | Giữ k thành phần: `A_k = U_k Σ_k V_kᵀ`, `1 ≤ k ≤ rank(A)`. Nếu k > rank, tự giảm về rank và báo cho frontend (`k_used`, `warnings`). |
| 5. Điểm LSA | Steinberger–Ježek: `Score(j) = √(Σ_{i=1..k} σᵢ² · v_ij²)` — luôn ≥ 0. |
| 6. Chọn câu | `N = max(1, ceil(n·r/100))`; xếp hạng theo điểm giảm dần (bằng điểm → câu đứng trước ưu tiên); lấy N câu đầu, **sắp lại theo thứ tự gốc** rồi ghép. |
| 7. Thống kê | số câu, số câu giữ lại, tỷ lệ thực tế, số từ khác nhau, k, rank, σ, năng lượng giữ lại `Σσᵢ²(i≤k)/Σσᵢ²`, ma trận thu nhỏ. |

**Ý nghĩa:** mỗi hàng của `Vᵀ` là một "chủ đề ẩn"; `σᵢ` đo độ quan trọng của chủ đề i;
`|v_ij|` đo mức câu j thể hiện chủ đề i. Câu có điểm cao là câu thể hiện mạnh các chủ đề chính.
Khi `k = rank(A)`, `Score(j) = ‖A[:, j]‖₂` (độ dài vector câu) — đã được kiểm chứng trong test.

---

## 4. API

`POST /api/summarize`

```json
{ "text": "AI phát triển công nghệ. AI ứng dụng trong giáo dục. Giáo dục sử dụng công nghệ AI.",
  "k": 2, "ratio": 30, "weighting": "bow", "remove_stopwords": true }
```

Phản hồi (rút gọn): `success`, `summary`, `config`, `statistics`
(`original_sentences`, `selected_sentences`, `actual_ratio`, `unique_terms`, `k_requested`,
`k_used`, `matrix_rank`, `energy_retained`…), `sentences[]` (`id`, `text`, `score`, `rank`,
`selected`), `selected_indices`, `singular_values`, `matrix` (`words`, `columns`, `values`,
`truncated`), `tokenizer`, `warnings`.

Lỗi: `{"success": false, "error": {"code": "...", "message": "..."}}` với mã HTTP
400 (đầu vào sai), 413 (quá dài), 422 (không có từ/câu hợp lệ), 500 (lỗi SVD / máy chủ).

Khác: `GET /api/health`, `POST /api/upload` (multipart, `.txt` UTF-8 ≤ 2 MB).
Giao diện đọc tệp TXT trực tiếp trên trình duyệt (FileReader/TextDecoder UTF-8).

---

## 5. Giả định & giới hạn

- Giới hạn 100.000 ký tự và 2.000 câu/văn bản; tệp TXT ≤ 2 MB, bắt buộc UTF-8.
- Tách câu dựa trên quy tắc + danh sách viết tắt; có thể sai với văn bản rất đặc thù.
- Lần chạy đầu underthesea nạp mô hình nên chậm hơn vài giây.
- Ma trận hiển thị tối đa 15 từ (xuất hiện ở nhiều câu nhất) × 12 câu.
- Tóm tắt trích xuất nên các câu được giữ nguyên; tính liên kết giữa câu phụ thuộc văn bản gốc.

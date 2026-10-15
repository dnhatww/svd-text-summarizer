/* ==========================================================================
   SVD Text Summarizer — Frontend logic (Vanilla JS ES6+)
   Mọi số liệu hiển thị đều lấy từ phản hồi của POST /api/summarize.
   ========================================================================== */
(() => {
  "use strict";

  const MAX_FILE_BYTES = 2 * 1024 * 1024;
  const COLORS = {
    navy: "#102A43", blue: "#2589D8", green: "#1E9E61",
    orange: "#F59E0B", gray: "#CBD5E1", border: "#D6DEE8", muted: "#64748B",
  };
  const LONG_SAMPLE =
`Trí tuệ nhân tạo (AI) đang thay đổi mạnh mẽ nhiều lĩnh vực của đời sống xã hội. Trong giáo dục, AI giúp cá nhân hóa lộ trình học tập cho từng học sinh. Các hệ thống học tập thông minh có thể phân tích dữ liệu học tập và đề xuất nội dung phù hợp. Giáo viên sử dụng công cụ AI để chấm bài và theo dõi tiến độ của học sinh nhanh hơn.
Trong y tế, công nghệ AI hỗ trợ bác sĩ chẩn đoán bệnh qua hình ảnh y khoa. Nhiều bệnh viện tại Việt Nam đã thử nghiệm phần mềm AI để phát hiện sớm ung thư. Tuy nhiên, việc ứng dụng AI vẫn chưa thể thay thế hoàn toàn vai trò của bác sĩ.
Bên cạnh lợi ích, AI cũng đặt ra nhiều thách thức về đạo đức và quyền riêng tư dữ liệu. Dữ liệu cá nhân nếu không được bảo vệ có thể bị khai thác sai mục đích. Vì vậy, các quốc gia cần xây dựng khung pháp lý rõ ràng cho việc phát triển AI. Phát triển AI có trách nhiệm là xu hướng tất yếu trong kỷ nguyên số.`;

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const el = {
    text: $("inputText"), charCount: $("charCount"),
    fileInput: $("fileInput"), uploadBtn: $("uploadBtn"), fileName: $("fileName"),
    clearBtn: $("clearBtn"), sampleBtn: $("sampleBtn"),
    k: $("kInput"), kMinus: $("kMinus"), kPlus: $("kPlus"), kHint: $("kHint"),
    weighting: $("weighting"), stopwords: $("stopwords"),
    ratioBtns: document.querySelectorAll(".ratio-btn"),
    submit: $("summarizeBtn"),
    summaryBody: $("summaryBody"), configBadge: $("configBadge"),
    staleBanner: $("staleBanner"), warnList: $("warnList"),
    copyBtn: $("copyBtn"), downloadBtn: $("downloadBtn"),
    statInput: $("statInput"), statKept: $("statKept"), statRatio: $("statRatio"),
    miniStats: $("miniStats"),
    scoreBody: $("scoreBody"),
    matrixTable: $("matrixTable"), matrixNote: $("matrixNote"), matrixTag: $("matrixTag"),
    sigmaTag: $("sigmaTag"), sigmaNote: $("sigmaNote"), sigmaEmpty: $("sigmaEmpty"),
    scoreEmpty: $("scoreEmpty"),
    statusPill: $("statusPill"), statusText: $("statusText"),
    tokenizerName: $("tokenizerName"), toastStack: $("toastStack"),
  };

  // ---------- State ----------
  const state = {
    ratio: 30,
    loading: false,
    result: null,          // phản hồi gần nhất từ backend
    resultConfig: null,    // cấu hình (text, k, ratio, weighting, stopwords) ứng với result
    lastRank: null,        // rank(A) của lần phân tích gần nhất
    charts: { sigma: null, score: null },
  };

  // ---------- Utilities ----------
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 4) => Number(x).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtPct = (x) => `${Number(x).toLocaleString("vi-VN", { maximumFractionDigits: 2 })}%`;

  function toast(message, type = "info", ms = 3200) {
    const t = document.createElement("div");
    t.className = `toast ${type}`;
    t.textContent = message;
    el.toastStack.appendChild(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 260); }, ms);
  }

  function setStatus(stateName, text) {
    el.statusPill.dataset.state = stateName;
    el.statusText.textContent = text;
  }

  function currentConfig() {
    return {
      text: el.text.value,
      k: el.k.value.trim(),
      ratio: state.ratio,
      weighting: el.weighting.value,
      remove_stopwords: el.stopwords.checked,
    };
  }

  // ---------- Input handling ----------
  function updateCharCount() {
    const n = el.text.value.length;
    el.charCount.textContent = `${n.toLocaleString("vi-VN")} ký tự`;
  }

  /** Kiểm tra k phía client; backend vẫn kiểm tra lại và tự điều chỉnh nếu k > rank(A). */
  function validateK() {
    const raw = el.k.value.trim();
    const v = Number(raw);
    const ok = raw !== "" && Number.isInteger(v) && v >= 1;
    el.k.classList.toggle("invalid", !ok);
    el.kHint.classList.remove("warn");
    if (!ok) {
      el.kHint.textContent = "k phải là số nguyên ≥ 1";
      el.kHint.classList.add("warn");
    } else if (state.lastRank !== null && v > state.lastRank && isSameText()) {
      el.kHint.textContent = `k > rank(A) = ${state.lastRank} → sẽ dùng k = ${state.lastRank}`;
      el.kHint.classList.add("warn");
    } else if (state.lastRank !== null && isSameText()) {
      el.kHint.textContent = `1 ≤ k ≤ rank(A) = ${state.lastRank}`;
    } else {
      el.kHint.textContent = "1 ≤ k ≤ rank(A)";
    }
    return ok;
  }

  const isSameText = () => state.resultConfig && state.resultConfig.text === el.text.value;

  /** Đánh dấu kết quả hiện tại "cũ" nếu cấu hình khác với lần phân tích gần nhất. */
  function checkStale() {
    if (!state.result || state.loading) return;
    const now = currentConfig();
    const prev = state.resultConfig;
    const stale = ["text", "k", "ratio", "weighting", "remove_stopwords"]
      .some((key) => String(now[key]) !== String(prev[key]));
    el.staleBanner.hidden = !stale;
    el.summaryBody.classList.toggle("stale", stale);
    if (stale) setStatus("stale", "Cấu hình đã thay đổi");
    else setStatus("ok", "Phân tích thành công");
  }

  function onConfigChange() { validateK(); checkStale(); }

  el.text.addEventListener("input", () => { updateCharCount(); onConfigChange(); });
  el.k.addEventListener("input", onConfigChange);
  el.weighting.addEventListener("change", onConfigChange);
  el.stopwords.addEventListener("change", onConfigChange);

  const stepK = (delta) => {
    const v = parseInt(el.k.value, 10);
    el.k.value = Math.max(1, (Number.isFinite(v) ? v : 1) + delta);
    onConfigChange();
  };
  el.kMinus.addEventListener("click", () => stepK(-1));
  el.kPlus.addEventListener("click", () => stepK(1));

  el.ratioBtns.forEach((btn) => btn.addEventListener("click", () => {
    el.ratioBtns.forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-checked", "false"); });
    btn.classList.add("active");
    btn.setAttribute("aria-checked", "true");
    state.ratio = Number(btn.dataset.ratio);
    onConfigChange();
  }));

  el.clearBtn.addEventListener("click", () => {
    el.text.value = ""; el.fileName.hidden = true;
    updateCharCount(); onConfigChange(); el.text.focus();
  });
  el.sampleBtn.addEventListener("click", () => {
    el.text.value = LONG_SAMPLE; el.fileName.hidden = true;
    updateCharCount(); onConfigChange();
  });

  // ---------- TXT upload (đọc trực tiếp trên trình duyệt, UTF-8) ----------
  el.uploadBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", async () => {
    const file = el.fileInput.files[0];
    el.fileInput.value = "";               // cho phép chọn lại cùng một tệp
    if (!file) return;
    if (!/\.txt$/i.test(file.name)) { toast("Chỉ chấp nhận tệp .txt.", "error"); return; }
    if (file.size === 0) { toast("Tệp rỗng.", "error"); return; }
    if (file.size > MAX_FILE_BYTES) { toast("Tệp vượt quá 2 MB.", "error"); return; }
    try {
      const buf = await file.arrayBuffer();
      // fatal: true -> ném lỗi nếu tệp không phải UTF-8 hợp lệ
      let content = new TextDecoder("utf-8", { fatal: true }).decode(buf);
      content = content.replace(/^\uFEFF/, "");
      if (!content.trim()) { toast("Tệp không có nội dung.", "error"); return; }
      const max = Number(el.text.getAttribute("maxlength")) || Infinity;
      if (content.length > max) {
        toast(`Tệp có ${content.length.toLocaleString("vi-VN")} ký tự, vượt giới hạn ${max.toLocaleString("vi-VN")}.`, "error", 4500);
        return;
      }
      el.text.value = content.normalize("NFC");
      el.fileName.textContent = `✓ Đã tải: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      el.fileName.hidden = false;
      updateCharCount(); onConfigChange();
      toast("Đã tải nội dung tệp vào ô văn bản.", "success");
    } catch (e) {
      toast("Không đọc được tệp. Hãy đảm bảo tệp được mã hóa UTF-8.", "error", 4500);
    }
  });

  // ---------- Summarize ----------
  function setLoading(on) {
    state.loading = on;
    el.submit.disabled = on;
    el.submit.classList.toggle("loading", on);
    el.submit.querySelector(".btn-label").textContent = on ? "Đang phân tích..." : "Tóm tắt văn bản";
    if (on) setStatus("loading", "Đang phân tích");
  }

  async function summarize() {
    if (state.loading) return;
    if (!el.text.value.trim()) {
      showError("Vui lòng nhập văn bản cần tóm tắt.");
      el.text.focus();
      return;
    }
    if (!validateK()) { toast("Số thành phần k không hợp lệ.", "error"); el.k.focus(); return; }

    const cfg = currentConfig();
    setLoading(true);
    try {
      const res = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...cfg, k: Number(cfg.k) }),
      });
      let data;
      try { data = await res.json(); } catch { throw new Error(`Máy chủ trả về phản hồi không hợp lệ (HTTP ${res.status}).`); }
      if (!res.ok || !data.success) throw new Error(data?.error?.message || `Lỗi HTTP ${res.status}`);

      state.result = data;
      state.resultConfig = cfg;
      state.lastRank = data.statistics.matrix_rank;
      renderAll(data);
      setLoading(false);
      validateK();
      checkStale();
      if (data.warnings?.length) data.warnings.forEach((w) => toast(w, "warning", 5000));
      else toast("Tóm tắt thành công.", "success", 2000);
    } catch (err) {
      setLoading(false);
      const msg = err instanceof TypeError ? "Không kết nối được máy chủ Flask. Hãy kiểm tra server đang chạy." : err.message;
      showError(msg);
    }
  }
  el.submit.addEventListener("click", summarize);
  el.text.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") summarize();
  });

  function showError(message) {
    setStatus("error", "Có lỗi");
    toast(message, "error", 5000);
    if (!state.result) {
      el.summaryBody.innerHTML = `
        <div class="empty-state error-state">
          <div class="empty-icon">!</div>
          <p>Không thể tóm tắt</p>
          <small>${escapeHtml(message)}</small>
        </div>`;
    }
  }

  // ---------- Rendering ----------
  function renderAll(data) {
    renderSummary(data);
    renderStats(data.statistics);
    renderScoreTable(data.sentences);
    renderMatrix(data.matrix);
    renderSigmaChart(data.singular_values, data.statistics.k_used);
    renderScoreChart(data.sentences);
    el.tokenizerName.textContent = data.tokenizer;
  }

  function renderSummary(data) {
    const selected = data.sentences.filter((s) => s.selected);
    el.summaryBody.innerHTML = `<p class="summary-text">${selected.map((s, i) =>
      `<span class="summary-sentence" style="animation-delay:${i * 70}ms" title="Câu ${s.id} · điểm ${fmt(s.score)} · hạng ${s.rank}"><sup>[C${s.id}]</sup>${escapeHtml(s.text)}</span>`
    ).join(" ")}</p>`;
    const c = data.config;
    el.configBadge.innerHTML = [
      `k = ${c.k_used}${c.k_used !== c.k_requested ? ` (yêu cầu ${c.k_requested})` : ""}`,
      `Tỷ lệ ${c.ratio}%`,
      c.weighting === "tfidf" ? "TF-IDF" : "Bag of Words",
      c.remove_stopwords ? "Bỏ stopwords" : "Giữ stopwords",
      `${data.processing_ms} ms`,
    ].map((t) => `<span>${escapeHtml(t)}</span>`).join("");
    el.configBadge.hidden = false;
    el.warnList.innerHTML = (data.warnings || []).map((w) => `<div>⚠ ${escapeHtml(w)}</div>`).join("");
    el.warnList.hidden = !(data.warnings || []).length;
    el.copyBtn.disabled = el.downloadBtn.disabled = !data.summary;
  }

  function bump(node, value) {
    node.textContent = value;
    node.classList.remove("bump"); void node.offsetWidth; node.classList.add("bump");
  }

  function renderStats(st) {
    bump(el.statInput, st.original_sentences);
    bump(el.statKept, st.selected_sentences);
    bump(el.statRatio, fmtPct(st.actual_ratio));
    el.miniStats.innerHTML = `
      <span>Số từ khác nhau (m): <b>${st.unique_terms}</b></span>
      <span>Ma trận A: <b>${st.matrix_shape[0]}×${st.matrix_shape[1]}</b></span>
      <span>rank(A): <b>${st.matrix_rank}</b></span>
      <span>k sử dụng: <b>${st.k_used}</b></span>
      <span>Năng lượng giữ lại: <b>${fmtPct(st.energy_retained)}</b></span>
      <span>Số tiếng: <b>${st.original_words} → ${st.summary_words}</b></span>`;
    el.miniStats.hidden = false;
  }

  function renderScoreTable(sentences) {
    el.scoreBody.innerHTML = sentences.map((s) => `
      <tr class="${s.selected ? "selected" : ""}" title="${escapeHtml(s.text)}">
        <td>Câu ${s.id}</td>
        <td class="num" title="${s.score}">${fmt(s.score)}</td>
        <td><span class="rank-badge">${s.rank}</span></td>
      </tr>`).join("");
  }

  function renderMatrix(matrix) {
    const isBow = matrix.weighting === "bow";
    el.matrixTag.textContent = isBow ? "BoW" : "TF-IDF";
    const head = `<tr><th>Từ</th>${matrix.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
    // Màu nền ô tỷ lệ với giá trị (giá trị > 0 được tô xanh dương)
    const maxVal = Math.max(1e-9, ...matrix.values.flat());
    const body = matrix.values.map((row, i) => `<tr><td>${escapeHtml(matrix.words[i])}</td>${row.map((v) => {
      if (v <= 0) return `<td class="cell">0</td>`;
      const a = (0.15 + 0.6 * (v / maxVal)).toFixed(2);
      return `<td class="cell pos" style="background:rgba(37,137,216,${a});${a > 0.5 ? "color:#fff" : ""}">${isBow ? v : fmt(v, 3)}</td>`;
    }).join("")}</tr>`).join("");
    el.matrixTable.innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
    const [m, n] = matrix.full_shape;
    el.matrixNote.textContent = matrix.truncated
      ? `Đã thu nhỏ: hiển thị ${matrix.shown_shape[0]}/${m} từ (xuất hiện nhiều nhất) × ${matrix.shown_shape[1]}/${n} câu.`
      : `Ma trận đầy đủ A ∈ ℝ^(${m}×${n}) — hàng: từ, cột: câu.`;
  }

  const baseChartOptions = () => ({
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 600, easing: "easeOutQuart" },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: COLORS.navy, padding: 10, cornerRadius: 6,
        titleFont: { family: "Be Vietnam Pro", weight: "600" },
        bodyFont: { family: "Be Vietnam Pro" },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: COLORS.muted, font: { family: "Be Vietnam Pro" }, maxRotation: 0, autoSkip: true } },
      y: { beginAtZero: true, grid: { color: "#EEF2F6" }, border: { display: false }, ticks: { color: COLORS.muted, font: { family: "Be Vietnam Pro" } } },
    },
  });

  function upsertChart(key, canvasId, config) {
    if (typeof Chart === "undefined") {
      toast("Không tải được thư viện Chart.js (cần Internet).", "warning", 5000);
      return;
    }
    if (state.charts[key]) state.charts[key].destroy();
    state.charts[key] = new Chart(document.getElementById(canvasId), config);
  }

  function renderSigmaChart(sigmas, k) {
    el.sigmaEmpty.hidden = true;
    el.sigmaTag.textContent = `k = ${k}`;
    const total = sigmas.reduce((a, s) => a + s * s, 0) || 1;
    const opts = baseChartOptions();
    opts.plugins.tooltip.callbacks = {
      title: (items) => items[0].label,
      label: (ctx) => [
        `σ = ${sigmas[ctx.dataIndex].toFixed(6)}`,
        `σ² chiếm ${(sigmas[ctx.dataIndex] ** 2 / total * 100).toFixed(2)}% năng lượng`,
        ctx.dataIndex < k ? "✓ Được giữ lại" : "✗ Bị loại bỏ",
      ],
    };
    upsertChart("sigma", "sigmaChart", {
      type: "bar",
      data: {
        labels: sigmas.map((_, i) => `σ${i + 1}`),
        datasets: [{
          data: sigmas,
          backgroundColor: sigmas.map((_, i) => (i < k ? COLORS.orange : COLORS.gray)),
          hoverBackgroundColor: sigmas.map((_, i) => (i < k ? "#D98706" : "#94A3B8")),
          borderRadius: 5, maxBarThickness: 46,
        }],
      },
      options: opts,
    });
    el.sigmaNote.textContent = `Cam: ${k} thành phần giữ lại · Xám: ${Math.max(0, sigmas.length - k)} thành phần bị loại.`;
  }

  function renderScoreChart(sentences) {
    el.scoreEmpty.hidden = true;
    const opts = baseChartOptions();
    opts.plugins.tooltip.callbacks = {
      title: (items) => `Câu ${sentences[items[0].dataIndex].id} — hạng ${sentences[items[0].dataIndex].rank}`,
      label: (ctx) => {
        const s = sentences[ctx.dataIndex];
        const preview = s.text.length > 70 ? `${s.text.slice(0, 70)}…` : s.text;
        return [`Điểm: ${s.score.toFixed(6)}`, s.selected ? "✓ Được chọn" : "Không chọn", preview];
      },
    };
    upsertChart("score", "scoreChart", {
      type: "bar",
      data: {
        labels: sentences.map((s) => `Câu ${s.id}`),
        datasets: [{
          data: sentences.map((s) => s.score),
          backgroundColor: sentences.map((s) => (s.selected ? COLORS.orange : COLORS.blue)),
          borderRadius: 5, maxBarThickness: 46,
        }],
      },
      options: opts,
    });
  }

  // ---------- Copy / download ----------
  el.copyBtn.addEventListener("click", async () => {
    const text = state.result?.summary;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = Object.assign(document.createElement("textarea"), { value: text });
      document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
    }
    toast("Đã sao chép bản tóm tắt.", "success", 1800);
  });

  el.downloadBtn.addEventListener("click", () => {
    const r = state.result;
    if (!r?.summary) return;
    const header = `BẢN TÓM TẮT (SVD + LSA)\nk = ${r.statistics.k_used} | tỷ lệ ${r.config.ratio}% | ${r.statistics.selected_sentences}/${r.statistics.original_sentences} câu\n${"-".repeat(48)}\n`;
    // BOM để Notepad trên Windows nhận đúng UTF-8
    const blob = new Blob(["\uFEFF" + header + r.summary + "\n"], { type: "text/plain;charset=utf-8" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob), download: "ban-tom-tat.txt",
    });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ---------- Init ----------
  updateCharCount();
  validateK();
  fetch("/api/health").then((r) => r.json())
    .then((d) => { el.tokenizerName.textContent = d.tokenizer; })
    .catch(() => { el.tokenizerName.textContent = "không xác định"; setStatus("error", "Mất kết nối máy chủ"); });
})();

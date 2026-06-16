DEMO_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic ASR Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d8dee8;
      --brand: #116d6e;
      --brand-dark: #0b5253;
      --danger: #b42318;
      --ok: #067647;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      padding: 20px;
      min-height: calc(100vh - 64px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .controls { padding: 18px; align-self: start; }
    .jobs { overflow: hidden; }
    label {
      display: block;
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    input, select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    input[type="file"] {
      height: auto;
      padding: 9px;
    }
    .field { margin-bottom: 14px; }
    .formats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .formats label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      font-weight: 500;
      font-size: 13px;
    }
    .formats input { width: 16px; height: 16px; }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--brand);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--brand-dark); }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .pager {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      padding: 10px 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .pager button {
      height: 30px;
      padding: 0 10px;
      background: #fff;
      border: 1px solid var(--line);
      color: var(--brand);
      font-size: 12px;
    }
    .toolbar h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fbfcfe;
    }
    .status {
      display: inline-flex;
      align-items: center;
      min-width: 86px;
      height: 24px;
      border-radius: 999px;
      padding: 0 9px;
      font-size: 12px;
      font-weight: 700;
      background: #eef2f6;
      color: var(--muted);
    }
    .succeeded { background: #dcfae6; color: var(--ok); }
    .failed { background: #fee4e2; color: var(--danger); }
    .running { background: #fef0c7; color: var(--warn); }
    .queued { background: #e0f2fe; color: #026aa2; }
    .downloads {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .downloads button {
      display: inline-flex;
      align-items: center;
      height: 28px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--brand);
      font-weight: 700;
      font-size: 12px;
    }
    .downloads button:hover { background: #f8fafc; }
    .review {
      grid-column: 1 / -1;
      padding: 16px;
    }
    .review h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .review-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }
    audio {
      width: 100%;
      margin-bottom: 10px;
    }
    .review-tools {
      display: grid;
      grid-template-columns: 80px minmax(0, 1fr) 96px;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .review-tools input {
      height: 28px;
      padding: 0;
    }
    .review-tools span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .wave-wrap {
      position: relative;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      overflow-x: auto;
      overflow-y: hidden;
      cursor: grab;
      user-select: none;
      touch-action: none;
    }
    .wave-wrap.dragging { cursor: grabbing; }
    canvas {
      display: block;
      height: 220px;
    }
    .segment-list {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .segment-item {
      width: 100%;
      height: auto;
      min-height: 48px;
      display: block;
      padding: 8px 10px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      background: #fff;
      color: var(--ink);
      text-align: left;
      font-weight: 500;
    }
    .segment-item:hover,
    .segment-item.active {
      background: #ecfeff;
      color: var(--brand-dark);
    }
    .segment-time {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 3px;
    }
    .message {
      min-height: 24px;
      color: var(--muted);
      font-size: 13px;
    }
    .message.error { color: var(--danger); }
    @media (max-width: 820px) {
      header { padding: 0 16px; }
      main { grid-template-columns: 1fr; padding: 14px; }
      .review-grid { grid-template-columns: 1fr; }
      th:nth-child(2), td:nth-child(2) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Semantic ASR Demo</h1>
    <div id="health" class="message">Checking service...</div>
  </header>
  <main>
    <section class="controls">
      <div class="field">
        <label for="user-name">User Name</label>
        <input id="user-name" type="text" value="dev" autocomplete="off">
      </div>
      <div class="field">
        <label for="token">API Token</label>
        <input id="token" type="password" value="dev-token" autocomplete="off">
      </div>
      <div class="field">
        <label for="config">Language / Profile</label>
        <select id="config"></select>
      </div>
      <div class="field">
        <label for="audio">Audio Files</label>
        <input id="audio" type="file" multiple accept=".wav,.flac,.mp3,.m4a,.ogg,audio/*">
      </div>
      <div class="field">
        <label>Outputs</label>
        <div class="formats">
          <label><input type="checkbox" name="format" value="json" checked> JSON</label>
          <label><input type="checkbox" name="format" value="srt" checked> SRT</label>
          <label><input type="checkbox" name="format" value="csv" checked> CSV</label>
          <label><input type="checkbox" name="format" value="textgrid" checked> TextGrid</label>
        </div>
      </div>
      <button id="submit">Submit</button>
      <p id="message" class="message"></p>
    </section>
    <section class="jobs">
      <div class="toolbar">
        <h2>Jobs</h2>
        <button id="refresh" type="button">Refresh</button>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 20%">File</th>
            <th style="width: 25%">Job ID</th>
            <th style="width: 14%">Status</th>
            <th style="width: 17%">Stage</th>
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody id="jobs"></tbody>
      </table>
      <div class="pager">
        <button id="prev-page" type="button">Prev</button>
        <span id="page-info">0-0 / 0</span>
        <button id="next-page" type="button">Next</button>
      </div>
    </section>
    <section class="review" id="review" hidden>
      <h2 id="review-title">Review</h2>
      <div class="review-grid">
        <div>
          <audio id="review-audio" controls></audio>
          <div class="review-tools">
            <span>Zoom</span>
            <input id="wave-zoom" type="range" min="1" max="48" step="1" value="1">
            <span id="wave-zoom-label">1x</span>
            <span>Visible</span>
            <span id="wave-visible">00:00.000 - 00:00.000</span>
            <span></span>
            <span>Text</span>
            <select id="text-mode">
              <option value="original">Original</option>
              <option value="translation">Translation</option>
              <option value="bilingual">Bilingual</option>
            </select>
            <span></span>
            <span>Target</span>
            <select id="translation-target"></select>
            <button id="translate" type="button">Translate</button>
          </div>
          <div class="wave-wrap">
            <canvas id="waveform" width="1200" height="220"></canvas>
          </div>
          <p id="review-message" class="message"></p>
        </div>
        <div id="segment-list" class="segment-list"></div>
      </div>
    </section>
  </main>
  <script>
    const state = {
      jobs: [],
      timer: null,
      review: null,
      waveDrag: null,
      translationsByJob: {},
      translatingByJob: {},
      page: { offset: 0, limit: 8, total: 0 },
    };
    const userNameInput = document.getElementById("user-name");
    const tokenInput = document.getElementById("token");
    const configSelect = document.getElementById("config");
    const fileInput = document.getElementById("audio");
    const jobsBody = document.getElementById("jobs");
    const message = document.getElementById("message");
    const health = document.getElementById("health");
    const review = document.getElementById("review");
    const reviewTitle = document.getElementById("review-title");
    const reviewAudio = document.getElementById("review-audio");
    const reviewMessage = document.getElementById("review-message");
    const segmentList = document.getElementById("segment-list");
    const waveform = document.getElementById("waveform");
    const waveContext = waveform.getContext("2d");
    const waveWrap = document.querySelector(".wave-wrap");
    const waveZoom = document.getElementById("wave-zoom");
    const waveZoomLabel = document.getElementById("wave-zoom-label");
    const waveVisible = document.getElementById("wave-visible");
    const textMode = document.getElementById("text-mode");
    const translationTarget = document.getElementById("translation-target");
    const translateButton = document.getElementById("translate");
    const pageInfo = document.getElementById("page-info");
    const prevPage = document.getElementById("prev-page");
    const nextPage = document.getElementById("next-page");

    function authHeaders() {
      const token = tokenInput.value.trim() || userNameInput.value.trim();
      return {
        Authorization: `Bearer ${token}`,
        "X-Semantic-ASR-User": userNameInput.value.trim() || "dev",
      };
    }

    async function apiFetch(url, options = {}) {
      const headers = Object.assign({}, options.headers || {}, authHeaders());
      const response = await fetch(url, Object.assign({}, options, { headers }));
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `${response.status} ${response.statusText}`);
      }
      return response;
    }

    async function loadConfigs() {
      const response = await apiFetch("/v1/configs");
      const data = await response.json();
      configSelect.innerHTML = "";
      data.configs.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        configSelect.appendChild(option);
      });
      if (data.configs.includes("vi_vn")) {
        configSelect.value = "vi_vn";
      }
      const savedConfig = localStorage.getItem("semanticAsrDemo.config");
      if (savedConfig && data.configs.includes(savedConfig)) {
        configSelect.value = savedConfig;
      }
    }

    async function loadTranslationTargets() {
      const response = await apiFetch("/v1/translation-targets");
      const data = await response.json();
      translationTarget.innerHTML = "";
      data.targets.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        translationTarget.appendChild(option);
      });
      if (data.targets.includes("zh_cn")) {
        translationTarget.value = "zh_cn";
      }
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        health.textContent = data.ok ? "Service online" : "Service unavailable";
        health.className = data.ok ? "message" : "message error";
      } catch (error) {
        health.textContent = "Service unavailable";
        health.className = "message error";
      }
    }

    function selectedFormats() {
      return Array.from(document.querySelectorAll("input[name='format']:checked"))
        .map((item) => item.value)
        .join(",");
    }

    async function submitJobs() {
      message.textContent = "";
      message.className = "message";
      const files = Array.from(fileInput.files || []);
      if (!files.length) {
        message.textContent = "请选择音频文件。";
        message.className = "message error";
        return;
      }
      document.getElementById("submit").disabled = true;
      try {
        for (const file of files) {
          const form = new FormData();
          const localPath = localPathForFile(file);
          form.append("audio", file);
          form.append("config", configSelect.value);
          form.append("formats", selectedFormats());
          form.append("local_path", localPath);
          const response = await apiFetch("/v1/jobs", { method: "POST", body: form });
          const job = await response.json();
          state.jobs.unshift({
            file: localPath,
            filename: file.name,
            local_path: localPath,
            fileObject: file,
            job_id: job.job_id,
            status: job.status,
            config: configSelect.value,
            progress: {},
            artifacts: {},
          });
        }
        savePreferences();
        renderJobs();
        startPolling();
        message.textContent = `已提交 ${files.length} 个任务。`;
      } catch (error) {
        message.textContent = error.message;
        message.className = "message error";
      } finally {
        document.getElementById("submit").disabled = false;
      }
    }

    async function refreshJobs() {
      await Promise.all(state.jobs.map(async (item) => {
        if (!item.job_id || item.status === "succeeded" || item.status === "failed") {
          return;
        }
        try {
          const response = await apiFetch(`/v1/jobs/${item.job_id}`);
          Object.assign(item, await response.json());
        } catch (error) {
          item.status = "failed";
          item.progress = { stage: error.message };
        }
      }));
      renderJobs();
      if (!state.jobs.some((item) => item.status === "queued" || item.status === "running")) {
        stopPolling();
      }
    }

    async function loadJobs() {
      message.textContent = "";
      message.className = "message";
      try {
        const response = await apiFetch(`/v1/jobs?limit=${state.page.limit}&offset=${state.page.offset}`);
        const data = await response.json();
        state.page.total = Number(data.total || 0);
        state.page.limit = Number(data.limit || state.page.limit);
        state.page.offset = Number(data.offset || state.page.offset);
        replaceJobs(data.jobs || []);
        renderJobs();
        if (state.jobs.some((item) => item.status === "queued" || item.status === "running")) {
          startPolling();
        }
      } catch (error) {
        message.textContent = error.message;
        message.className = "message error";
      }
    }

    async function loadMe() {
      const response = await apiFetch("/v1/me");
      const data = await response.json();
      message.textContent = `Current user: ${data.user_id}${data.is_admin ? " (admin)" : ""}`;
      message.className = "message";
    }

    function resetToFirstPage() {
      state.page.offset = 0;
    }

    function replaceJobs(jobs) {
      const currentFiles = new Map(state.jobs.map((item) => [item.job_id, item.fileObject]));
      state.jobs = jobs.map((job) => Object.assign({}, job, {
        file: job.local_path || job.filename || job.job_id,
        fileObject: currentFiles.get(job.job_id) || null,
      }));
    }

    function mergeJobs(jobs) {
      const existing = new Map(state.jobs.map((item) => [item.job_id, item]));
      jobs.forEach((job) => {
        const current = existing.get(job.job_id) || {};
        existing.set(job.job_id, Object.assign({}, current, job, {
          file: current.file || job.filename || job.job_id,
          filename: job.filename || current.filename || "",
          local_path: job.local_path || current.local_path || "",
          fileObject: current.fileObject || null,
        }));
      });
      state.jobs = Array.from(existing.values()).sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
    }

    function startPolling() {
      stopPolling();
      state.timer = setInterval(refreshJobs, 3000);
      refreshJobs();
    }

    function stopPolling() {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
    }

    function renderJobs() {
      jobsBody.innerHTML = "";
      state.jobs.forEach((job) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${escapeHtml(displayJobName(job))}<br><span class="message">${escapeHtml(job.config || "")}</span></td>
          <td>${escapeHtml(job.job_id || "")}</td>
          <td><span class="status ${escapeHtml(job.status || "")}">${escapeHtml(job.status || "")}</span></td>
          <td>${escapeHtml((job.progress && job.progress.stage) || "")}</td>
          <td>${artifactControls(job)}</td>
        `;
        jobsBody.appendChild(row);
      });
      renderPager();
    }

    function renderPager() {
      const total = state.page.total;
      const start = total ? state.page.offset + 1 : 0;
      const end = Math.min(total, state.page.offset + state.page.limit);
      pageInfo.textContent = `${start}-${end} / ${total}`;
      prevPage.disabled = state.page.offset <= 0;
      nextPage.disabled = state.page.offset + state.page.limit >= total;
    }

    function artifactControls(job) {
      if (!job.artifacts || !Object.keys(job.artifacts).length) {
        return job.error ? `<span class="message error">${escapeHtml(job.error)}</span>` : "";
      }
      const downloads = Object.keys(job.artifacts).map((name) => {
        return `<button type="button" data-download-job="${escapeHtml(job.job_id)}" data-download-format="${escapeHtml(name)}">${escapeHtml(name)}</button>`;
      }).join("");
      return `<div class="downloads"><button type="button" data-review-job="${escapeHtml(job.job_id)}">View</button>${downloads}</div>`;
    }

    async function downloadArtifact(jobId, format) {
      message.textContent = "";
      message.className = "message";
      try {
        const response = await apiFetch(`/v1/jobs/${jobId}/artifacts/${format}`);
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${jobId}.${format === "textgrid" ? "TextGrid" : format}`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        message.textContent = error.message;
        message.className = "message error";
      }
    }

    async function openReview(jobId) {
      const job = state.jobs.find((item) => item.job_id === jobId);
      if (!job) {
        return;
      }
      message.textContent = "";
      message.className = "message";
      review.hidden = false;
      reviewTitle.textContent = `Review - ${displayJobName(job)}`;
      reviewMessage.textContent = "Loading waveform...";
      reviewMessage.className = "message";
      segmentList.innerHTML = "";

      const resultResponse = await apiFetch(`/v1/jobs/${jobId}/result`);
      const result = await resultResponse.json();
      const audioFile = await audioFileForJob(job);
      const audioUrl = URL.createObjectURL(audioFile);
      if (reviewAudio.src) {
        URL.revokeObjectURL(reviewAudio.src);
      }
      reviewAudio.src = audioUrl;
      const buffer = await decodeAudioFile(audioFile);
      const segments = normalizeSegments(result.sentences || []);
      state.review = {
        jobId,
        file: audioFile,
        audioBuffer: buffer,
        duration: result.dur_s || buffer.duration,
        segments,
        translations: translationsForJob(jobId, translationTarget.value),
        displayMode: textMode.value,
        zoom: Number(waveZoom.value || 1),
        playUntilMs: null,
        peaksByWidth: new Map(),
      };
      renderSegmentList(segments);
      resizeWaveformForZoom();
      drawWaveform();
      reviewMessage.textContent = `${segments.length} segments loaded.`;
      updateTranslationUi();
      review.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    async function audioFileForJob(job) {
      if (job.fileObject) {
        return job.fileObject;
      }
      const response = await apiFetch(`/v1/jobs/${job.job_id}/audio`);
      const blob = await response.blob();
      const name = job.filename || `${job.job_id}.wav`;
      return new File([blob], name, { type: blob.type || "audio/wav" });
    }

    function localPathForFile(file) {
      return file.webkitRelativePath || file.name || "";
    }

    function displayJobName(job) {
      return job.local_path || job.file || job.filename || job.job_id || "";
    }

    async function decodeAudioFile(file) {
      const arrayBuffer = await file.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      try {
        return await audioContext.decodeAudioData(arrayBuffer.slice(0));
      } finally {
        if (audioContext.close) {
          audioContext.close();
        }
      }
    }

    function normalizeSegments(sentences) {
      return sentences.map((sentence, index) => ({
        index,
        startMs: Number(sentence.cut_start_ms ?? sentence.start_ms ?? 0),
        endMs: Number(sentence.cut_end_ms ?? sentence.end_ms ?? 0),
        text: sentence.text || "",
      })).filter((segment) => segment.endMs > segment.startMs);
    }

    function renderSegmentList(segments) {
      segmentList.innerHTML = segments.map((segment) => `
        <button type="button" class="segment-item" data-segment-index="${segment.index}">
          <span class="segment-time">${formatMs(segment.startMs)} - ${formatMs(segment.endMs)}</span>
          ${segmentTextHtml(segment)}
        </button>
      `).join("");
    }

    function segmentTextHtml(segment) {
      const reviewState = state.review || {};
      const translated = (reviewState.translations || {})[segment.index] || "";
      const mode = reviewState.displayMode || "original";
      if (mode === "translation") {
        return escapeHtml(translated || segment.text);
      }
      if (mode === "bilingual" && translated) {
        return `${escapeHtml(segment.text)}<br><span class="message">${escapeHtml(translated)}</span>`;
      }
      return escapeHtml(segment.text);
    }

    async function translateReview() {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const target = translationTarget.value;
      const jobId = reviewState.jobId;
      const key = translationKey(jobId, target);
      state.translatingByJob[key] = true;
      state.translationsByJob[key] = state.translationsByJob[key] || {};
      reviewState.translations = state.translationsByJob[key];
      textMode.value = "bilingual";
      reviewState.displayMode = textMode.value;
      renderSegmentList(reviewState.segments);
      updateTranslationUi();
      reviewMessage.textContent = "Translating...";
      reviewMessage.className = "message";
      try {
        const response = await apiFetch(`/v1/jobs/${jobId}/translations/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_language: target }),
        });
        await readTranslationStream(response, jobId, target);
      } catch (error) {
        if (state.review && state.review.jobId === jobId) {
          reviewMessage.textContent = error.message;
          reviewMessage.className = "message error";
        }
      } finally {
        delete state.translatingByJob[key];
        updateTranslationUi();
      }
    }

    async function readTranslationStream(response, jobId, target) {
      if (!response.body) {
        const result = await response.json();
        applyTranslationPayload(jobId, target, result);
        return;
      }
      const key = translationKey(jobId, target);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\\n");
        buffer = lines.pop() || "";
        lines.forEach((line) => handleTranslationEvent(line, jobId, target, key));
      }
      if (buffer.trim()) {
        handleTranslationEvent(buffer, jobId, target, key);
      }
    }

    function handleTranslationEvent(line, jobId, target, key) {
      const event = JSON.parse(line);
      if (event.type === "error") {
        throw new Error(event.error || "Translation failed");
      }
      if (event.type === "sentence" && event.sentence) {
        const sentence = event.sentence;
        state.translationsByJob[key] = state.translationsByJob[key] || {};
        state.translationsByJob[key][Number(sentence.index)] = sentence.translation || "";
        if (state.review && state.review.jobId === jobId && translationTarget.value === target) {
          state.review.translations = state.translationsByJob[key];
          renderSegmentList(state.review.segments);
          updateActiveSegment(reviewAudio.currentTime * 1000);
          reviewMessage.textContent = `Translated ${Object.keys(state.review.translations).length} segments...`;
          reviewMessage.className = "message";
        }
      }
      if (event.type === "done") {
        applyTranslationPayload(jobId, target, event.payload || {});
      }
    }

    function applyTranslationPayload(jobId, target, payload) {
      const key = translationKey(jobId, target);
      const translations = {};
      (payload.sentences || []).forEach((sentence) => {
        translations[Number(sentence.index)] = sentence.translation || "";
      });
      state.translationsByJob[key] = Object.keys(translations).length ? translations : (state.translationsByJob[key] || {});
      if (state.review && state.review.jobId === jobId && translationTarget.value === target) {
        state.review.translations = state.translationsByJob[key];
        textMode.value = "bilingual";
        state.review.displayMode = textMode.value;
        renderSegmentList(state.review.segments);
        reviewMessage.textContent = `Translation loaded: ${target}`;
        reviewMessage.className = "message";
      }
    }

    function translationKey(jobId, target) {
      return `${jobId}:${target}`;
    }

    function translationsForJob(jobId, target) {
      return state.translationsByJob[translationKey(jobId, target)] || {};
    }

    function updateTranslationUi() {
      const reviewState = state.review;
      if (!reviewState) {
        translateButton.disabled = true;
        return;
      }
      const key = translationKey(reviewState.jobId, translationTarget.value);
      const translating = Boolean(state.translatingByJob[key]);
      translateButton.disabled = translating;
      translateButton.textContent = translating ? "Translating..." : "Translate";
      if (translating) {
        reviewMessage.textContent = "Translating...";
        reviewMessage.className = "message";
      }
    }

    function savePreferences() {
      localStorage.setItem("semanticAsrDemo.userName", userNameInput.value.trim());
      localStorage.setItem("semanticAsrDemo.token", tokenInput.value.trim());
      localStorage.setItem("semanticAsrDemo.config", configSelect.value);
    }

    function loadPreferences() {
      const userName = localStorage.getItem("semanticAsrDemo.userName");
      const token = localStorage.getItem("semanticAsrDemo.token");
      if (userName) userNameInput.value = userName;
      if (token) tokenInput.value = token;
    }

    function drawWaveform() {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      resizeWaveformForZoom();
      const width = waveform.width;
      const height = waveform.height;
      const durationMs = Math.max(1, reviewState.duration * 1000);
      waveContext.clearRect(0, 0, width, height);
      waveContext.fillStyle = "#fbfcfe";
      waveContext.fillRect(0, 0, width, height);

      const peaks = peaksForWidth(reviewState, width);
      waveContext.strokeStyle = "#475467";
      waveContext.lineWidth = 1;
      waveContext.beginPath();
      for (let x = 0; x < width; x += 1) {
        const min = peaks[x * 2];
        const max = peaks[x * 2 + 1];
        waveContext.moveTo(x, (1 - max) * height / 2);
        waveContext.lineTo(x, (1 - min) * height / 2);
      }
      waveContext.stroke();

      reviewState.segments.forEach((segment, index) => {
        const x = segment.startMs / durationMs * width;
        const w = Math.max(1, (segment.endMs - segment.startMs) / durationMs * width);
        waveContext.fillStyle = index % 2 === 0 ? "rgba(17, 109, 110, 0.18)" : "rgba(180, 84, 8, 0.16)";
        waveContext.fillRect(x, 0, w, height);
      });
      drawPlaybackCursor();
      updateVisibleWindow();
    }

    function resizeWaveformForZoom() {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const zoom = clampZoom(Number(waveZoom.value || reviewState.zoom || 1));
      reviewState.zoom = zoom;
      waveZoom.value = String(zoom);
      waveZoomLabel.textContent = `${zoom}x`;
      const visibleWidth = Math.max(600, Math.floor(waveWrap.clientWidth || 1200));
      const targetWidth = Math.min(60000, Math.max(visibleWidth, Math.floor(visibleWidth * zoom)));
      if (waveform.width !== targetWidth) {
        waveform.width = targetWidth;
        waveform.style.width = `${targetWidth}px`;
      }
      waveform.height = 220;
    }

    function setWaveZoom(nextZoom, anchorClientX = null) {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const oldWidth = waveform.width || waveWrap.clientWidth || 1;
      let anchorRatio;
      if (anchorClientX === null) {
        anchorRatio = (waveWrap.scrollLeft + waveWrap.clientWidth / 2) / oldWidth;
      } else {
        const rect = waveform.getBoundingClientRect();
        anchorRatio = (waveWrap.scrollLeft + anchorClientX - rect.left) / oldWidth;
      }
      reviewState.zoom = clampZoom(nextZoom);
      waveZoom.value = String(reviewState.zoom);
      resizeWaveformForZoom();
      drawWaveform();
      const anchorX = anchorRatio * waveform.width;
      if (anchorClientX === null) {
        waveWrap.scrollLeft = Math.max(0, anchorX - waveWrap.clientWidth / 2);
      } else {
        const wrapRect = waveWrap.getBoundingClientRect();
        waveWrap.scrollLeft = Math.max(0, anchorX - (anchorClientX - wrapRect.left));
      }
      updateVisibleWindow();
    }

    function clampZoom(value) {
      return Math.max(1, Math.min(48, Math.round(value || 1)));
    }

    function peaksForWidth(reviewState, width) {
      if (reviewState.peaksByWidth.has(width)) {
        return reviewState.peaksByWidth.get(width);
      }
      const channel = reviewState.audioBuffer.getChannelData(0);
      const peaks = new Float32Array(width * 2);
      for (let x = 0; x < width; x += 1) {
        const start = Math.floor(x * channel.length / width);
        const stop = Math.max(start + 1, Math.floor((x + 1) * channel.length / width));
        let min = 1;
        let max = -1;
        for (let i = start; i < stop && i < channel.length; i += 1) {
          const value = channel[i];
          if (value < min) min = value;
          if (value > max) max = value;
        }
        peaks[x * 2] = min;
        peaks[x * 2 + 1] = max;
      }
      reviewState.peaksByWidth.set(width, peaks);
      return peaks;
    }

    function drawPlaybackCursor() {
      const reviewState = state.review;
      if (!reviewState || !reviewAudio.duration) {
        return;
      }
      const x = reviewAudio.currentTime / reviewAudio.duration * waveform.width;
      waveContext.strokeStyle = "#b42318";
      waveContext.lineWidth = 2;
      waveContext.beginPath();
      waveContext.moveTo(x, 0);
      waveContext.lineTo(x, waveform.height);
      waveContext.stroke();
      updateActiveSegment(reviewAudio.currentTime * 1000);
      keepPlaybackCursorVisible(x);
    }

    function keepPlaybackCursorVisible(cursorX) {
      if (reviewAudio.paused) {
        return;
      }
      const left = waveWrap.scrollLeft;
      const right = left + waveWrap.clientWidth;
      if (cursorX < left + 24 || cursorX > right - 24) {
        waveWrap.scrollLeft = Math.max(0, cursorX - waveWrap.clientWidth * 0.35);
        updateVisibleWindow();
      }
    }

    function seekSegment(index) {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const segment = reviewState.segments.find((item) => item.index === Number(index));
      if (!segment) {
        return;
      }
      reviewAudio.currentTime = segment.startMs / 1000;
      reviewState.playUntilMs = segment.endMs;
      scrollToTime(segment.startMs);
      reviewAudio.play();
      drawWaveform();
    }

    function handleAudioTimeUpdate() {
      const reviewState = state.review;
      if (reviewState && reviewState.playUntilMs !== null && reviewAudio.currentTime * 1000 >= reviewState.playUntilMs) {
        reviewAudio.pause();
        reviewAudio.currentTime = reviewState.playUntilMs / 1000;
        reviewState.playUntilMs = null;
      }
      drawWaveform();
    }

    function updateActiveSegment(currentMs) {
      document.querySelectorAll(".segment-item").forEach((item) => item.classList.remove("active"));
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const active = reviewState.segments.find((segment) => currentMs >= segment.startMs && currentMs < segment.endMs);
      if (!active) {
        return;
      }
      const button = segmentList.querySelector(`[data-segment-index="${active.index}"]`);
      if (button) {
        button.classList.add("active");
      }
    }

    function scrollToTime(ms) {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const durationMs = Math.max(1, reviewState.duration * 1000);
      const x = ms / durationMs * waveform.width;
      waveWrap.scrollLeft = Math.max(0, x - waveWrap.clientWidth * 0.2);
      updateVisibleWindow();
    }

    function updateVisibleWindow() {
      const reviewState = state.review;
      if (!reviewState) {
        return;
      }
      const durationMs = Math.max(1, reviewState.duration * 1000);
      const startMs = waveWrap.scrollLeft / waveform.width * durationMs;
      const endMs = (waveWrap.scrollLeft + waveWrap.clientWidth) / waveform.width * durationMs;
      waveVisible.textContent = `${formatMs(startMs)} - ${formatMs(Math.min(durationMs, endMs))}`;
    }

    function formatMs(ms) {
      const totalSeconds = Math.max(0, Math.floor(ms / 1000));
      const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
      const seconds = String(totalSeconds % 60).padStart(2, "0");
      const millis = String(Math.floor(ms % 1000)).padStart(3, "0");
      return `${minutes}:${seconds}.${millis}`;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    document.getElementById("submit").addEventListener("click", submitJobs);
    document.getElementById("refresh").addEventListener("click", loadJobs);
    prevPage.addEventListener("click", () => {
      state.page.offset = Math.max(0, state.page.offset - state.page.limit);
      loadJobs();
    });
    nextPage.addEventListener("click", () => {
      state.page.offset += state.page.limit;
      loadJobs();
    });
    jobsBody.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-download-job]");
      if (!button) {
        return;
      }
      downloadArtifact(button.dataset.downloadJob, button.dataset.downloadFormat);
    });
    jobsBody.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-review-job]");
      if (!button) {
        return;
      }
      openReview(button.dataset.reviewJob).catch((error) => {
        message.textContent = error.message;
        message.className = "message error";
      });
    });
    segmentList.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-segment-index]");
      if (!button) {
        return;
      }
      seekSegment(button.dataset.segmentIndex);
    });
    reviewAudio.addEventListener("timeupdate", handleAudioTimeUpdate);
    waveZoom.addEventListener("input", () => setWaveZoom(Number(waveZoom.value)));
    textMode.addEventListener("change", () => {
      if (!state.review) {
        return;
      }
      state.review.displayMode = textMode.value;
      renderSegmentList(state.review.segments);
      updateActiveSegment(reviewAudio.currentTime * 1000);
    });
    translationTarget.addEventListener("change", () => {
      if (!state.review) {
        return;
      }
      state.review.translations = translationsForJob(state.review.jobId, translationTarget.value);
      renderSegmentList(state.review.segments);
      updateTranslationUi();
    });
    translateButton.addEventListener("click", translateReview);
    waveWrap.addEventListener("scroll", updateVisibleWindow);
    window.addEventListener("resize", drawWaveform);
    waveWrap.addEventListener("wheel", (event) => {
      if (!state.review) {
        return;
      }
      event.preventDefault();
      const direction = event.deltaY < 0 ? 1 : -1;
      const factor = event.shiftKey ? 4 : 2;
      setWaveZoom(Number(waveZoom.value) + direction * factor, event.clientX);
    }, { passive: false });
    waveWrap.addEventListener("pointerdown", (event) => {
      if (!state.review) {
        return;
      }
      state.waveDrag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        scrollLeft: waveWrap.scrollLeft,
        moved: false,
      };
      waveWrap.classList.add("dragging");
      waveWrap.setPointerCapture(event.pointerId);
    });
    waveWrap.addEventListener("pointermove", (event) => {
      const drag = state.waveDrag;
      if (!drag || drag.pointerId !== event.pointerId) {
        return;
      }
      const deltaX = event.clientX - drag.startX;
      if (Math.abs(deltaX) > 3) {
        drag.moved = true;
      }
      waveWrap.scrollLeft = drag.scrollLeft - deltaX;
      updateVisibleWindow();
    });
    waveWrap.addEventListener("pointerup", (event) => finishWaveDrag(event));
    waveWrap.addEventListener("pointercancel", (event) => finishWaveDrag(event));
    waveform.addEventListener("click", (event) => {
      const finishedDrag = state.waveDrag;
      state.waveDrag = null;
      if (finishedDrag && finishedDrag.moved) {
        state.waveDrag = null;
        return;
      }
      const rect = waveform.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      if (reviewAudio.duration) {
        if (state.review) {
          state.review.playUntilMs = null;
        }
        reviewAudio.currentTime = ratio * reviewAudio.duration;
        drawWaveform();
      }
    });

    function finishWaveDrag(event) {
      const drag = state.waveDrag;
      if (!drag || drag.pointerId !== event.pointerId) {
        return;
      }
      waveWrap.classList.remove("dragging");
      try {
        waveWrap.releasePointerCapture(event.pointerId);
      } catch (_error) {
      }
    }
    userNameInput.addEventListener("change", () => {
      savePreferences();
      resetToFirstPage();
      state.jobs = [];
      state.review = null;
      review.hidden = true;
      renderJobs();
      Promise.all([loadMe(), loadJobs()]).catch((error) => {
        message.textContent = error.message;
        message.className = "message error";
      });
    });
    tokenInput.addEventListener("change", () => {
      savePreferences();
      resetToFirstPage();
      state.review = null;
      review.hidden = true;
      Promise.all([loadMe(), loadConfigs(), loadTranslationTargets(), loadJobs()]).catch((error) => {
        message.textContent = error.message;
        message.className = "message error";
      });
    });
    configSelect.addEventListener("change", savePreferences);
    loadPreferences();
    checkHealth();
    Promise.all([loadMe(), loadConfigs(), loadTranslationTargets(), loadJobs()]).catch((error) => {
      message.textContent = error.message;
      message.className = "message error";
    });
  </script>
</body>
</html>
"""

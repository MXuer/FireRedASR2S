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
    .downloads a {
      display: inline-flex;
      align-items: center;
      height: 28px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--brand);
      text-decoration: none;
      font-weight: 700;
      font-size: 12px;
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
    </section>
  </main>
  <script>
    const state = { jobs: [], timer: null };
    const tokenInput = document.getElementById("token");
    const configSelect = document.getElementById("config");
    const fileInput = document.getElementById("audio");
    const jobsBody = document.getElementById("jobs");
    const message = document.getElementById("message");
    const health = document.getElementById("health");

    function authHeaders() {
      return { Authorization: `Bearer ${tokenInput.value.trim()}` };
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
          form.append("audio", file);
          form.append("config", configSelect.value);
          form.append("formats", selectedFormats());
          const response = await apiFetch("/v1/jobs", { method: "POST", body: form });
          const job = await response.json();
          state.jobs.unshift({ file: file.name, job_id: job.job_id, status: job.status, progress: {}, artifacts: {} });
        }
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
          <td>${escapeHtml(job.file || "")}</td>
          <td>${escapeHtml(job.job_id || "")}</td>
          <td><span class="status ${escapeHtml(job.status || "")}">${escapeHtml(job.status || "")}</span></td>
          <td>${escapeHtml((job.progress && job.progress.stage) || "")}</td>
          <td>${artifactLinks(job)}</td>
        `;
        jobsBody.appendChild(row);
      });
    }

    function artifactLinks(job) {
      if (!job.artifacts || !Object.keys(job.artifacts).length) {
        return job.error ? `<span class="message error">${escapeHtml(job.error)}</span>` : "";
      }
      return `<div class="downloads">${Object.keys(job.artifacts).map((name) => {
        const href = job.artifacts[name];
        return `<a href="${href}" target="_blank" rel="noopener">${escapeHtml(name)}</a>`;
      }).join("")}</div>`;
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
    document.getElementById("refresh").addEventListener("click", refreshJobs);
    tokenInput.addEventListener("change", loadConfigs);
    checkHealth();
    loadConfigs().catch((error) => {
      message.textContent = error.message;
      message.className = "message error";
    });
  </script>
</body>
</html>
"""

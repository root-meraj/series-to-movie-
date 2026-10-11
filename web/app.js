"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const DEFAULT_UPLOAD_LIMIT = 2 * 1024 ** 3;
  const STAGES = [
    ["ingest", "Import episodes", "Put your episodes in order."],
    ["transcribe", "Capture dialogue", "Use subtitles first; recognize dialogue when needed."],
    ["scenes", "Find the scenes", "Build complete story sequences."],
    ["nlp", "Understand the story", "Score the moments that move the story forward."],
    ["select", "Shape your edit", "Balance the story against your chosen runtime."],
    ["render", "Render your film", "Prepare the film and its subtitles."],
  ];
  const app = {
    api: "", token: "", connected: false, health: null, jobs: [], detail: null,
    route: "", poll: null, pollBusy: false, busy: false, file: null, selectedFile: null,
    fileError: "", notice: null,
    mediaJob: "", mediaPromise: null, ticketExpires: 0, renderSerial: 0,
  };
  let storedAddress = "";
  try { storedAddress = localStorage.getItem("sts.workspaceAddress") || ""; } catch { /* Storage may be unavailable. */ }
  app.api = window.STS_CONFIG?.apiBaseUrl || storedAddress;
  $("#api-url").value = app.api;

  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }
  function clamp(value, low = 0, high = 1) { return Math.min(high, Math.max(low, value)); }
  function minuteLabel(seconds) {
    if (!Number.isFinite(Number(seconds)) || Number(seconds) <= 0) return "—";
    const minutes = Number(seconds) / 60;
    return `${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} min`;
  }
  function sizeLabel(bytes) {
    if (bytes < 1024) return `${bytes} ${bytes === 1 ? "byte" : "bytes"}`;
    if (bytes < 1024 ** 2) return `${Math.ceil(bytes / 1024)} KB`;
    return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, "")} GB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
  }
  function uploadLimit() {
    const limit = Number(app.health?.max_upload_bytes);
    return Number.isSafeInteger(limit) && limit > 0 ? limit : DEFAULT_UPLOAD_LIMIT;
  }
  function archiveError(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) return "Choose a ZIP archive containing your episode videos.";
    if (file.size === 0) return "This ZIP is empty. Choose an archive with your episodes inside.";
    if (file.size > uploadLimit()) return `This ZIP exceeds the ${sizeLabel(uploadLimit())} upload limit. Choose a smaller archive.`;
    return "";
  }
  function syncSelectedFile() {
    const file = app.selectedFile;
    if (!file) return;
    app.fileError = archiveError(file);
    app.file = app.fileError ? null : file;
    $("#file-name").textContent = file.name;
    const status = app.fileError ? "Cannot upload this file" : app.connected ? "Ready to upload" : "Connect workspace to upload";
    $("#file-description").textContent = `${sizeLabel(file.size)} · ${status}`;
    $("#archive").setAttribute("aria-invalid", String(Boolean(app.fileError)));
    setInlineError("#create-error", app.fileError);
  }
  function syncCreateConnection() {
    if (!$("#create-form")) return;
    $(".upload-meta").textContent = `ZIP · MP4, MKV, AVI, MOV, WEBM · Up to ${sizeLabel(uploadLimit())}`;
    if (!app.busy) {
      $("#create-submit").innerHTML = `${app.connected ? "Create my film" : "Connect to create my film"} <span aria-hidden="true">↗</span>`;
      $(".create-caption").textContent = app.connected
        ? "You can leave the page after upload. Your progress is saved."
        : "Connect a processing server before uploading. Your chosen file stays here.";
    }
    syncSelectedFile();
  }
  function dateLabel(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Saved project" : date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }
  function statusFor(state, detail = null) {
    const stages = STAGES.map(([key]) => state.stages?.[key]?.status);
    if (stages.every(status => status === "done")) return ["Ready", "ready"];
    if (detail?.stopping) return ["Stopping", "processing"];
    if (detail?.running) return ["Processing", "processing"];
    if (stages.includes("failed") || detail?.error) return ["Needs attention", "attention"];
    if (stages.includes("in_progress")) return detail ? ["Paused", "paused"] : ["In progress", "processing"];
    return ["Waiting", "paused"];
  }
  function stageFraction(stage) {
    if (!stage) return 0;
    if (stage.status === "done") return 1;
    const total = (stage.completed?.length || 0) + (stage.pending?.length || 0);
    return total ? clamp((stage.completed?.length || 0) / total) : 0;
  }
  function overallFraction(state) {
    return STAGES.reduce((sum, [key]) => {
      const stage = state.stages?.[key];
      if (key === "nlp" && stage?.status !== "done") {
        return sum + ["characters", "baseline", "llm"].reduce((count, sub) => count + stageFraction(stage[sub]), 0) / 3;
      }
      return sum + stageFraction(stage);
    }, 0) / STAGES.length;
  }
  function announce(message) {
    clearTimeout(app.notice);
    $("#notice").textContent = message;
    $("#notice").hidden = false;
    app.notice = setTimeout(() => { $("#notice").hidden = true; }, 6000);
  }
  function setInlineError(selector, message) {
    const element = $(selector);
    if (element) { element.textContent = message || ""; element.hidden = !message; }
  }
  function responseError(payload, status) {
    if (status === 401 || status === 403) return "The workspace access key was not accepted. Open connection settings and try again.";
    if (status === 413) return "This ZIP is larger than the workspace upload limit. Choose a smaller archive.";
    if (status === 404) return "This project is no longer available in the workspace.";
    const detail = payload?.error || payload?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return "Check the values you entered and try again.";
    return `The workspace could not finish this request (${status}). Please try again.`;
  }
  function apiUrl(path) { return `${app.api}${path}`; }
  async function request(path, { method = "GET", body, publicRequest = false, signal } = {}) {
    const headers = {};
    if (!publicRequest) headers.Authorization = `Bearer ${app.token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const response = await fetch(apiUrl(path), {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal, credentials: "omit", cache: "no-store", referrerPolicy: "no-referrer",
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(responseError(payload, response.status));
      if (payload === null) throw new Error("The workspace sent an unreadable response. Please try again.");
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The workspace took too long to respond. Check your connection and try again.");
      if (error instanceof TypeError) throw new Error("Could not reach the workspace. Check its address, HTTPS connection, and allowed website origin.");
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    }
  }
  function normalizedAddress(value) {
    const url = new URL(value.trim());
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new Error("Enter the server’s HTTP or HTTPS address without credentials or query parameters.");
    }
    if (location.protocol === "https:" && url.protocol !== "https:") {
      throw new Error("This online website needs an HTTPS workspace address. A laptop’s local address is not an online processing server.");
    }
    return url.href.replace(/\/+$/, "");
  }
  function openConnection() {
    if (app.busy) { announce("Wait for the current request to finish before changing your connection."); return; }
    $("#api-url").value = app.api;
    $("#access-key").value = "";
    $("#disconnect-button").hidden = !app.connected;
    setInlineError("#connect-error", "");
    $("#connect-dialog").showModal();
  }
  function syncConnection() {
    document.body.classList.toggle("connected", app.connected);
    $("#connection-label").textContent = app.connected ? "Workspace connected" : "Connect workspace";
    $("#topbar-status").textContent = app.connected ? "Workspace connected" : "Not connected";
    $("#film-count").textContent = app.jobs.length;
    syncCreateConnection();
  }
  function disconnect() {
    clearTimeout(app.poll);
    app.token = "";
    app.connected = false;
    app.jobs = [];
    app.detail = null;
    app.health = null;
    clearMedia();
    syncConnection();
    $("#connect-dialog").close();
    if (location.hash !== "#new") location.hash = "new"; else renderRoute();
    announce("Workspace disconnected. Your saved projects stay on the server.");
  }
  async function refreshJobs() {
    const jobs = await request("/api/jobs");
    if (!Array.isArray(jobs)) throw new Error("The workspace sent an unreadable project list.");
    app.jobs = jobs.filter(job => typeof job?.job_id === "string");
    syncConnection();
    return app.jobs;
  }
  $("#connect-form").addEventListener("submit", async event => {
    event.preventDefault();
    const button = $("#connect-submit");
    const previous = { api: app.api, token: app.token, connected: app.connected };
    button.disabled = true;
    button.textContent = "Connecting…";
    setInlineError("#connect-error", "");
    try {
      app.api = normalizedAddress($("#api-url").value);
      app.token = $("#access-key").value.trim();
      if (!app.token) throw new Error("Enter your workspace access key.");
      const health = await request("/api/health", { publicRequest: true });
      if (health.status !== "ok") throw new Error("The processing server is not ready. Please try again shortly.");
      await refreshJobs();
      app.health = health;
      app.connected = true;
      try { localStorage.setItem("sts.workspaceAddress", app.api); } catch { /* Connection still works without storage. */ }
      $("#access-key").value = "";
      syncConnection();
      $("#connect-dialog").close();
      clearMedia();
      if (app.route === "new") {
        $(".connection-callout")?.remove();
      } else {
        const targetRoute = app.route.startsWith("film/") && app.jobs.some(job => `film/${job.job_id}` === app.route) ? app.route : "films";
        if (location.hash !== `#${targetRoute}`) location.hash = targetRoute; else renderRoute();
      }
      announce("Your editing workspace is connected.");
    } catch (error) {
      app.api = previous.api;
      app.token = previous.token;
      app.connected = previous.connected;
      syncConnection();
      setInlineError("#connect-error", error.message);
    } finally {
      button.disabled = false;
      button.innerHTML = 'Connect workspace <span aria-hidden="true">↗</span>';
    }
  });
  $("#disconnect-button").addEventListener("click", disconnect);
  $("#connection-button").addEventListener("click", openConnection);
  $("#topbar-connect").addEventListener("click", openConnection);
  document.querySelectorAll("[data-close-dialog]").forEach(button => button.addEventListener("click", () => button.closest("dialog").close()));
  $("#connect-dialog").addEventListener("close", () => { $("#access-key").value = ""; });

  function connectionCallout() {
    return `<div class="connection-callout"><span aria-hidden="true">↗</span><p>Your editing workspace is not connected yet.<br>This website needs a processing server to turn episodes into films. <button type="button" data-action="connect">Connect workspace</button> using that server’s address and access key.</p></div>`;
  }
  function renderNew() {
    const limit = uploadLimit();
    const target = Number(app.health?.default_target_minutes) || 120;
    $("#main").innerHTML = `
      <section class="intro" aria-labelledby="create-title"><div class="intro-copy"><span class="eyebrow">A SEASON, DISTILLED</span><h1 id="create-title">All the story.<br><em>Your kind of film.</em></h1><p>Turn your episodes into one considered cut.<br>The key moments, the connections, the ending.</p></div><div class="story-art" aria-hidden="true"><div class="art-front"><small>THE EDITING ROOM · 01</small><strong>A season.<br>One story.<br>Your cut.</strong><span><i></i>SEASON TO SCREEN</span></div></div></section>
      <div class="steps-strip" aria-label="Three simple steps"><div><b>01</b><span>Bring your episodes</span></div><div><b>02</b><span>Choose your runtime</span></div><div><b>03</b><span>Watch your film</span></div></div>
      <form id="create-form"><div class="create-grid"><section class="panel"><div class="panel-heading"><h3>Bring the season</h3><small>YOUR SOURCE</small></div><p class="panel-description">One ZIP. All your episodes. We’ll put them in order.</p><label class="upload-zone" id="upload-zone" for="archive"><span class="upload-icon" aria-hidden="true">↑</span><strong id="file-name">Choose your episode ZIP</strong><small id="file-description">Browse files or drop your archive here</small><span class="file-picker-caption">Choose a file</span><input type="file" id="archive" name="archive" accept=".zip,application/zip,application/x-zip-compressed" aria-label="Choose an episode ZIP archive"></label><p class="upload-meta">ZIP · MP4, MKV, AVI, MOV, WEBM · Up to ${escape(sizeLabel(limit))}</p><label for="project-name">Film name</label><input id="project-name" name="name" maxlength="160" type="text" placeholder="Give your story a name" autocomplete="off"><p class="field-help">A name makes your film easy to find later.</p></section>
      <section class="panel"><div class="panel-heading"><h3>Make it your cut</h3><small>YOUR EDIT</small></div><p class="panel-description">Choose how much time you have for the story.</p><label for="target-minutes">Target film length</label><div class="runtime-row"><input id="target-minutes" name="target_minutes" type="number" min="1" step="0.5" value="${target}" inputmode="decimal" required><span>minutes</span></div><div class="runtime-presets" aria-label="Suggested film lengths"><button class="preset" type="button" data-minutes="60">60 min</button><button class="preset" type="button" data-minutes="90">90 min</button><button class="preset ${target === 120 ? "selected" : ""}" type="button" data-minutes="120">120 min</button></div><p class="field-help">The finished length depends on available story footage.</p><ul class="included"><li><span aria-hidden="true">✓</span> Subtitles first, for faster story analysis</li><li><span aria-hidden="true">✓</span> Keep context, key moments, and the ending</li><li><span aria-hidden="true">✓</span> Remove detected episode credits</li><li><span aria-hidden="true">✓</span> Download your film and subtitles</li></ul><p id="create-error" class="inline-error" role="alert" hidden></p><div id="upload-progress" class="upload-progress" role="status" aria-live="polite" hidden><div class="progress-caption"><span id="upload-label">Uploading your episodes</span><span id="upload-percent">0%</span></div><div class="progress-track" role="progressbar" aria-label="ZIP upload" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span id="upload-bar" style="width:0%"></span></div></div><button id="create-submit" class="button primary full-width" type="submit">Create my film <span aria-hidden="true">↗</span></button><p class="create-caption">You can leave the page after upload. Your progress is saved.</p></section></div></form>
      ${!app.connected ? connectionCallout() : '<p class="field-help" style="margin-top:22px">Your workspace handles processing and story scoring. You can follow the edit from any device connected to it.</p>'}`;
    app.file = null;
    app.selectedFile = null;
    app.fileError = "";
    $("#archive").addEventListener("change", event => chooseFile(event.target.files?.[0]));
    const zone = $("#upload-zone");
    for (const name of ["dragenter", "dragover"]) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("dragging"); });
    for (const name of ["dragleave", "drop"]) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("dragging"); });
    zone.addEventListener("drop", event => chooseFile(event.dataTransfer.files?.[0]));
    $("#create-form").addEventListener("submit", createFilm);
    $("#target-minutes").addEventListener("input", syncRuntimePresets);
    syncRuntimePresets();
    syncCreateConnection();
  }
  function syncRuntimePresets() {
    document.querySelectorAll("[data-minutes]").forEach(button => {
      const selected = Number(button.dataset.minutes) === Number($("#target-minutes")?.value);
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }
  function chooseFile(file) {
    if (!file) return;
    app.selectedFile = file;
    syncSelectedFile();
    if (!app.file) return;
    if (!$("#project-name").value.trim()) $("#project-name").value = file.name.replace(/\.zip$/i, "").replace(/[_.]/g, " ").slice(0, 160);
  }
  function uploadArchive(form) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", apiUrl("/api/jobs"));
      xhr.setRequestHeader("Authorization", `Bearer ${app.token}`);
      xhr.upload.addEventListener("progress", event => {
        if (!event.lengthComputable) return;
        const percent = Math.round(event.loaded / event.total * 100);
        const bar = $("#upload-bar");
        if (bar) {
          bar.style.width = `${percent}%`;
          bar.parentElement.setAttribute("aria-valuenow", percent);
          $("#upload-percent").textContent = `${percent}%`;
          if (percent === 100) $("#upload-label").textContent = "Upload complete. Preparing your project…";
        }
      });
      xhr.addEventListener("load", () => {
        let payload;
        try { payload = JSON.parse(xhr.responseText); } catch { reject(new Error("The workspace sent an unreadable response. Refresh your films before uploading again.")); return; }
        if (xhr.status < 200 || xhr.status >= 300) { reject(new Error(responseError(payload, xhr.status))); return; }
        resolve(payload);
      });
      xhr.addEventListener("error", () => reject(new Error("The upload connection was interrupted. Check your internet connection and refresh your films before trying again.")));
      xhr.send(form);
    });
  }
  async function createFilm(event) {
    event.preventDefault();
    if (app.busy) return;
    if (app.fileError) { setInlineError("#create-error", app.fileError); $("#archive").focus(); return; }
    if (!app.connected) { openConnection(); return; }
    if (!app.file) { setInlineError("#create-error", "Choose your episode ZIP first."); $("#archive").focus(); return; }
    if (archiveError(app.file)) { syncSelectedFile(); return; }
    const minutes = Number($("#target-minutes").value);
    if (!Number.isFinite(minutes) || minutes <= 0) { setInlineError("#create-error", "Choose a film length greater than zero."); return; }
    const form = new FormData();
    form.append("archive", app.file);
    form.append("target_minutes", String(minutes));
    form.append("name", $("#project-name").value.trim());
    app.busy = true;
    const button = $("#create-submit");
    $("#create-form").querySelectorAll("input, button").forEach(element => { element.disabled = true; });
    button.disabled = true;
    button.textContent = "Uploading episodes…";
    $("#upload-progress").hidden = false;
    setInlineError("#create-error", "");
    try {
      const detail = await uploadArchive(form);
      if (!detail?.state?.job_id) throw new Error("The workspace did not return a project ID. Check Your films before trying again.");
      app.detail = detail;
      updateJobList(detail.state);
      app.file = null;
      app.selectedFile = null;
      app.fileError = "";
      location.hash = `film/${encodeURIComponent(detail.state.job_id)}`;
      announce("Your episodes are in. The editing room is at work.");
    } catch (error) {
      setInlineError("#create-error", error.message);
      $("#create-form")?.querySelectorAll("input, button").forEach(element => { element.disabled = false; });
      if (button.isConnected) { button.disabled = false; button.innerHTML = 'Try creating again <span aria-hidden="true">↗</span>'; }
    } finally { app.busy = false; }
  }
  function updateJobList(state) {
    const index = app.jobs.findIndex(job => job.job_id === state.job_id);
    if (index >= 0) app.jobs[index] = state; else app.jobs.unshift(state);
    syncConnection();
  }
  function filmCards(search = "") {
    const filtered = app.jobs.filter(job => (job.params?.name || job.job_id).toLowerCase().includes(search.toLowerCase()));
    if (!filtered.length) return `<div class="empty-state"><h2>No films match that name.</h2><p>Try another search or create a new film.</p></div>`;
    return `<div class="film-grid">${filtered.map(job => {
      const [status, type] = statusFor(job, app.detail?.state?.job_id === job.job_id ? app.detail : null);
      const title = job.params?.name || "Untitled film";
      const progress = Math.round(overallFraction(job) * 100);
      const episodes = job.stages?.ingest?.episode_count;
      return `<a class="film-card" href="#film/${encodeURIComponent(job.job_id)}" aria-label="Open ${escape(title)}, ${escape(status)}"><div class="card-cover"><span aria-hidden="true">${escape(title.charAt(0).toUpperCase())}</span><span class="pill ${type}">${escape(status)}</span></div><div class="card-body"><h3>${escape(title)}</h3><div class="card-meta"><span>${escape(minuteLabel(job.params?.target_seconds))} target</span>${episodes ? `<span>· ${episodes} episodes</span>` : ""}</div><div class="card-footer"><span>${status === "Ready" ? "Watch your film" : `${progress}% complete`}</span><span aria-hidden="true">↗</span></div></div></a>`;
    }).join("")}</div>`;
  }
  async function renderFilms(serial) {
    $("#main").innerHTML = `<div class="page-heading"><div><span class="eyebrow">YOUR EDITING WORKSPACE</span><h1>Your films.</h1><p>Every story has a place here.</p></div><a class="button primary" href="#new">＋ New film</a></div><div id="library-content"></div>`;
    if (!app.connected) {
      $("#library-content").innerHTML = `<section class="empty-state"><div class="empty-icon" aria-hidden="true">S</div><h2>Your stories live in<br>your workspace.</h2><p>Connect your workspace to find your films, follow an edit, or start your next story.</p><button class="button primary" type="button" data-action="connect">Connect workspace <span aria-hidden="true">↗</span></button></section>`;
      return;
    }
    $("#library-content").innerHTML = '<div class="loading-state"><span class="loading-spinner" aria-hidden="true"></span> Finding your films…</div>';
    try {
      await refreshJobs();
      if (serial !== app.renderSerial) return;
      if (!app.jobs.length) {
        $("#library-content").innerHTML = `<section class="empty-state"><div class="empty-icon" aria-hidden="true">S</div><h2>Your first film<br>starts with a season.</h2><p>Bring a ZIP of your episodes, choose your runtime, and let the editing room find the story.</p><a class="button primary" href="#new">Create your first film <span aria-hidden="true">↗</span></a></section>`;
        return;
      }
      $("#library-content").innerHTML = '<div class="library-toolbar"><label class="search-field" style="margin:0"><span class="sr-only">Search films by name</span><input id="film-search" type="search" placeholder="Find a film…" aria-label="Search films by name"></label><button class="button ghost" type="button" data-action="refresh-library" aria-label="Refresh films">↻</button></div><div id="film-cards"></div>';
      $("#film-cards").innerHTML = filmCards();
      $("#film-search").addEventListener("input", event => { $("#film-cards").innerHTML = filmCards(event.target.value); });
    } catch (error) {
      if (serial !== app.renderSerial) return;
      $("#library-content").innerHTML = `<div class="alert-panel" role="alert"><p>${escape(error.message)}</p><button class="button secondary" type="button" data-action="refresh-library">Try again</button><button class="button ghost" type="button" data-action="connect">Connection settings</button></div>`;
    }
  }
  async function renderFilm(jobId, serial) {
    if (!app.connected) { $("#main").innerHTML = '<a class="back-link" href="#films">← Your films</a><h1>Connect your<br>editing room.</h1>' + connectionCallout(); return; }
    $("#main").innerHTML = '<div class="loading-state"><span class="loading-spinner" aria-hidden="true"></span> Opening your film…</div>';
    try {
      const detail = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (serial !== app.renderSerial) return;
      app.detail = detail;
      updateJobList(detail.state);
      const state = detail.state;
      const complete = state.stages?.render?.status === "done";
      $("#main").innerHTML = `
        <a class="back-link" href="#films">← Your films</a><div class="film-heading"><div><span class="eyebrow">YOUR EDITING WORKSPACE</span><h1>${escape(state.params?.name || "Untitled film")}</h1><div class="film-subtitle"><span>${escape(dateLabel(state.created_at))}</span><span>·</span><span>Your season edit</span></div></div><span id="film-status" class="pill"></span></div>
        <div class="metrics" aria-label="Film details"><div class="metric"><small>EPISODES</small><strong id="metric-episodes">—</strong></div><div class="metric"><small>TARGET LENGTH</small><strong id="metric-target">—</strong></div><div class="metric"><small>FILM LENGTH</small><strong id="metric-length">—</strong></div><div class="metric"><small>STORY CLIPS</small><strong id="metric-clips">—</strong></div></div>
        <div id="job-alert" hidden></div><div class="film-layout"><section class="panel player-panel"><div class="panel-heading"><h3>Your film</h3><small>WATCH & SAVE</small></div><div class="video-container"><div id="player-placeholder" class="player-placeholder"><span class="play-outline" aria-hidden="true">▷</span><strong>A story worth waiting for.</strong><p>Your finished film will appear here. Your progress is saved along the way.</p></div><video id="film-video" controls playsinline preload="metadata" aria-label="Condensed movie player" hidden></video></div><div class="player-meta"><span id="film-format">MP4 · with subtitles</span><span id="player-ready"></span></div><div class="download-actions" id="download-actions" hidden><a id="download-movie" class="button primary" download>↓ Download film</a><a id="download-subtitles" class="button secondary" download>↓ Subtitles · SRT</a></div><p id="playback-help" class="playback-help" hidden></p><button id="retry-media" class="button ghost full-width" type="button" data-action="retry-media" hidden>Reload film</button><div id="refine-area" class="refine-area" hidden><h3>Make the story yours</h3><p>Missing an important moment? Keep it by its original timestamp and create a refined edit.</p><button id="open-refine" class="button secondary" type="button" data-action="refine">Refine this film <span aria-hidden="true">↗</span></button><p id="protected-caption" class="protected-caption" hidden></p></div></section>
        <div><details class="panel journey-panel" id="journey" ${complete ? "" : "open"}><summary class="journey-summary"><h3>The journey</h3><span id="journey-percent"></span></summary><div class="journey-total"><div class="progress-caption"><span id="journey-message"></span><span>6 stages</span></div><div class="progress-track" role="progressbar" aria-label="Film processing" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="journey-progress"><span id="journey-bar" style="width:0%"></span></div></div><ol class="journey-list">${STAGES.map(([key, label, description], index) => `<li class="journey-step" data-stage="${key}"><span class="step-mark">${String(index + 1).padStart(2, "0")}</span><div><div class="step-heading">${label}</div><div class="step-description">${description}</div></div></li>`).join("")}</ol></details><p class="processing-note" id="processing-note"></p><div class="job-actions" id="job-actions"></div></div></div>`;
      $("#film-video").addEventListener("loadedmetadata", () => {
        const video = $("#film-video");
        if (video && Number.isFinite(video.duration)) {
          $("#metric-length").textContent = minuteLabel(video.duration);
          $("#film-format").textContent = `${minuteLabel(video.duration)} · MP4`;
        }
      });
      $("#film-video").addEventListener("error", () => {
        $("#playback-help").textContent = "The film could not load. Reconnect if your access expired, or reload the film.";
        $("#playback-help").hidden = false;
        $("#retry-media").hidden = false;
      });
      patchFilm(detail);
      document.title = `${state.params?.name || "Your film"} — Season to Screen`;
      schedulePoll();
    } catch (error) {
      if (serial !== app.renderSerial) return;
      $("#main").innerHTML = `<a class="back-link" href="#films">← Your films</a><div class="alert-panel" role="alert"><p>${escape(error.message)}</p><button class="button secondary" type="button" data-action="reload-project">Try again</button><button class="button ghost" type="button" data-action="connect">Connection settings</button></div>`;
    }
  }
  function stageDescription(key, stage, detail) {
    if (stage?.status === "done") return "Complete";
    if (stage?.status === "failed") return "Saved progress · ready to retry";
    if (key === "nlp" && stage?.llm?.active_episode) {
      const llm = stage.llm;
      return `${llm.active_episode} · ${llm.completed_scenes?.length || 0} of ${llm.scene_count || 0} story blocks`;
    }
    if (key === "transcribe" && stage?.current) {
      const names = { extracting_subtitles: "Reading subtitles", extracting_audio: "Preparing audio", loading_model: "Preparing dialogue recognition", decoding: "Recognizing dialogue" };
      return `${stage.current.episode} · ${names[stage.current.activity] || "Capturing dialogue"}`;
    }
    const total = (stage?.completed?.length || 0) + (stage?.pending?.length || 0);
    if (total && stage?.status === "in_progress") return `${stage.completed?.length || 0} of ${total} ${key === "render" ? "clips" : "episodes"}`;
    if (stage?.status === "in_progress") return detail.running ? "Working on this stage" : "Progress saved";
    return STAGES.find(([name]) => name === key)?.[2] || "Waiting";
  }
  function patchFilm(detail) {
    if (!$("#film-status")) return;
    app.detail = detail;
    const state = detail.state;
    updateJobList(state);
    const [status, type] = statusFor(state, detail);
    const done = status === "Ready";
    const fraction = Math.round(overallFraction(state) * 100);
    $("#film-status").textContent = status;
    $("#film-status").className = `pill ${type}`;
    $("#metric-episodes").textContent = state.stages?.ingest?.episode_count || "—";
    $("#metric-target").textContent = minuteLabel(state.params?.target_seconds);
    const video = $("#film-video");
    if (!video?.duration || !Number.isFinite(video.duration)) $("#metric-length").textContent = minuteLabel(detail.selection?.selected_total_seconds);
    $("#metric-clips").textContent = detail.selection?.stats?.scenes_kept ?? "—";
    $("#journey-percent").textContent = done ? "All complete" : `${fraction}%`;
    $("#journey-message").textContent = done ? "Your story is ready" : detail.stopping ? "Saving progress before stopping" : detail.running ? "The editing room is at work" : "Your progress is saved";
    $("#journey-bar").style.width = `${fraction}%`;
    $("#journey-progress").setAttribute("aria-valuenow", fraction);
    STAGES.forEach(([key], index) => {
      const row = $(`[data-stage="${key}"]`);
      const stage = state.stages?.[key];
      row.className = `journey-step ${stage?.status || "pending"}`;
      $(".step-mark", row).textContent = stage?.status === "done" ? "✓" : String(index + 1).padStart(2, "0");
      $(".step-description", row).textContent = stageDescription(key, stage, detail);
    });
    $("#processing-note").textContent = done ? "Your film is saved in your workspace. Watch it here, or download it to take the story with you." : detail.stopping ? "The current operation may need to finish before stopping. Completed work will stay saved." : detail.running ? "You can leave this page. Processing continues on your workspace server, and your progress is saved." : "Resume from your saved progress whenever you’re ready.";
    $("#job-actions").innerHTML = done ? "" : detail.running || detail.stopping ? `<button class="button ghost" type="button" data-action="stop" ${detail.stopping || app.busy ? "disabled" : ""}>${detail.stopping ? "Stopping…" : "Stop processing"}</button>` : `<button class="button primary" type="button" data-action="resume" ${app.busy ? "disabled" : ""}>Resume edit <span aria-hidden="true">▷</span></button>`;
    const error = detail.error || (type === "attention" ? state.errors?.at(-1)?.message : "");
    const alert = $("#job-alert");
    alert.hidden = !error;
    if (error) alert.innerHTML = `<div class="alert-panel" role="alert"><p>Your edit needs attention. Completed work is saved.</p><p>${escape(error)}</p><button class="button secondary" type="button" data-action="resume" ${app.busy || detail.running || detail.stopping ? "disabled" : ""}>Resume edit</button></div>`;
    $("#refine-area").hidden = !done;
    const ranges = state.params?.must_keep_ranges || [];
    $("#protected-caption").hidden = !ranges.length;
    $("#protected-caption").textContent = `${ranges.length} specific ${ranges.length === 1 ? "moment" : "moments"} protected in this edit.`;
    $("#open-refine").disabled = app.busy;
    if (done && app.mediaJob !== state.job_id) loadMedia(state.job_id);
  }
  function clearMedia() {
    app.mediaJob = "";
    app.ticketExpires = 0;
    app.mediaPromise = null;
    const video = $("#film-video");
    if (video) { video.pause(); video.removeAttribute("src"); video.load(); }
  }
  function mediaAddress(value) {
    const url = new URL(value, `${app.api}/`);
    if (!["http:", "https:"].includes(url.protocol) || url.origin !== new URL(app.api).origin || url.username || url.password) {
      throw new Error("The workspace sent an invalid film address.");
    }
    return url.href;
  }
  async function loadMedia(jobId, force = false) {
    if (app.mediaPromise) return app.mediaPromise;
    if (!force && app.mediaJob === jobId) return;
    const connectionApi = app.api;
    const video = $("#film-video");
    const previousTime = force && video ? video.currentTime : 0;
    app.mediaPromise = (async () => {
      try {
        const ticket = await request(`/api/jobs/${encodeURIComponent(jobId)}/media-ticket`, { method: "POST" });
        if (app.detail?.state?.job_id !== jobId || !$("#film-video") || !app.connected || app.api !== connectionApi) return;
        const movieUrl = mediaAddress(ticket.movie_url);
        const subtitlesUrl = mediaAddress(ticket.subtitles_url);
        app.mediaJob = jobId;
        app.ticketExpires = Date.now() + Number(ticket.expires_in || 28800) * 1000;
        $("#film-video").src = movieUrl;
        $("#film-video").hidden = false;
        $("#player-placeholder").hidden = true;
        const movieDownload = new URL(movieUrl);
        const subtitlesDownload = new URL(subtitlesUrl);
        movieDownload.searchParams.set("download", "1");
        subtitlesDownload.searchParams.set("download", "1");
        $("#download-movie").href = movieDownload.href;
        $("#download-subtitles").href = subtitlesDownload.href;
        $("#download-actions").hidden = false;
        $("#player-ready").textContent = "Ready to watch";
        $("#playback-help").hidden = true;
        $("#retry-media").hidden = true;
        if (previousTime) $("#film-video").addEventListener("loadedmetadata", event => { event.target.currentTime = Math.min(previousTime, event.target.duration || previousTime); }, { once: true });
      } catch (error) {
        if (app.detail?.state?.job_id !== jobId || !$("#playback-help")) return;
        $("#playback-help").textContent = error.message;
        $("#playback-help").hidden = false;
        $("#retry-media").hidden = false;
      } finally { app.mediaPromise = null; }
    })();
    return app.mediaPromise;
  }
  function schedulePoll() {
    clearTimeout(app.poll);
    if (!app.connected || !app.route.startsWith("film/") || !(app.detail?.running || app.detail?.stopping)) return;
    app.poll = setTimeout(pollFilm, document.hidden ? 12000 : 3000);
  }
  async function pollFilm() {
    if (app.pollBusy || !app.connected) return;
    if (app.busy) { schedulePoll(); return; }
    const id = app.detail?.state?.job_id;
    const route = app.route;
    if (!id || !route.startsWith("film/")) return;
    app.pollBusy = true;
    try {
      const detail = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (app.route !== route) return;
      patchFilm(detail);
      schedulePoll();
    } catch (error) {
      if (app.route !== route) return;
      $("#job-alert").hidden = false;
      $("#job-alert").innerHTML = `<div class="alert-panel" role="alert"><p>Live progress could not refresh. Your server may still be processing.</p><p>${escape(error.message)}</p><button class="button secondary" type="button" data-action="retry-progress">Retry live progress</button></div>`;
      clearTimeout(app.poll);
    } finally { app.pollBusy = false; }
  }
  async function jobAction(action) {
    if (app.busy || !app.detail) return;
    const id = app.detail.state.job_id;
    app.busy = true;
    patchFilm(app.detail);
    try {
      const detail = await request(`/api/jobs/${encodeURIComponent(id)}/${action}`, { method: "POST" });
      if (app.route !== `film/${id}`) return;
      app.busy = false;
      patchFilm(detail);
      schedulePoll();
      announce(action === "stop" ? "Stop requested. Completed work will be saved." : "Your edit is resuming from saved progress.");
    } catch (error) { announce(error.message); }
    finally { app.busy = false; if (app.detail?.state?.job_id === id && $("#film-status")) patchFilm(app.detail); }
  }
  function addRange(range = null) {
    const episodes = app.detail?.state?.stages?.ingest?.episodes || [];
    const fallback = Array.from({ length: app.detail?.state?.stages?.ingest?.episode_count || 1 }, (_, index) => ({ episode: `E${String(index + 1).padStart(2, "0")}` }));
    const knownEpisodes = episodes.length ? episodes : fallback;
    const row = document.createElement("div");
    row.className = "range-row";
    const index = Math.random().toString(36).slice(2, 10);
    row.innerHTML = `<div><label for="ep-${index}">Episode</label><select id="ep-${index}" name="episode">${knownEpisodes.map(item => `<option value="${escape(item.episode)}" ${item.episode === range?.episode ? "selected" : ""}>${escape(item.episode)}</option>`).join("")}</select></div><div><label for="start-${index}">Start · min</label><input id="start-${index}" name="start" type="number" min="0" step="any" inputmode="decimal" value="${range ? range.start / 60 : ""}" placeholder="3.5" required></div><div><label for="end-${index}">End · min</label><input id="end-${index}" name="end" type="number" min="0" step="any" inputmode="decimal" value="${range ? range.end / 60 : ""}" placeholder="6.5" required></div><button type="button" class="icon-button" aria-label="Remove this protected moment">×</button>`;
    $("button", row).addEventListener("click", () => row.remove());
    $("#range-list").append(row);
  }
  function openRefine() {
    if (!app.detail) return;
    $("#range-list").innerHTML = "";
    const ranges = app.detail.state.params?.must_keep_ranges || [];
    ranges.forEach(addRange);
    setInlineError("#refine-error", "");
    $("#refine-dialog").showModal();
  }
  $("#add-range").addEventListener("click", () => addRange());
  $("#refine-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (app.busy || !app.detail) return;
    const id = app.detail.state.job_id;
    const ranges = [...document.querySelectorAll(".range-row")].map(row => ({ episode: $('[name="episode"]', row).value, start: Number($('[name="start"]', row).value) * 60, end: Number($('[name="end"]', row).value) * 60 }));
    const durations = new Map((app.detail.state.stages?.ingest?.episodes || []).map(item => [item.episode, item.duration]));
    if (ranges.some(range => !Number.isFinite(range.start) || !Number.isFinite(range.end) || range.start < 0 || range.end <= range.start || (durations.has(range.episode) && range.end > durations.get(range.episode)))) {
      setInlineError("#refine-error", "Each moment needs an end after its start, within the original episode’s length.");
      return;
    }
    app.busy = true;
    $("#refine-submit").disabled = true;
    $("#refine-submit").textContent = "Starting your new edit…";
    setInlineError("#refine-error", "");
    try {
      const detail = await request(`/api/jobs/${encodeURIComponent(id)}/refine`, { method: "POST", body: { must_keep_ranges: ranges } });
      if (!detail?.state?.job_id) throw new Error("The workspace did not return your refined project. Check Your films before trying again.");
      app.detail = detail;
      updateJobList(detail.state);
      $("#refine-dialog").close();
      location.hash = `film/${encodeURIComponent(detail.state.job_id)}`;
      announce("Your refined edit is underway. The previous film is preserved.");
    } catch (error) { setInlineError("#refine-error", error.message); }
    finally { app.busy = false; $("#refine-submit").disabled = false; $("#refine-submit").innerHTML = 'Create refined film <span aria-hidden="true">↗</span>'; }
  });
  $("#main").addEventListener("click", event => {
    const button = event.target.closest("[data-action], [data-minutes]");
    if (!button || button.disabled) return;
    if (button.dataset.minutes) { $("#target-minutes").value = button.dataset.minutes; syncRuntimePresets(); return; }
    const action = button.dataset.action;
    if (action === "connect") openConnection();
    if (action === "refresh-library" || action === "reload-project") renderRoute();
    if (action === "retry-progress") pollFilm();
    if (action === "stop" || action === "resume") jobAction(action);
    if (action === "refine") openRefine();
    if (action === "retry-media") loadMedia(app.detail.state.job_id, true);
  });
  async function renderRoute() {
    clearTimeout(app.poll);
    const route = location.hash.replace(/^#/, "") || "new";
    app.route = /^(new|films|film\/[A-Za-z0-9_-]{1,64})$/.test(route) ? route : "new";
    const serial = ++app.renderSerial;
    clearMedia();
    document.querySelectorAll("[data-nav]").forEach(link => {
      const selected = link.dataset.nav === (app.route === "new" ? "new" : "films");
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    $("#breadcrumb").innerHTML = `WORKSPACE <span>/</span> ${app.route === "new" ? "NEW FILM" : app.route === "films" ? "YOUR FILMS" : "THE EDITING ROOM"}`;
    document.title = `${app.route === "new" ? "New film" : app.route === "films" ? "Your films" : "The editing room"} — Season to Screen`;
    if (app.route === "new") renderNew();
    else if (app.route === "films") await renderFilms(serial);
    else await renderFilm(app.route.slice(5), serial);
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  window.addEventListener("hashchange", () => {
    if (app.busy && $("#upload-progress") && !$("#upload-progress").hidden) {
      location.replace("#new");
      announce("Your upload is still running. Keep this page open until it finishes.");
      return;
    }
    renderRoute();
  });
  window.addEventListener("beforeunload", event => {
    if (app.busy) { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) schedulePoll(); });
  syncConnection();
  renderRoute();
})();

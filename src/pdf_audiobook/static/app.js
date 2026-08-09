(() => {
  "use strict";

  const views = ["add", "configure", "progress"];
  const activeGenerationStates = new Set(["starting", "synthesizing", "cancelling", "assembling", "encoding", "verifying", "publishing"]);
  const status = document.querySelector("#add-status");
  const analyzeButton = document.querySelector(".analyze");
  const input = document.querySelector("#pdf-input");
  const existing = document.querySelector("#existing-state");
  let analysis = null;
  let chapterPlan = null;
  let conversionId = null;
  let selectedFile = null;
  let planRequestInFlight = false;
  let labelsDirty = false;
  let generationRequestInFlight = false;
  let pollTimer = null;
  let elapsedTimer = null;
  let activePreview = null;
  let previewRequest = 0;
  let recoveredState = null;
  let recoveredRoute = false;
  let structuralPlanLocked = false;
  const typeIsInteger = (value) => typeof value === "number" && Number.isInteger(value);

  const currentPlanSpec = () => ({
    mode: document.querySelector("input[name=plan-mode]:checked")?.value,
    count: Number(document.querySelector("#chapter-count")?.value),
  });

  const currentChapterRange = () => {
    const start = Number(document.querySelector("#chapter-start")?.value);
    const end = Number(document.querySelector("#chapter-end")?.value);
    const total = chapterPlan?.chapters?.length || 0;
    return { start, end, total, valid: total > 0 && Number.isInteger(start) && Number.isInteger(end) && start >= 1 && end >= start && end <= total };
  };

  const updateChapterRange = () => {
    const controls = [document.querySelector("#chapter-start"), document.querySelector("#chapter-end")];
    const range = currentChapterRange();
    controls.forEach((control) => control?.setAttribute("aria-invalid", range.valid ? "false" : "true"));
    const target = document.querySelector("#chapter-range-status");
    if (!range.total) target.textContent = "Select an inclusive chapter range after generating a plan.";
    else if (!range.valid) target.textContent = `Choose an inclusive range from 1 through ${range.total}.`;
    else target.textContent = `${range.end - range.start + 1} chapter${range.end - range.start === 0 ? "" : "s"} selected (inclusive).`;
    const ready = Boolean(chapterPlan && planMatchesSelection(chapterPlan) && range.valid && !planRequestInFlight);
    document.querySelector(".next").disabled = !ready;
    return range;
  };

  const planMatchesSelection = (plan) => {
    const selection = currentPlanSpec();
    return Boolean(plan && plan.mode === selection.mode && (plan.mode !== "custom" || plan.requested_count === selection.count));
  };

  const setPlanControlsDisabled = (disabled) => {
    document.querySelectorAll("input[name=plan-mode], #chapter-count, #regenerate-plan").forEach((control) => { control.disabled = disabled || structuralPlanLocked; });
    if (disabled) document.querySelector("#save-labels").disabled = true;
  };

  const setStructuralPlanLocked = (locked) => {
    structuralPlanLocked = locked;
    setPlanControlsDisabled(planRequestInFlight);
    if (locked) document.querySelector("#plan-status").textContent = "Chapter boundaries are locked for resumable generation; labels remain editable.";
  };

  const invalidatePlan = (message) => {
    chapterPlan = null;
    document.querySelector(".next").disabled = true;
    document.querySelector("#save-labels").disabled = true;
    if (message) document.querySelector("#plan-status").textContent = message;
  };

  const resetExistingJob = () => {
    recoveredRoute = false;
    setStructuralPlanLocked(false);
    setGenerationControlsDisabled(false);
    analyzeButton.disabled = !selectedFile;
    existing.hidden = true;
    document.querySelector("#existing-title").textContent = "Current job";
    document.querySelector("#existing-state-label").textContent = "";
    document.querySelector("#existing-message").textContent = "";
    document.querySelector("#view-current-job").hidden = true;
    document.querySelector("#cancel-current-generation").hidden = true;
    document.querySelector("#cancel-current-generation").disabled = true;
    document.querySelector(".delete-job").hidden = true;
    document.querySelector(".delete-job").disabled = true;
  };

  const updateExistingJob = (body, state, message) => {
    analyzeButton.disabled = true;
    const active = activeGenerationStates.has(state);
    const invalid = state === "invalid";
    const unknown = state === "unknown";
    const title = body?.analysis?.title || body?.job?.original_display_filename || "Current job";
    const stateText = state ? state.replaceAll("_", " ") : "status unavailable";
    const defaultMessage = active ? "A generation is active. View progress or cancel it; the worker keeps completed chunks." : invalid ? "The saved job is invalid. Delete it explicitly before starting another." : state === "completed" ? "Generation is complete." : unknown ? "Status could not be refreshed. View Current Job to try again." : "The current job is ready to review.";
    document.querySelector("#existing-title").textContent = title;
    document.querySelector("#existing-state-label").textContent = stateText;
    document.querySelector("#existing-message").textContent = message || defaultMessage;
    existing.hidden = false;
    const view = document.querySelector("#view-current-job");
    const cancel = document.querySelector("#cancel-current-generation");
    const deleteButton = document.querySelector(".delete-job");
    document.querySelectorAll(".chapter-label").forEach((control) => { control.disabled = active; });
    const saveButton = document.querySelector("#save-labels");
    saveButton.disabled = active || planRequestInFlight || !labelsDirty || !chapterPlan || !planMatchesSelection(chapterPlan);
    view.hidden = invalid;
    view.disabled = false;
    cancel.hidden = !active;
    cancel.disabled = !active || state === "cancelling";
    deleteButton.hidden = active || unknown;
    deleteButton.disabled = active || unknown;
  };

  const show = (name) => {
    if (name === "configure" && activeGenerationStates.has(recoveredState)) {
      if (analysis && chapterPlan) show("progress");
      document.querySelector("#progress-status").textContent = "Generation is active. Cancel generation before editing chapter labels.";
      return;
    }
    if ((name === "configure" || name === "progress") && !analysis) {
      status.textContent = "Analyze a PDF successfully before opening review or generation.";
      return;
    }
    if (name === "progress" && !chapterPlan) {
      document.querySelector("#plan-status").textContent = "Generate and review a chapter plan before continuing.";
      return;
    }
    views.forEach((view) => {
      const panel = document.querySelector(`#view-${view}`);
      const step = document.querySelector(`.step[data-view="${view}"]`);
      const active = view === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
      step.classList.toggle("active", active);
      if (active) step.setAttribute("aria-current", "step"); else step.removeAttribute("aria-current");
    });
    history.replaceState(null, "", `#${name}`);
    document.querySelector(`#view-${name} h1`)?.focus?.();
  };

  const renderAnalysis = (value) => {
    analysis = value;
    document.querySelector("#analysis-metadata").textContent = `${value.title} - ${value.page_count} pages - ${value.word_count} words - ${value.detected_language} - about ${value.estimated_duration_minutes} minutes`;
    const warnings = document.querySelector("#analysis-warnings");
    warnings.replaceChildren();
    (value.warnings && value.warnings.length ? value.warnings : ["No warnings reported"]).forEach((warning) => {
      const li = document.createElement("li"); li.textContent = warning; warnings.append(li);
    });
    document.querySelector("#cleaned-preview").textContent = value.preview || "No cleaned preview available.";
    const candidates = document.querySelector("#chapter-candidates");
    candidates.replaceChildren();
    (value.chapter_candidates || []).forEach((candidate) => {
      const li = document.createElement("li"); li.textContent = `${candidate.title} (page ${candidate.source_page || "?"})`; candidates.append(li);
    });
    if (!candidates.children.length) { const li = document.createElement("li"); li.textContent = "No candidates detected."; candidates.append(li); }
  };

  const renderPlan = (plan, resetRange = false) => {
    const hadPlan = Boolean(chapterPlan);
    chapterPlan = plan;
    labelsDirty = false;
    document.querySelectorAll("input[name=plan-mode]").forEach((control) => { control.checked = control.value === plan.mode; });
    const count = document.querySelector("#chapter-count");
    count.value = plan.requested_count || count.value;
    const total = plan.chapters?.length || 0;
    const startControl = document.querySelector("#chapter-start");
    const endControl = document.querySelector("#chapter-end");
    startControl.max = String(total); endControl.max = String(total);
    if (resetRange || !hadPlan || !currentChapterRange().valid) { startControl.value = "1"; endControl.value = String(total); }
    document.querySelector(".custom-count").hidden = plan.mode !== "custom";
    const warnings = document.querySelector("#plan-warnings"); warnings.replaceChildren();
    (plan.warnings || []).forEach((warning) => { const li = document.createElement("li"); li.textContent = warning; warnings.append(li); });
    const list = document.querySelector("#chapter-plan"); list.replaceChildren();
    (plan.chapters || []).forEach((chapter) => {
      const li = document.createElement("li"); li.className = "planned-chapter";
      const label = document.createElement("input");
      label.className = "chapter-label"; label.type = "text"; label.maxLength = 200; label.value = chapter.title;
      label.setAttribute("aria-label", `Chapter ${chapter.index} label`);
      label.addEventListener("input", () => { labelsDirty = true; document.querySelector("#save-labels").disabled = false; });
      li.append(label);
      const meta = document.createElement("small"); meta.textContent = `pages ${chapter.start_page}-${chapter.end_page} - ${chapter.word_count} words - ${chapter.source_type}`; li.append(meta);
      list.append(li);
    });
    const validForSelection = planMatchesSelection(plan);
    document.querySelector("#save-labels").disabled = !validForSelection || planRequestInFlight || !labelsDirty;
    document.querySelector("#plan-status").textContent = structuralPlanLocked ? "Chapter boundaries are locked for resumable generation; labels remain editable." : validForSelection ? `${plan.chapters.length} chapter${plan.chapters.length === 1 ? "" : "s"} planned.` : "This plan is stale. Generate a plan for the selected mode.";
    updateChapterRange();
  };

  const readError = async (response) => {
    try { const body = await response.json(); return body.error || { code: "REQUEST_FAILED", message: "The request failed." }; }
    catch (_) { return { code: "REQUEST_FAILED", message: "The request failed." }; }
  };

  const requestPlan = async (mode, count) => {
    const payload = { mode }; if (mode === "custom") payload.count = Number(count);
    const response = await fetch("/api/chapter-plan", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify(payload) });
    if (!response.ok) { const error = await readError(response); throw new Error(`${error.code}: ${error.message}${error.recommended_maximum ? ` Try ${error.recommended_maximum} or fewer.` : ""}`); }
    const body = await response.json(); conversionId = body.conversion_id || conversionId; renderPlan(body.chapter_plan, true); return body;
  };

  const runPlanRequest = async (mode, count) => {
    if (planRequestInFlight) return false;
    if (structuralPlanLocked) return false;
    planRequestInFlight = true;
    invalidatePlan(`Generating ${mode === "whole" ? "the Whole Book" : mode === "original" ? "the Original Chapters" : "your custom"} plan...`);
    setPlanControlsDisabled(true);
    try {
      const body = await requestPlan(mode, count);
      if (!planMatchesSelection(body.chapter_plan)) { invalidatePlan("The returned plan no longer matches the selected mode. Generate it again."); return false; }
      document.querySelector("#plan-status").textContent = `${body.chapter_plan.chapters.length} chapter${body.chapter_plan.chapters.length === 1 ? "" : "s"} planned.`;
      return true;
    } catch (error) { invalidatePlan(error.message || "Chapter planning failed."); return false; }
    finally {
      planRequestInFlight = false; setPlanControlsDisabled(false);
      if (chapterPlan && planMatchesSelection(chapterPlan)) { document.querySelector("#save-labels").disabled = !labelsDirty; document.querySelector(".next").disabled = false; }
    }
  };

  const generateOriginal = async () => runPlanRequest("original");

  const upload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) { status.textContent = "Please choose a PDF file."; return; }
    analyzeButton.disabled = true; status.textContent = `${file.name} is uploading and being analyzed locally...`;
    try {
      const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/pdf", "X-PDF-Filename": encodeURIComponent(file.name), "Origin": location.origin }, body: file });
      if (!response.ok) {
        const error = await readError(response);
        if (error.code === "ACTIVE_JOB") {
          const refreshed = await refreshCurrentJob("An active conversion already exists. Review the current job below.");
          const factualState = recoveredState ? recoveredState.replaceAll("_", " ") : "available";
          status.textContent = refreshed ? `An existing job is ${factualState}. Use View Current Job to inspect it.` : recoveredState === "no_active" ? "The prior job is no longer active. Try Analyze PDF again." : "An active conversion already exists. Use View Current Job to retry status refresh.";
          return;
        }
        if (error.conversion_id) { conversionId = error.conversion_id; updateExistingJob({ conversion_id: conversionId }, "failed", "Analysis failed. Delete the failed job explicitly before starting another."); }
        throw new Error(`${error.code}: ${error.message}`);
      }
      const body = await response.json(); conversionId = body.conversion_id; setStructuralPlanLocked(false); updateExistingJob(body, "analyzed", "Analysis is complete. Review the job before generating."); invalidatePlan("Analysis complete. Generating the Original Chapters plan..."); renderAnalysis(body.analysis); status.textContent = "Analysis complete. Generating the Original Chapters plan..."; show("configure");
      if (await generateOriginal()) { updateExistingJob(body, "planned", "A planned job is ready to review."); status.textContent = "Analysis and chapter planning complete. Review the labels before continuing."; }
      else { updateExistingJob(body, "analyzed", "Analysis is complete. Chapter planning needs attention."); status.textContent = "Analysis complete. Chapter planning needs attention in the Review panel."; }
    } catch (error) { status.textContent = error instanceof Error && error.message.includes("OCR_REQUIRED") ? "OCR_REQUIRED: this PDF has no usable selectable text." : String(error.message || "Analysis failed."); }
    finally { analyzeButton.disabled = !selectedFile || !existing.hidden || Boolean(recoveredState && recoveredState !== "no_active"); }
  };

  const setGenerationControlsDisabled = (disabled) => {
    document.querySelectorAll("#view-configure input, #view-configure button").forEach((control) => { control.disabled = disabled; });
    setPlanControlsDisabled(disabled);
    if (!disabled) {
      const active = activeGenerationStates.has(recoveredState);
      document.querySelector("#start-generation").disabled = active || !chapterPlan || !planMatchesSelection(chapterPlan) || !currentChapterRange().valid;
      document.querySelector("#save-labels").disabled = active || !labelsDirty;
    }
  };

  const formatElapsed = (started) => {
    const timestamp = Date.parse(started || ""); if (!Number.isFinite(timestamp)) return "Elapsed time unavailable";
    const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
    return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
  };

  const generationData = (body) => body.generation_summary || body.job?.generation_summary || body.job?.progress || {};

  const renderProgress = (body) => {
    const summary = generationData(body); const job = body.job || {};
    const completedChunks = Number.isFinite(summary.completed_chunks) ? summary.completed_chunks : Array.isArray(job.completed_chunks) ? job.completed_chunks.length : Number(job.progress?.completed || 0);
    const totalChunks = Number(summary.total_chunks || job.total_chunks || job.progress?.total || 0);
    const percent = totalChunks > 0 ? Math.min(100, Math.round((completedChunks / totalChunks) * 100)) : 0;
    const totalChapters = Number(summary.total_chapters || chapterPlan?.chapters?.length || 0);
    const currentChapter = Number(summary.current_chapter || 0);
    const completedChapters = Number(summary.completed_chapters || 0);
    const stage = summary.stage || job.stage || body.state || "preparing";
    document.querySelector("#progress-stage").textContent = stage.replaceAll("_", " ");
    document.querySelector("#progress-percent").textContent = `${percent}%`;
    const bar = document.querySelector("#progress-bar"); bar.setAttribute("aria-valuenow", String(percent)); bar.querySelector("span").style.width = `${percent}%`;
    document.querySelector("#progress-chapter").textContent = totalChapters ? `${currentChapter || "-"} / ${totalChapters}` : "-";
    document.querySelector("#progress-chunks").textContent = totalChunks ? `${completedChunks} / ${totalChunks}` : "-";
    document.querySelector("#progress-completed-chapters").textContent = totalChapters ? `${completedChapters} / ${totalChapters}` : "-";
    document.querySelector("#progress-elapsed").textContent = formatElapsed(summary.run_started_at || job.run_started_at);
    document.querySelector("#progress-status").textContent = body.state === "cancelling" ? "Cancellation requested. Completed work is kept for a future resume." : "Generation is running locally.";
  };

  const stopPolling = () => { if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; } };
  const pollStatus = async () => {
    try {
      const response = await fetch("/api/status"); if (!response.ok) throw new Error("Status is temporarily unavailable.");
      const body = await response.json(); applyStatus(body);
      if (activeGenerationStates.has(body.state)) pollTimer = setTimeout(pollStatus, 1500); else stopPolling();
    } catch (error) { document.querySelector("#progress-status").textContent = error.message || "Status is temporarily unavailable."; pollTimer = setTimeout(pollStatus, 3000); }
  };

  const startPolling = () => { stopPolling(); elapsedTimer = setInterval(() => { const state = document.querySelector("#progress-elapsed"); if (state.dataset.started) state.textContent = formatElapsed(state.dataset.started); }, 1000); pollTimer = setTimeout(pollStatus, 0); };

  const renderOutput = (output) => {
    const value = output || {};
    document.querySelector("#output-filename").textContent = value.filename || "Unavailable";
    document.querySelector("#output-path").textContent = value.path || "Unavailable";
    document.querySelector("#output-size").textContent = Number.isFinite(value.size_bytes) ? `${(value.size_bytes / 1048576).toFixed(1)} MB` : "Unavailable";
    document.querySelector("#output-duration").textContent = Number.isFinite(value.duration_seconds) ? `${Math.floor(value.duration_seconds / 60)}m ${Math.round(value.duration_seconds % 60)}s` : "Unavailable";
    document.querySelector("#output-chapters").textContent = Number.isFinite(value.chapter_count) ? String(value.chapter_count) : "Unavailable";
  };

  const applyStatus = (body) => {
    conversionId = body.conversion_id || conversionId;
    setStructuralPlanLocked(body.job?.schema_version === 4);
    if (body.analysis) renderAnalysis(body.analysis);
    if (body.chapter_plan) renderPlan(body.chapter_plan);
    const recordedSettings = body.job?.tts?.settings || {};
    if (body.chapter_plan && (Object.hasOwn(recordedSettings, "chapter_start") || Object.hasOwn(recordedSettings, "chapter_end"))) {
      const total = chapterPlan.chapters.length;
      const start = typeIsInteger(recordedSettings.chapter_start) ? recordedSettings.chapter_start : 1;
      const end = typeIsInteger(recordedSettings.chapter_end) ? recordedSettings.chapter_end : total;
      document.querySelector("#chapter-start").value = String(start);
      document.querySelector("#chapter-end").value = String(end);
      updateChapterRange();
    }
    const state = body.state || body.job?.status;
    recoveredState = state;
    recoveredRoute = ["invalid", "analyzed", "planned", "resumable", ...activeGenerationStates, "completed", "cancelled", "failed"].includes(state);
    if (state === "no_active") { recoveredRoute = false; resetExistingJob(); return; }
    setGenerationControlsDisabled(activeGenerationStates.has(state));
    updateExistingJob(body, state);
    const recordedVoice = body.job?.tts?.voice || body.job?.voice;
    const recordedSpeed = body.job?.tts?.speed || body.job?.speed;
    if (recordedVoice) {
      const voiceControl = document.querySelector(`input[name=voice][value="${recordedVoice}"]`);
      if (voiceControl) { voiceControl.checked = true; voiceControl.dispatchEvent(new Event("change")); }
    }
    if (Number.isFinite(Number(recordedSpeed)) && Number(recordedSpeed) >= 0.5 && Number(recordedSpeed) <= 2) {
      document.querySelector("#speed").value = Number(recordedSpeed);
      document.querySelector("#speed-value").textContent = `${Number(recordedSpeed).toFixed(1)}x`;
    }
    if (state === "invalid") {
      stopPolling(); existing.hidden = false; document.querySelector("#existing-message").textContent = `Existing job needs attention: ${body.reason || "the saved job is invalid"}. Delete it explicitly before starting another.`; show("add"); return;
    }
    if (activeGenerationStates.has(state)) {
      show("progress"); renderProgress(body); document.querySelector("#completion-card").hidden = true; document.querySelector("#progress-live").hidden = false; document.querySelector("#cancel-generation").disabled = state === "cancelling";
      const started = generationData(body).run_started_at || body.job?.run_started_at; document.querySelector("#progress-elapsed").dataset.started = started || ""; return;
    }
    if (state === "completed") {
      show("progress"); stopPolling(); renderProgress(body); renderOutput(body.job?.output || body.output); document.querySelector("#progress-live").hidden = true; document.querySelector("#completion-card").hidden = false; document.querySelector("#progress-note").hidden = true; return;
    }
    if (state === "cancelled" || state === "failed") {
      stopPolling(); if (analysis) { show("configure"); document.querySelector("#generation-status").textContent = state === "cancelled" ? "Generation was cancelled. Review your settings and Generate audiobook to resume." : `Generation failed: ${body.job?.error || body.reason || "The local worker reported an error."} Review your settings and retry.`; } else { existing.hidden = false; document.querySelector("#existing-message").textContent = state === "cancelled" ? "Generation was cancelled. Delete the job or reload it to retry." : `Generation failed: ${body.job?.error || body.reason || "The local worker reported an error."}`; show("add"); } return;
    }
    if ((state === "analyzed" || state === "planned" || state === "resumable") && analysis) {
      stopPolling(); show("configure"); existing.hidden = false; document.querySelector("#existing-message").textContent = state === "planned" ? "A planned job is ready to review." : "An existing job is ready. Review settings before generating.";
    }
  };

  const saveLabels = async () => {
    if (!labelsDirty) return true;
    if (activeGenerationStates.has(recoveredState)) {
      document.querySelector("#plan-status").textContent = "Cancel generation before editing chapter labels.";
      return false;
    }
    const titles = [...document.querySelectorAll(".chapter-label")].map((control) => control.value);
    const target = document.querySelector("#plan-status"); target.textContent = "Saving labels...";
    try {
      const response = await fetch("/api/chapter-plan/titles", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ titles }) });
      if (!response.ok) { const error = await readError(response); throw new Error(`${error.code}: ${error.message}`); }
      renderPlan((await response.json()).chapter_plan); target.textContent = "Labels saved."; return true;
    } catch (error) { target.textContent = error.message || "Labels could not be saved."; return false; }
  };

  const startGeneration = async () => {
    if (generationRequestInFlight || activeGenerationStates.has(recoveredState) || !chapterPlan || !planMatchesSelection(chapterPlan)) return;
    const range = updateChapterRange();
    if (!range.valid) { document.querySelector("#generation-status").textContent = "Choose a valid inclusive chapter range before generating."; return; }
    generationRequestInFlight = true; setGenerationControlsDisabled(true); document.querySelector("#generation-status").textContent = "Preparing local generation...";
    try {
      if (!(await saveLabels())) throw new Error("Labels could not be saved. Generation did not start.");
      const voice = document.querySelector("input[name=voice]:checked").value; const speed = Number(document.querySelector("#speed").value); const performance_mode = document.querySelector("input[name=performance-mode]:checked").value;
      const response = await fetch("/api/generation/start", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ voice, speed, performance_mode, chapter_start: range.start, chapter_end: range.end }) });
      if (!response.ok) { const error = await readError(response); throw new Error(`${error.code}: ${error.message}`); }
      const body = await response.json(); conversionId = body.conversion_id || conversionId; const startState = body.status === "planned" ? "starting" : body.status || "starting"; applyStatus({ ...body, state: startState }); show("progress"); startPolling();
    } catch (error) { document.querySelector("#generation-status").textContent = error.message || "Generation could not start."; }
    finally { generationRequestInFlight = false; if (!pollTimer) setGenerationControlsDisabled(false); }
  };

  const stopPreview = () => { previewRequest += 1; if (activePreview) { activePreview.pause(); activePreview.currentTime = 0; activePreview = null; } document.querySelectorAll(".preview-voice").forEach((button) => { button.textContent = "Preview"; }); };
  const previewVoice = async (voice, button) => {
    stopPreview(); button.textContent = "Loading..."; document.querySelector("#preview-status").textContent = `Loading ${voice} preview...`;
    const requestId = previewRequest;
    try {
      activePreview = new Audio(`/api/voice-preview/${encodeURIComponent(voice)}`);
      activePreview.addEventListener("error", () => { if (requestId === previewRequest) { stopPreview(); document.querySelector("#preview-status").textContent = "Preview unavailable: the local preview file could not be loaded. Generation is still available."; } }, { once: true });
      button.textContent = "Stop preview"; document.querySelector("#preview-status").textContent = `${voice} preview playing.`;
      activePreview.addEventListener("ended", stopPreview, { once: true }); await activePreview.play();
    } catch (error) { stopPreview(); document.querySelector("#preview-status").textContent = `Preview unavailable: ${error.message || "the local preview file could not be played"}. Generation is still available.`; }
  };

  const openOutput = async (target) => {
    const targetStatus = document.querySelector("#output-status");
    const outputControls = [document.querySelector("#open-audiobook"), document.querySelector("#open-folder"), document.querySelector("#convert-another")];
    outputControls.forEach((control) => { control.disabled = true; });
    targetStatus.textContent = `Opening ${target === "folder" ? "output folder" : "audiobook"}...`;
    try {
      const response = await fetch("/api/output/open", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ target }) });
      if (!response.ok) { const error = await readError(response); throw new Error(`${error.code}: ${error.message}`); }
      targetStatus.textContent = target === "folder" ? "Output folder opened." : "Audiobook opened.";
    } catch (error) { targetStatus.textContent = error.message || "The output could not be opened."; }
    finally { outputControls.forEach((control) => { control.disabled = false; }); }
  };

  const refreshCurrentJob = async (message) => {
    try {
      const response = await fetch("/api/status");
      if (!response.ok) throw new Error("Status is temporarily unavailable.");
      const body = await response.json();
      applyStatus(body);
      if (activeGenerationStates.has(recoveredState)) startPolling();
      return recoveredState !== "no_active";
    } catch (_) {
      updateExistingJob({}, "unknown", `${message || "Current job status is unavailable."} View Current Job to retry.`);
      return false;
    }
  };

  const cancelGeneration = async (button, target) => {
    if (!conversionId || !activeGenerationStates.has(recoveredState)) {
      target.textContent = "There is no active generation to cancel.";
      return false;
    }
    button.disabled = true;
    target.textContent = "Requesting cancellation...";
    try {
      const response = await fetch("/api/generation/cancel", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ conversion_id: conversionId }) });
      if (!response.ok) { const error = await readError(response); throw new Error(`${error.code}: ${error.message}`); }
      target.textContent = "Cancellation requested. Completed work is kept.";
      await refreshCurrentJob("Cancellation requested.");
      return true;
    } catch (error) {
      target.textContent = error.message || "Cancellation could not be requested.";
      button.disabled = false;
      return false;
    }
  };

  const loadStatus = async () => {
    try { const response = await fetch("/api/status"); if (!response.ok) return; const body = await response.json(); applyStatus(body); }
    catch (_) { /* The shell remains usable if status is unavailable. */ }
  };

  document.querySelectorAll(".step,[data-next],[data-view]").forEach((button) => button.addEventListener("click", () => show(button.dataset.next || button.dataset.view)));
  document.querySelector(".browse").addEventListener("click", () => input.click()); analyzeButton.addEventListener("click", () => upload(selectedFile));
  const chosen = (file) => { if (!file) return; selectedFile = file; analyzeButton.disabled = !existing.hidden || Boolean(recoveredState && recoveredState !== "no_active"); status.textContent = `${file.name} selected. Choose Analyze PDF to continue.`; document.querySelector(".dropzone").dataset.fileName = file.name; };
  input.addEventListener("change", () => chosen(input.files[0]));
  const drop = document.querySelector(".dropzone");
  drop.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  drop.addEventListener("dragover", (event) => { event.preventDefault(); }); drop.addEventListener("drop", (event) => { event.preventDefault(); chosen(event.dataTransfer.files[0]); });
  document.querySelectorAll("input[name=plan-mode]").forEach((control) => control.addEventListener("change", () => { const mode = control.value; document.querySelector(".custom-count").hidden = mode !== "custom"; if (mode === "custom") { invalidatePlan("Choose a count from 2–50, then select Generate plan."); return; } runPlanRequest(mode); }));
  document.querySelector("#chapter-count").addEventListener("input", () => { if (currentPlanSpec().mode === "custom") invalidatePlan("Choose a count from 2–50, then select Generate plan."); });
  document.querySelectorAll("#chapter-start, #chapter-end").forEach((control) => control.addEventListener("input", updateChapterRange));
  document.querySelector("#regenerate-plan").addEventListener("click", async () => { const selection = currentPlanSpec(); await runPlanRequest(selection.mode, selection.mode === "custom" ? selection.count : undefined); });
  document.querySelector("#save-labels").addEventListener("click", () => saveLabels()); document.querySelector("#start-generation").addEventListener("click", startGeneration);
  document.querySelectorAll("input[name=voice]").forEach((control) => control.addEventListener("change", () => { document.querySelectorAll(".voice-card").forEach((card) => card.classList.toggle("selected", card.querySelector("input").checked)); }));
  document.querySelectorAll(".preview-voice").forEach((button) => button.addEventListener("click", () => { if (button.textContent === "Stop preview") stopPreview(); else previewVoice(button.dataset.voice, button); }));
  document.querySelector("#speed").addEventListener("input", (event) => { document.querySelector("#speed-value").textContent = `${Number(event.target.value).toFixed(1)}x`; });
  document.querySelector("#cancel-generation").addEventListener("click", () => cancelGeneration(document.querySelector("#cancel-generation"), document.querySelector("#progress-status")));
  document.querySelector("#view-current-job").addEventListener("click", () => refreshCurrentJob("Current job status could not be refreshed."));
  document.querySelector("#cancel-current-generation").addEventListener("click", () => cancelGeneration(document.querySelector("#cancel-current-generation"), document.querySelector("#existing-message")));
  document.querySelector("#open-audiobook").addEventListener("click", () => openOutput("audiobook")); document.querySelector("#open-folder").addEventListener("click", () => openOutput("folder"));
  document.querySelector("#exit-complete").addEventListener("click", async () => { const target = document.querySelector("#output-status"); try { const response = await fetch("/api/shutdown", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin } }); target.textContent = response.ok ? "The local app is shutting down. You can close this tab." : "Exit requires an authenticated local session."; } catch (_) { target.textContent = "The local app is no longer reachable."; } });
  document.querySelector("#convert-another").addEventListener("click", async () => {
    if (!conversionId) return;
    const button = document.querySelector("#convert-another"); button.disabled = true;
    try {
      const response = await fetch(`/api/workspace/${conversionId}`, { method: "DELETE", headers: { "Origin": location.origin } });
      if (!response.ok) { document.querySelector("#output-status").textContent = "The existing conversion could not be reset."; button.disabled = false; return; }
      stopPolling(); analysis = null; chapterPlan = null; conversionId = null; recoveredState = null; labelsDirty = false; selectedFile = null; input.value = ""; document.querySelector("#completion-card").hidden = true; document.querySelector("#progress-live").hidden = false; document.querySelector("#generation-status").textContent = ""; resetExistingJob(); invalidatePlan("Select a PDF to begin."); show("add");
    } catch (_) { document.querySelector("#output-status").textContent = "The existing conversion could not be reset."; button.disabled = false; }
  });
  document.querySelector(".delete-job").addEventListener("click", async () => {
    if (activeGenerationStates.has(recoveredState) || (!conversionId && recoveredState !== "invalid")) return;
    const button = document.querySelector(".delete-job"); button.disabled = true;
    const endpoint = recoveredState === "invalid" ? "/api/workspace/active" : conversionId ? `/api/workspace/${conversionId}` : "/api/workspace/active";
    try {
      const response = await fetch(endpoint, { method: "DELETE", headers: { "Origin": location.origin } });
      let payload = null;
      try { payload = await response.json(); } catch (_) { /* Treat malformed deletion responses as failures. */ }
      if (response.ok && payload?.deleted === true) { analysis = null; conversionId = null; recoveredState = null; resetExistingJob(); invalidatePlan("Select a PDF to begin."); status.textContent = "Existing job deleted explicitly."; show("add"); }
      else { status.textContent = "The existing job could not be deleted right now."; button.disabled = false; }
    } catch (_) { status.textContent = "The existing job could not be deleted right now."; button.disabled = false; }
  });

  const token = new URLSearchParams(location.hash.slice(1)).get("session");
  const ready = token ? fetch("/api/session/bootstrap", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ token }) }).then((response) => { if (!response.ok) throw new Error("session bootstrap failed"); history.replaceState(null, "", "#add"); status.textContent = "Secure local session ready."; }) : Promise.resolve();
  ready.then(loadStatus).then(() => {
    if (activeGenerationStates.has(recoveredState)) { startPolling(); return; }
    if (recoveredRoute) return;
    const initial = location.hash.slice(1); show(views.includes(initial) ? initial : "add");
  }).catch(() => { status.textContent = "Could not establish a secure local session. Relaunch the app to try again."; });
})();

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
  let activePreviewButton = null;
  let activePreviewStatus = null;
  let previewRequest = 0;
  let recoveredState = null;
  let recoveredRoute = false;
  let structuralPlanLocked = false;
  let analyzedChapterRange = null;
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
    const chapterTitle = (index) => {
      const title = chapterPlan?.chapters?.[index - 1]?.title;
      return typeof title === "string" && title.trim() ? title.trim() : `Chapter ${index}`;
    };
    if (!range.total) target.textContent = "Select an inclusive chapter range after generating a plan.";
    else if (!range.valid) target.textContent = `Choose an inclusive range from 1 through ${range.total}.`;
    else if (range.start === range.end) target.textContent = `${chapterTitle(range.start)} selected (inclusive).`;
    else target.textContent = `${range.end - range.start + 1} chapters selected: ${chapterTitle(range.start)} through ${chapterTitle(range.end)} (inclusive).`;
    const ready = Boolean(chapterPlan && planMatchesSelection(chapterPlan) && range.valid && !planRequestInFlight);
    document.querySelector(".next").disabled = !ready;
    if (interactiveMode) interactive("#start-voice-analysis").disabled = !chapterPlan || planRequestInFlight || !range.valid;
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
    analyzedChapterRange = null;
    document.querySelector(".next").disabled = true;
    document.querySelector("#save-labels").disabled = true;
    if (message) document.querySelector("#plan-status").textContent = message;
    updateModeReadiness();
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
      if (interactiveMode) { interactive("#start-voice-analysis").disabled = !chapterPlan; interactive("#interactive-entry-status").textContent = chapterPlan ? "Chapter plan ready. Start local voice analysis when you are ready to review the cast." : "Generate a chapter plan before starting voice analysis."; }
      updateModeReadiness();
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
    document.querySelectorAll("#view-configure input, #view-configure button, #view-configure select").forEach((control) => { control.disabled = disabled; });
    setPlanControlsDisabled(disabled);
    if (!disabled) {
      const active = activeGenerationStates.has(recoveredState);
      document.querySelector("#save-labels").disabled = active || !labelsDirty;
      updateModeReadiness();
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
    if (body.job?.mode === "interactive_voices") {
      const modeControl = document.querySelector('input[name="voice-mode"][value="interactive_voices"]');
      if (modeControl && !modeControl.checked) { modeControl.checked = true; setInteractiveMode(true); }
    }
    setStructuralPlanLocked(body.job?.schema_version === 4 || body.job?.schema_version === 5);
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
    if (interactiveMode) { await startInteractiveGeneration(); return; }
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

  const previewStatusFor = (button, statusTarget) => statusTarget || button?.closest(".cast-entry")?.querySelector(".interactive-preview-status") || document.querySelector("#preview-status");
  const stopPreview = (statusTarget) => {
    const stoppedStatus = activePreviewStatus || statusTarget;
    previewRequest += 1;
    if (activePreview) { activePreview.pause(); activePreview.currentTime = 0; activePreview = null; }
    activePreviewButton = null;
    activePreviewStatus = null;
    document.querySelectorAll(".preview-voice").forEach((button) => { button.textContent = "Preview"; });
    if (stoppedStatus) stoppedStatus.textContent = "Stopped.";
  };
  const previewVoice = (voice, button, statusTarget) => {
    stopPreview();
    const targetStatus = previewStatusFor(button, statusTarget);
    activePreviewButton = button;
    activePreviewStatus = targetStatus;
    button.textContent = "Loading...";
    if (targetStatus) targetStatus.textContent = `Loading ${voice} preview...`;
    const requestId = previewRequest;
    let audio;
    try {
      audio = new Audio(`/api/voice-preview/${encodeURIComponent(voice)}`);
    } catch (error) {
      if (requestId !== previewRequest) return;
      stopPreview(targetStatus);
      if (targetStatus) targetStatus.textContent = `Preview unavailable: ${error.message || "the local preview file could not be played"}. Generation is still available.`;
      return;
    }
    if (requestId !== previewRequest) { audio.pause(); return; }
    activePreview = audio;
    audio.addEventListener("error", () => {
      if (requestId !== previewRequest) return;
      stopPreview(targetStatus);
      if (targetStatus) targetStatus.textContent = "Preview unavailable: the local preview file could not be loaded. Generation is still available.";
    }, { once: true });
    button.textContent = "Stop preview";
    if (targetStatus) targetStatus.textContent = `${voice} preview playing.`;
    audio.addEventListener("ended", () => {
      if (requestId !== previewRequest) return;
      activePreview = null;
      activePreviewButton = null;
      activePreviewStatus = null;
      button.textContent = "Preview";
      if (targetStatus) targetStatus.textContent = "Finished.";
    }, { once: true });
    try {
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch((error) => {
          if (requestId !== previewRequest) return;
          stopPreview(targetStatus);
          if (targetStatus) targetStatus.textContent = `Preview unavailable: ${error.message || "the local preview file could not be played"}. Generation is still available.`;
        });
      }
    } catch (error) {
      if (requestId !== previewRequest) return;
      stopPreview(targetStatus);
      if (targetStatus) targetStatus.textContent = `Preview unavailable: ${error.message || "the local preview file could not be played"}. Generation is still available.`;
    }
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
      await restoreInteractiveRecovery(body);
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

  let interactiveMode = false;
  let voiceCatalog = [];
  let voicePlan = null;
  let voicePlanRevision = null;
  let voiceAnalysis = null;
  let voiceAnalysisPollTimer = null;
  let spanOffset = 0;
  let spanTotal = 0;

  const interactive = (selector) => document.querySelector(selector);
  const interactiveError = async (response) => {
    const error = await readError(response);
    return new Error(`${error.code}: ${error.message}`);
  };

  const updateSingleVoiceReadiness = () => {
    if (interactiveMode) return false;
    const ready = Boolean(chapterPlan && planMatchesSelection(chapterPlan) && currentChapterRange().valid && !generationRequestInFlight && !planRequestInFlight && !activeGenerationStates.has(recoveredState));
    document.querySelector("#start-generation").disabled = !ready;
    return ready;
  };

  const updateInteractiveReadiness = () => {
    if (!interactiveMode) return false;
    const accepted = Boolean(interactive("#accept-narrator-fallback")?.checked);
    const unresolved = Number(voicePlan?.review?.unresolved_count || 0);
    const approved = voicePlan?.approval?.state === "approved";
    const range = currentChapterRange();
    const analysisComplete = voiceAnalysis?.status === "completed";
    const rangeAnalyzed = Boolean(analysisComplete && range.valid && analyzedChapterRange && range.start >= analyzedChapterRange.start && range.end <= analyzedChapterRange.end);
    const canApprove = Boolean(voicePlan && !approved && rangeAnalyzed && (!unresolved || accepted));
    if (interactive("#approve-voice-plan")) interactive("#approve-voice-plan").disabled = !canApprove;
    const context = interactive("#interactive-generation-context");
    if (context) {
      context.textContent = rangeAnalyzed
        ? "Interactive Voices plan: approval is required before generation."
        : analyzedChapterRange
          ? `Selected chapters ${range.start}–${range.end} are outside the analyzed range (${analyzedChapterRange.start}–${analyzedChapterRange.end}). Start Interactive Voice Analysis again for this range.`
          : "Interactive Voices analysis is required for the selected chapter range before approval or generation.";
    }
    const ready = Boolean(approved && rangeAnalyzed && !generationRequestInFlight && !planRequestInFlight && !activeGenerationStates.has(recoveredState));
    document.querySelector("#start-generation").disabled = !ready;
    return ready;
  };

  const updateModeReadiness = () => {
    if (interactiveMode) updateInteractiveReadiness();
    else updateSingleVoiceReadiness();
  };

  const setInteractiveMode = (enabled) => {
    interactiveMode = enabled;
    interactive("#interactive-entry").hidden = !enabled;
    interactive("#interactive-review").hidden = !enabled;
    interactive("#single-voice-controls").hidden = enabled;
    interactive("#interactive-generation-context").hidden = !enabled;
    if (enabled) {
      interactive("#start-voice-analysis").disabled = !chapterPlan || planRequestInFlight || !currentChapterRange().valid;
      interactive("#interactive-entry-status").textContent = chapterPlan ? "Chapter plan ready. Start local voice analysis when you are ready to review the cast." : "Generate a chapter plan before starting voice analysis.";
    }
    updateModeReadiness();
  };

  const renderAnalysisProgress = (body) => {
    const trackedRange = body && typeIsInteger(body.chapter_start) && typeIsInteger(body.chapter_end)
      ? { start: body.chapter_start, end: body.chapter_end }
      : null;
    if (trackedRange) analyzedChapterRange = trackedRange;
    const progress = body.progress || {};
    const completed = Number(progress.completed || 0);
    const total = Number(progress.total || 0);
    const percent = total > 0 ? Math.min(100, Math.round(completed / total * 100)) : body.status === "completed" ? 100 : 0;
    interactive("#interactive-analysis-stage").textContent = String(body.stage || body.status || "queued").replaceAll("_", " ");
    interactive("#interactive-analysis-percent").textContent = `${percent}%`;
    interactive("#interactive-analysis-bar").setAttribute("aria-valuenow", String(percent));
    interactive("#interactive-analysis-bar span").style.width = `${percent}%`;
    interactive("#interactive-analysis-progress").hidden = false;
    interactive("#cancel-voice-analysis").disabled = !body.cancelable;
    const rangeLabel = analyzedChapterRange ? ` Chapters ${analyzedChapterRange.start}–${analyzedChapterRange.end} are included.` : "";
    const message = body.error?.message || (body.status === "completed" ? `Analysis complete. Preparing a reviewable voice plan.${rangeLabel}` : body.status === "cancelled" ? "Voice analysis was cancelled. You can start it again when ready." : `Voice analysis ${body.status || "is running"}.${rangeLabel}`);
    interactive("#interactive-analysis-status").textContent = message;
    updateModeReadiness();
  };

  const stopVoiceAnalysisPolling = () => { if (voiceAnalysisPollTimer) { clearTimeout(voiceAnalysisPollTimer); voiceAnalysisPollTimer = null; } };

  const loadVoices = async () => {
    if (voiceCatalog.length) return voiceCatalog;
    const response = await fetch("/api/voices");
    if (!response.ok) throw await interactiveError(response);
    const body = await response.json();
    voiceCatalog = Array.isArray(body.voices) ? body.voices.filter((entry) => entry && entry.enabled !== false) : [];
    if (!voiceCatalog.length) throw new Error("VOICE_CATALOG_EMPTY: no local voices are available.");
    return voiceCatalog;
  };

  const voiceOptions = (selected) => voiceCatalog.map((entry) => {
    const option = document.createElement("option");
    const label = entry.display_label || entry.display_name || entry.label || entry.name || entry.id;
    const description = typeof entry.description === "string" && entry.description.trim() ? ` - ${entry.description.trim()}` : "";
    option.value = entry.id; option.textContent = `${label}${description}`; option.selected = entry.id === selected; return option;
  });

  const mutateVoicePlan = async (endpoint, payload) => {
    const response = await fetch(endpoint, { method: endpoint === "/api/voice-plan" ? "PUT" : "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify(payload) });
    if (!response.ok) throw await interactiveError(response);
    const body = await response.json();
    voicePlanRevision = body.revision || voicePlanRevision;
    await loadVoicePlan();
    return body;
  };

  const renderCast = () => {
    const target = interactive("#interactive-cast"); target.replaceChildren();
    (voicePlan?.cast || []).forEach((entry) => {
      const article = document.createElement("article"); article.className = "cast-entry";
      const header = document.createElement("div"); header.className = "cast-entry-header";
      const title = document.createElement("strong"); title.textContent = entry.display_label; header.append(title);
      const meta = document.createElement("small"); meta.textContent = `${entry.role || "character"} · ${entry.relationship || "third_person"}`; header.append(meta); article.append(header);
      const fields = document.createElement("div"); fields.className = "cast-fields";
      const nameLabel = document.createElement("label"); nameLabel.textContent = "Name"; const name = document.createElement("input"); name.type = "text"; name.value = entry.display_label; name.maxLength = 512; name.setAttribute("aria-label", `${entry.display_label} display name`); nameLabel.append(name);
      const voiceField = document.createElement("div"); voiceField.className = "cast-voice-field";
      const voiceLabel = document.createElement("label"); voiceLabel.textContent = "Voice"; const voice = document.createElement("select"); voice.setAttribute("aria-label", `${entry.display_label} voice`); voiceOptions(entry.voice_id).forEach((option) => voice.append(option)); voiceLabel.append(voice); voiceField.append(voiceLabel);
      const previewControls = document.createElement("span"); previewControls.className = "cast-preview";
      const preview = document.createElement("button"); preview.className = "preview-voice interactive-preview"; preview.type = "button"; preview.textContent = "Preview"; preview.setAttribute("aria-label", `Preview ${entry.display_label} voice`);
      const previewStatus = document.createElement("span"); previewStatus.className = "status interactive-preview-status"; previewStatus.setAttribute("role", "status"); previewStatus.setAttribute("aria-live", "polite"); previewStatus.textContent = "";
      preview.addEventListener("click", () => { if (preview.textContent === "Stop preview") stopPreview(previewStatus); else previewVoice(voice.value, preview, previewStatus); });
      voice.addEventListener("change", () => { if (activePreviewButton === preview) stopPreview(previewStatus); });
      previewControls.append(preview, previewStatus); voiceField.append(previewControls);
      const speedLabel = document.createElement("label"); speedLabel.textContent = "Speed"; const speed = document.createElement("input"); speed.type = "number"; speed.min = "0.5"; speed.max = "2"; speed.step = "0.1"; speed.value = Number(entry.voice_settings?.speed || 1).toFixed(1); speed.setAttribute("aria-label", `${entry.display_label} speed`); speedLabel.append(speed);
      fields.append(nameLabel, voiceField, speedLabel);
      article.append(fields);
      const relationshipActions = document.createElement("div"); relationshipActions.className = "cast-action-row cast-primary-actions";
      const relationshipLabel = document.createElement("label"); relationshipLabel.className = "cast-relationship-label"; relationshipLabel.textContent = "Relationship";
      const relationship = document.createElement("select"); relationship.setAttribute("aria-label", `${entry.display_label} relationship`); ["third_person", "same_as_narrator", "separate_from_narrator"].forEach((value) => { const option = document.createElement("option"); option.value = value; option.textContent = value.replaceAll("_", " "); option.selected = value === entry.relationship; relationship.append(option); }); relationshipLabel.append(relationship); relationshipActions.append(relationshipLabel);
      const save = document.createElement("button"); save.className = "secondary cast-save"; save.type = "button"; save.textContent = "Save cast"; save.addEventListener("click", async () => { save.disabled = true; interactive("#interactive-plan-status").textContent = "Saving cast changes..."; try { await mutateVoicePlan("/api/voice-plan", { expected_revision: voicePlanRevision, cast_id: entry.cast_id, display_label: name.value.trim(), voice_id: voice.value, speed: Number(speed.value), relationship: relationship.value }); interactive("#interactive-plan-status").textContent = "Cast changes saved."; } catch (error) { interactive("#interactive-plan-status").textContent = error.message; } finally { save.disabled = false; } }); relationshipActions.append(save); article.append(relationshipActions);
      if (entry.role === "character") {
        const characters = (voicePlan?.cast || []).filter((candidate) => candidate.role === "character" && candidate.cast_id !== entry.cast_id);
        const mergeLabel = document.createElement("label"); mergeLabel.textContent = "Merge into";
        const mergeTarget = document.createElement("select"); mergeTarget.className = "cast-merge-target"; mergeTarget.setAttribute("aria-label", `Merge ${entry.display_label} into`);
        characters.forEach((candidate) => { const option = document.createElement("option"); option.value = candidate.cast_id; option.textContent = candidate.display_label || candidate.cast_id; mergeTarget.append(option); });
        const merge = document.createElement("button"); merge.className = "secondary cast-merge"; merge.type = "button"; merge.textContent = "Merge"; merge.disabled = !characters.length; merge.addEventListener("click", async () => { const targetId = mergeTarget.value; if (!targetId || !window.confirm(`Merge ${entry.display_label} into ${characters.find((candidate) => candidate.cast_id === targetId)?.display_label || targetId}?`)) return; merge.disabled = true; try { await mutateVoicePlan("/api/voice-plan/cast/merge", { expected_revision: voicePlanRevision, source_cast_id: entry.cast_id, target_cast_id: targetId }); interactive("#interactive-plan-status").textContent = `${entry.display_label} merged into the selected voice.`; } catch (error) { interactive("#interactive-plan-status").textContent = error.message; merge.disabled = false; } });
        const remove = document.createElement("button"); remove.className = "secondary cast-remove"; remove.type = "button"; remove.textContent = "Remove voice"; remove.addEventListener("click", async () => { if (!window.confirm(`Remove ${entry.display_label}? Its attributed lines will use Narrator.`)) return; remove.disabled = true; try { await mutateVoicePlan("/api/voice-plan/cast/remove", { expected_revision: voicePlanRevision, cast_id: entry.cast_id }); interactive("#interactive-plan-status").textContent = `${entry.display_label} removed; its lines now use Narrator.`; } catch (error) { interactive("#interactive-plan-status").textContent = error.message; remove.disabled = false; } });
        mergeLabel.append(mergeTarget);
        const characterActions = document.createElement("div"); characterActions.className = "cast-action-row cast-character-actions"; characterActions.append(mergeLabel, merge, remove);
        article.append(characterActions);
      }
      target.append(article);
    });
    if (!target.children.length) target.textContent = "No cast entries are available yet.";
  };

  const renderAliasRows = () => {
    const target = interactive("#interactive-aliases"); target.replaceChildren();
    const aliases = Array.isArray(voicePlan?.aliases) ? voicePlan.aliases : [];
    aliases.forEach((alias) => {
      const article = document.createElement("article"); article.className = "alias-entry";
      const label = document.createElement("span"); label.textContent = `${alias.text} → ${(voicePlan?.cast || []).find((entry) => entry.cast_id === alias.character_id)?.display_label || alias.character_id}`; article.append(label);
      const actions = document.createElement("div"); actions.className = "alias-actions";
      const merge = document.createElement("button"); merge.className = "secondary"; merge.type = "button"; merge.textContent = "Merge"; merge.addEventListener("click", async () => { const targetId = window.prompt("Merge this alias into cast ID:", alias.character_id); if (!targetId) return; try { await mutateVoicePlan("/api/voice-plan/aliases/merge", { expected_revision: voicePlanRevision, target_character_id: targetId, alias_ids: [alias.alias_id] }); interactive("#interactive-plan-status").textContent = "Alias merged."; } catch (error) { interactive("#interactive-plan-status").textContent = error.message; } }); actions.append(merge);
      const split = document.createElement("button"); split.className = "secondary"; split.type = "button"; split.textContent = "Split"; split.addEventListener("click", async () => { const labelText = window.prompt("New cast label:", alias.text); if (!labelText) return; try { await mutateVoicePlan("/api/voice-plan/aliases/split", { expected_revision: voicePlanRevision, alias_ids: [alias.alias_id], display_label: labelText }); interactive("#interactive-plan-status").textContent = "Alias split into a new cast entry."; } catch (error) { interactive("#interactive-plan-status").textContent = error.message; } }); actions.append(split); article.append(actions); target.append(article);
    });
    if (!target.children.length) target.textContent = "No aliases detected.";
  };

  const renderAliases = () => {
    const target = interactive("#interactive-aliases");
    const disclosure = interactive("#interactive-aliases-disclosure");
    const aliases = Array.isArray(voicePlan?.aliases) ? voicePlan.aliases : [];
    const suppliedCount = voicePlan?.alias_count;
    const numericCount = Number(suppliedCount);
    const aliasCount = suppliedCount !== null && suppliedCount !== undefined && Number.isFinite(numericCount) && numericCount >= 0 ? Math.floor(numericCount) : aliases.length;
    const truncated = voicePlan?.aliases_truncated === true;
    const summary = interactive("#interactive-aliases-summary");
    summary.textContent = aliasCount === 0
      ? "No aliases detected"
      : `${aliasCount} alias${aliasCount === 1 ? "" : "es"}${truncated ? " (only the loaded subset is shown when expanded)" : ""}`;
    target.replaceChildren();
    if (disclosure?.open) renderAliasRows();
  };

  const renderSpan = (span) => {
    const article = document.createElement("article"); article.className = "span-entry";
    const meta = document.createElement("div"); meta.className = "span-meta"; const left = document.createElement("span"); left.textContent = `Chapter ${span.chapter_index} · ${span.type}`; const right = document.createElement("span"); right.textContent = `${span.confidence?.band || "unknown"} confidence`; meta.append(left, right); article.append(meta);
    const excerpt = document.createElement("p"); excerpt.className = "span-excerpt"; excerpt.textContent = span.excerpt || "(No excerpt available)"; article.append(excerpt);
    const form = document.createElement("div"); form.className = "span-override";
    const kindLabel = document.createElement("label"); kindLabel.textContent = "Override"; const kind = document.createElement("select"); kind.setAttribute("aria-label", `Override kind for span ${span.span_id}`); [["speaker", "Speaker"], ["type", "Span type"]].forEach(([value, label]) => { const option = document.createElement("option"); option.value = value; option.textContent = label; option.selected = value === "speaker"; kind.append(option); }); kindLabel.append(kind); form.append(kindLabel);
    const toLabel = document.createElement("label"); toLabel.textContent = "To"; const to = document.createElement("select"); to.setAttribute("aria-label", `Override value for span ${span.span_id}`); const fillTo = () => { to.replaceChildren(); const values = kind.value === "speaker" ? (voicePlan?.cast || []).map((entry) => [entry.cast_id, entry.display_label]) : [["narration", "Narration"], ["dialogue", "Dialogue"], ["thought", "Thought"], ["unknown", "Unknown"]]; values.forEach(([value, label]) => { const option = document.createElement("option"); option.value = value; option.textContent = label; option.selected = value === (kind.value === "speaker" ? span.speaker_id : span.type); to.append(option); }); }; kind.addEventListener("change", fillTo); fillTo(); toLabel.append(to); form.append(toLabel);
    const reasonLabel = document.createElement("label"); reasonLabel.textContent = "Reason"; const reason = document.createElement("input"); reason.type = "text"; reason.placeholder = "Why this is clearer"; reason.setAttribute("aria-label", `Reason for span ${span.span_id} override`); reasonLabel.append(reason); form.append(reasonLabel);
    const save = document.createElement("button"); save.className = "secondary"; save.type = "button"; save.textContent = "Apply"; save.addEventListener("click", async () => { if (!reason.value.trim()) { interactive("#span-filter-status").textContent = "Add a reason before applying a span override."; reason.focus(); return; } save.disabled = true; try { await mutateVoicePlan("/api/voice-plan/spans/override", { expected_revision: voicePlanRevision, span_id: span.span_id, kind: kind.value, to: to.value, reason: reason.value.trim() }); interactive("#span-filter-status").textContent = "Span override saved."; } catch (error) { interactive("#span-filter-status").textContent = error.message; } finally { save.disabled = false; } }); form.append(save); article.append(form); return article;
  };

  const renderSpans = (spans, total, hasMore, append = false) => {
    const target = interactive("#interactive-spans"); if (!append) target.replaceChildren(); (spans || []).forEach((span) => target.append(renderSpan(span))); spanTotal = total || 0; interactive("#load-more-spans").hidden = !hasMore; interactive("#span-filter-status").textContent = `${target.children.length} of ${spanTotal} speaker span${spanTotal === 1 ? "" : "s"} shown.`;
  };

  const loadVoicePlan = async (append = false) => {
    const chapter = interactive("#span-chapter-filter").value; const confidence = interactive("#span-confidence-filter").value; const params = new URLSearchParams({ limit: "100", offset: String(append ? spanOffset : 0) }); if (chapter) params.set("chapter", chapter); if (confidence) params.set("confidence", confidence);
    const response = await fetch(`/api/voice-plan?${params.toString()}`); if (!response.ok) throw await interactiveError(response); const body = await response.json(); voicePlan = body; voicePlanRevision = body.revision; if (!append) { spanOffset = 0; renderCast(); renderAliases(); const chapters = [...new Set((chapterPlan?.chapters || []).map((entry) => entry.index))]; const select = interactive("#span-chapter-filter"); const selected = select.value; select.replaceChildren(); const all = document.createElement("option"); all.value = ""; all.textContent = "All chapters"; select.append(all); chapters.forEach((index) => { const option = document.createElement("option"); option.value = String(index); option.textContent = `Chapter ${index}`; option.selected = String(index) === selected; select.append(option); }); }
    spanOffset = Number(body.offset || 0) + (body.spans || []).length; renderSpans(body.spans, body.total, body.has_more, append); updateModeReadiness(); return body;
  };

  const draftVoicePlan = async (revision) => {
    interactive("#interactive-analysis-status").textContent = "Analysis complete. Drafting the reviewable voice plan...";
    const response = await fetch("/api/voice-plan/draft", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ analysis_revision: revision }) });
    if (!response.ok) throw await interactiveError(response); const body = await response.json(); voicePlanRevision = body.revision; await loadVoicePlan(); interactive("#interactive-editor").hidden = false; interactive("#interactive-analysis-status").textContent = "Voice plan ready for review.";
  };

  const pollVoiceAnalysis = async () => {
    try {
      const response = await fetch("/api/voice-analysis/status"); if (!response.ok) throw await interactiveError(response); const body = await response.json(); voiceAnalysis = body; renderAnalysisProgress(body);
      if (body.status === "completed") { stopVoiceAnalysisPolling(); try { await draftVoicePlan(body.revision); } catch (error) { interactive("#interactive-analysis-status").textContent = error.message || "The voice plan could not be drafted."; } }
      else if (body.status === "cancelled" || body.status === "failed") stopVoiceAnalysisPolling();
      else voiceAnalysisPollTimer = setTimeout(pollVoiceAnalysis, 1200);
    } catch (error) { interactive("#interactive-analysis-status").textContent = error.message || "Voice analysis status is unavailable."; voiceAnalysisPollTimer = setTimeout(pollVoiceAnalysis, 3000); }
  };

  const startVoiceAnalysis = async () => {
    const range = updateChapterRange();
    if (!chapterPlan || planRequestInFlight) { interactive("#interactive-entry-status").textContent = "Generate a chapter plan before starting voice analysis."; return; }
    if (!range.valid) { interactive("#interactive-entry-status").textContent = "Choose a valid inclusive chapter range before starting voice analysis."; return; }
    const button = interactive("#start-voice-analysis"); button.disabled = true; interactive("#interactive-entry-status").textContent = "Loading available local voices...";
    try { await loadVoices(); const response = await fetch("/api/voice-analysis", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ mode: "interactive", chapter_start: range.start, chapter_end: range.end }) }); if (!response.ok) throw await interactiveError(response); voiceAnalysis = await response.json(); analyzedChapterRange = { start: range.start, end: range.end }; interactive("#interactive-editor").hidden = true; renderAnalysisProgress(voiceAnalysis); pollVoiceAnalysis(); } catch (error) { interactive("#interactive-entry-status").textContent = error.message || "Interactive voice analysis could not start."; button.disabled = false; }
  };

  const cancelVoiceAnalysis = async () => {
    if (!voiceAnalysis?.analysis_id || !voiceAnalysis?.revision) return;
    const button = interactive("#cancel-voice-analysis"); button.disabled = true; interactive("#interactive-analysis-status").textContent = "Requesting voice analysis cancellation...";
    try { const response = await fetch("/api/voice-analysis/cancel", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ analysis_id: voiceAnalysis.analysis_id, revision: voiceAnalysis.revision }) }); if (!response.ok) throw await interactiveError(response); voiceAnalysis = await response.json(); renderAnalysisProgress(voiceAnalysis); pollVoiceAnalysis(); } catch (error) { interactive("#interactive-analysis-status").textContent = error.message || "Voice analysis cancellation failed."; button.disabled = false; }
  };

  const approveVoicePlan = async () => {
    const range = updateChapterRange();
    const rangeAnalyzed = Boolean(voiceAnalysis?.status === "completed" && range.valid && analyzedChapterRange && range.start >= analyzedChapterRange.start && range.end <= analyzedChapterRange.end);
    if (!range.valid) { interactive("#interactive-approval-status").textContent = "Choose a valid inclusive chapter range before approving the voice plan."; return; }
    if (!rangeAnalyzed) { interactive("#interactive-approval-status").textContent = "The selected range was not analyzed. Start Interactive Voice Analysis again before approving."; return; }
    const accepted = Boolean(interactive("#accept-narrator-fallback").checked); const unresolved = Number(voicePlan?.review?.unresolved_count || 0); if (unresolved && !accepted) { interactive("#interactive-approval-status").textContent = "Review unresolved spans and explicitly accept Narrator as their fallback."; return; }
    const button = interactive("#approve-voice-plan"); button.disabled = true; interactive("#interactive-approval-status").textContent = "Approving voice plan...";
    try { const response = await fetch("/api/voice-plan/approve", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ expected_revision: voicePlanRevision, accept_narrator_fallback: accepted }) }); if (!response.ok) throw await interactiveError(response); const body = await response.json(); voicePlanRevision = body.revision; await loadVoicePlan(); interactive("#interactive-approval-status").textContent = "Voice plan approved. Choose the chapter range and start generation."; updateInteractiveReadiness(); } catch (error) { interactive("#interactive-approval-status").textContent = error.message || "Voice plan approval failed."; updateInteractiveReadiness(); }
  };

  const startInteractiveGeneration = async () => {
    const range = updateChapterRange(); if (!range.valid) { interactive("#generation-status").textContent = "Choose a valid inclusive chapter range before generating."; return; } if (voicePlan?.approval?.state !== "approved") { interactive("#generation-status").textContent = "Approve the Interactive Voices plan before generating."; return; }
    generationRequestInFlight = true; setGenerationControlsDisabled(true); interactive("#generation-status").textContent = "Preparing Interactive Voices generation...";
    try { const performance_mode = document.querySelector("input[name=performance-mode]:checked").value; const response = await fetch("/api/generation/start", { method: "POST", headers: { "Content-Type": "application/json", "Origin": location.origin }, body: JSON.stringify({ mode: "interactive_voices", voice_plan_sha256: voicePlan.canonical_artifact_sha256, voice_plan_revision: voicePlan.revision, chapter_start: range.start, chapter_end: range.end, performance_mode }) }); if (!response.ok) throw await interactiveError(response); const body = await response.json(); conversionId = body.conversion_id || conversionId; applyStatus({ ...body, state: "starting" }); show("progress"); startPolling(); } catch (error) { interactive("#generation-status").textContent = error.message || "Interactive Voices generation could not start."; } finally { generationRequestInFlight = false; if (!pollTimer) setGenerationControlsDisabled(false); }
  };

  const restoreInteractiveRecovery = async (body) => {
    if (body?.job?.mode !== "interactive_voices") return;
    setInteractiveMode(true);
    try {
      const analysisResponse = await fetch("/api/voice-analysis/status");
      if (analysisResponse.ok) { voiceAnalysis = await analysisResponse.json(); renderAnalysisProgress(voiceAnalysis); }
      await loadVoices();
      await loadVoicePlan();
      interactive("#interactive-editor").hidden = false;
      interactive("#interactive-analysis-status").textContent = "Recovered Interactive Voices plan. Review the saved cast before continuing.";
      updateInteractiveReadiness();
    } catch (error) {
      interactive("#interactive-editor").hidden = false;
      interactive("#interactive-analysis-status").textContent = error.message || "Saved Interactive Voices review could not be restored.";
    }
  };

  const loadStatus = async () => {
    try { const response = await fetch("/api/status"); if (!response.ok) return; const body = await response.json(); applyStatus(body); await restoreInteractiveRecovery(body); }
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
  document.querySelectorAll("#chapter-start, #chapter-end").forEach((control) => control.addEventListener("input", () => { updateChapterRange(); updateModeReadiness(); }));
  document.querySelector("#regenerate-plan").addEventListener("click", async () => { const selection = currentPlanSpec(); await runPlanRequest(selection.mode, selection.mode === "custom" ? selection.count : undefined); });
  document.querySelector("#save-labels").addEventListener("click", () => saveLabels()); document.querySelector("#start-generation").addEventListener("click", startGeneration);
  document.querySelectorAll("input[name=voice]").forEach((control) => control.addEventListener("change", () => { document.querySelectorAll(".voice-card").forEach((card) => card.classList.toggle("selected", card.querySelector("input").checked)); }));
  document.querySelectorAll(".preview-voice").forEach((button) => button.addEventListener("click", () => { if (button.textContent === "Stop preview") stopPreview(); else previewVoice(button.dataset.voice, button); }));
  document.querySelector("#speed").addEventListener("input", (event) => { document.querySelector("#speed-value").textContent = `${Number(event.target.value).toFixed(1)}x`; });
  document.querySelector("#cancel-generation").addEventListener("click", () => cancelGeneration(document.querySelector("#cancel-generation"), document.querySelector("#progress-status")));
  document.querySelectorAll('input[name="voice-mode"]').forEach((control) => control.addEventListener("change", () => setInteractiveMode(control.value === "interactive_voices")));
  document.querySelector("#start-voice-analysis").addEventListener("click", startVoiceAnalysis);
  document.querySelector("#cancel-voice-analysis").addEventListener("click", cancelVoiceAnalysis);
  document.querySelector("#approve-voice-plan").addEventListener("click", approveVoicePlan);
  document.querySelector("#accept-narrator-fallback").addEventListener("change", updateInteractiveReadiness);
  document.querySelector("#reload-voice-plan").addEventListener("click", async () => { try { await loadVoices(); await loadVoicePlan(); interactive("#interactive-plan-status").textContent = "Voice plan refreshed."; } catch (error) { interactive("#interactive-plan-status").textContent = error.message || "Voice plan could not be refreshed."; } });
  const aliasDisclosure = document.querySelector("#interactive-aliases-disclosure");
  aliasDisclosure.addEventListener("toggle", () => { if (aliasDisclosure.open) renderAliasRows(); else interactive("#interactive-aliases").replaceChildren(); });
  ["#span-chapter-filter", "#span-confidence-filter"].forEach((selector) => document.querySelector(selector).addEventListener("change", async () => { spanOffset = 0; try { await loadVoicePlan(); } catch (error) { interactive("#span-filter-status").textContent = error.message || "Speaker spans could not be loaded."; } }));
  document.querySelector("#load-more-spans").addEventListener("click", async () => { try { await loadVoicePlan(true); } catch (error) { interactive("#span-filter-status").textContent = error.message || "More speaker spans could not be loaded."; } });
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

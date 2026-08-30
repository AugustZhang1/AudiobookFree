from pathlib import Path
from html.parser import HTMLParser


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "src" / "pdf_audiobook" / "static"


def test_interactive_mode_is_optional_and_single_voice_stays_default() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert '<input type="radio" name="voice-mode" value="single" checked>' in html
    assert 'name="voice-mode" value="interactive_voices"' in html
    assert 'value="af_heart" checked' in html
    assert "Start Interactive Voice Analysis" in html
    assert "const startInteractiveGeneration = async" in script
    assert 'if (interactiveMode) { await startInteractiveGeneration(); return; }' in script
    assert 'engine: "chatterbox"' in script and 'model: "nano"' in script


def test_single_engine_selection_is_exclusive_and_generation_payload_is_branch_safe() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    generation_start = script.index("const startGeneration = async")
    generation_end = script.index("  const previewStatusFor =", generation_start)
    generation = script[generation_start:generation_end]
    chatterbox_branch = generation.index('if (singleEngine === "chatterbox")')
    voice_lookup = generation.index('document.querySelector("input[name=voice]:checked")')
    assert voice_lookup > chatterbox_branch
    assert "let payload;" in generation
    assert 'payload = { engine: "chatterbox", model: "nano", voice: chatterboxVoice' in generation
    assert 'if (!voiceControl) throw new Error("Select a Kokoro voice before generating.");' in generation
    assert generation.count("JSON.stringify(payload)") == 1

    engine_start = script.index("const setSingleEngine =")
    engine_end = script.index("  const uploadChatterboxReference =", engine_start)
    engine_switch = script[engine_start:engine_end]
    assert 'let rememberedKokoroVoice = document.querySelector(\'input[name="voice"]:checked\')?.value || "af_heart";' in script
    assert "if (currentKokoro) rememberedKokoroVoice = currentKokoro.value;" in engine_switch
    assert "kokoroControls.forEach((control) => { control.checked = false; });" in engine_switch
    assert 'document.querySelectorAll(".voice-card").forEach((card) => card.classList.remove("selected"));' in engine_switch
    assert "const restoredKokoro = kokoroControls.find((control) => control.value === rememberedKokoroVoice) || kokoroControls[0];" in engine_switch
    assert "control === restoredKokoro" in engine_switch
    assert 'card.querySelector(\'input[name="voice"]\') === restoredKokoro' in engine_switch
    assert "if (control.checked) rememberedKokoroVoice = control.value;" in script


def test_chapter_range_status_uses_selected_plan_titles_with_safe_fallback() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "const chapterTitle = (index) =>" in script
    assert "chapterPlan?.chapters?.[index - 1]?.title" in script
    assert "typeof title === \"string\" && title.trim()" in script
    assert "return typeof title === \"string\" && title.trim() ? title.trim() : `Chapter ${index}`;" in script
    assert "${chapterTitle(range.start)} selected (inclusive)." in script
    assert "chapters selected: ${chapterTitle(range.start)} through ${chapterTitle(range.end)}" in script


def test_generation_readiness_routes_by_selected_voice_mode() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "const updateSingleVoiceReadiness = () =>" in script
    assert "const updateModeReadiness = () =>" in script
    assert "if (interactiveMode) updateInteractiveReadiness();" in script
    assert "else updateSingleVoiceReadiness();" in script
    assert "!generationRequestInFlight && !planRequestInFlight" in script
    assert "!activeGenerationStates.has(recoveredState)" in script

    single_start = script.index("const updateSingleVoiceReadiness =")
    single_end = script.index("const updateInteractiveReadiness =", single_start)
    assert "planMatchesSelection(chapterPlan)" in script[single_start:single_end]
    assert "voicePlan" not in script[single_start:single_end]

    handler_start = script.index('document.querySelectorAll("#chapter-start, #chapter-end")')
    handler_end = script.index('document.querySelector("#regenerate-plan")', handler_start)
    range_handler = script[handler_start:handler_end]
    assert "updateChapterRange(); updateModeReadiness();" in range_handler
    assert "updateInteractiveReadiness();" not in range_handler


def test_interactive_analysis_and_review_endpoint_contract_is_static() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for endpoint in (
        "/api/voices",
        "/api/voice-analysis",
        "/api/voice-analysis/status",
        "/api/voice-analysis/cancel",
        "/api/voice-plan/draft",
        "/api/voice-plan?",
        "/api/voice-plan",
        "/api/voice-plan/aliases/merge",
        "/api/voice-plan/aliases/split",
        "/api/voice-plan/spans/override",
        "/api/voice-plan/approve",
        "/api/voice-plan/cast/remove",
        "/api/voice-plan/cast/merge",
    ):
        assert endpoint in script
    assert 'JSON.stringify({ mode: "interactive", chapter_start: range.start, chapter_end: range.end })' in script
    assert "const range = updateChapterRange();" in script
    assert "Choose a valid inclusive chapter range before starting voice analysis." in script
    assert "if (!range.valid)" in script
    assert "analysis_revision: revision" in script
    assert "expected_revision: voicePlanRevision" in script
    assert "kind: kind.value" in script and "reason: reason.value.trim()" in script
    assert "accept_narrator_fallback: accepted" in script
    assert "chapter" in script and "confidence" in script and "limit" in script and "offset" in script
    assert "Remove voice" in script and "cast-remove" in script
    assert "Merge into" in script and "cast-merge" in script and "cast-merge-target" in script
    assert "cast_id: entry.cast_id" in script
    assert "source_cast_id: entry.cast_id" in script and "target_cast_id: targetId" in script


def test_interactive_review_accessibility_and_v5_payload_are_static() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for marker in (
        'aria-live="polite"',
        'aria-label="Filter speaker spans"',
        'aria-label="Interactive voice analysis progress"',
        'id="accept-narrator-fallback"',
        "I reviewed unresolved spans and accept Narrator as their fallback.",
        "role=\"progressbar\"",
    ):
        assert marker in html
    assert "review?.unresolved_count" in script
    assert "Approve the Interactive Voices plan before generating." in script
    assert 'mode: "interactive_voices"' in script
    assert "voice_plan_sha256: voicePlan.canonical_artifact_sha256" in script
    assert "voice_plan_revision: voicePlan.revision" in script
    assert "chapter_start: range.start" in script and "chapter_end: range.end" in script
    assert "performance_mode" in script
    assert "ETA" not in script and "remaining" not in script.lower()


def test_alias_review_uses_collapsed_native_disclosure_and_lazy_rows() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    details_start = html.index('<details id="interactive-aliases-disclosure"')
    details_tag = html[details_start : html.index(">", details_start) + 1]
    assert " open" not in details_tag
    assert 'id="interactive-aliases-summary"' in html
    assert 'aria-controls="interactive-aliases"' in html
    assert '<div id="interactive-aliases" class="alias-list"' in html

    for marker in (
        "const renderAliasRows = () =>",
        "alias_count",
        "aliases_truncated",
        "only the loaded subset is shown when expanded",
        "if (disclosure?.open) renderAliasRows();",
        'addEventListener("toggle"',
        "aliasDisclosure.open",
    ):
        assert marker in script
    assert "No aliases detected" in script


def test_interactive_recovery_locks_v5_and_restores_saved_plan() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "body.job?.schema_version === 4 || body.job?.schema_version === 5" in script
    assert '#view-configure input, #view-configure button, #view-configure select' in script
    assert "const restoreInteractiveRecovery = async" in script
    assert 'body?.job?.mode !== "interactive_voices"' in script
    assert 'await loadVoices();' in script and 'await loadVoicePlan();' in script
    assert 'const analysisResponse = await fetch("/api/voice-analysis/status")' in script
    assert 'voiceAnalysis = await analysisResponse.json(); renderAnalysisProgress(voiceAnalysis);' in script
    assert 'interactive("#interactive-editor").hidden = false' in script
    assert 'Saved Interactive Voices review could not be restored.' in script
    assert 'entry.display_label || entry.display_name || entry.label || entry.name || entry.id' in script
    assert 'entry.description.trim()' in script
    assert 'if (interactiveMode) updateInteractiveReadiness();' in script


def test_interactive_css_is_responsive_and_focus_visible() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert ".interactive-review" in css
    assert ".voice-mode-choice" in css
    assert ".cast-fields" in css and ".span-override" in css
    assert ".alias-disclosure" in css and ".alias-disclosure > summary" in css
    assert ".review-grid { display:grid; grid-template-columns:1fr;" in css
    assert html.index('id="interactive-cast"') < html.index('id="interactive-aliases-disclosure"') < html.index('id="interactive-spans"')
    assert "@media (max-width:800px)" in css
    assert "@media (max-width:600px)" in css
    assert ":focus-visible" in css


def test_preview_buttons_have_stable_affordance_and_bounded_labels() -> None:
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    preview_start = css.index(".preview-voice {")
    preview_rule = css[preview_start : css.index(".settings-row", preview_start)]
    for marker in (
        "display:inline-flex",
        "min-height:30px",
        "border:1px solid #b7c4be",
        "border-radius:2px",
        "padding:0 10px",
        "max-width:100%",
        "overflow:hidden",
        "text-overflow:ellipsis",
        "text-decoration:none",
    ):
        assert marker in preview_rule
    assert ".preview-voice:hover {" in css
    assert ".preview-voice:focus-visible {" in css
    assert ".interactive-preview { width:128px; min-width:128px; min-height:30px; }" in css


def test_interactive_cast_controls_use_aligned_rows_and_mobile_fallback() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'relationshipActions.className = "cast-action-row cast-primary-actions"' in script
    assert 'relationshipLabel.textContent = "Relationship"' in script
    assert 'relationshipLabel.append(relationship)' in script
    assert 'characterActions.className = "cast-action-row cast-character-actions"' in script
    assert 'mergeLabel.append(mergeTarget)' in script
    assert 'characterActions.append(mergeLabel, merge, remove)' in script
    assert "previewControls.append(preview, previewStatus); voiceField.append(previewControls)" in script
    assert "fields.append(nameLabel, voiceField, speedLabel)" in script
    assert "voiceField.append(voiceLabel)" in script
    assert ".cast-action-row { display:grid; align-items:end;" in css
    assert ".cast-primary-actions { grid-template-columns:minmax(0,1fr) auto auto; }" in css
    assert ".cast-character-actions { grid-template-columns:minmax(0,1fr) auto auto; }" in css
    cast_fields_start = css.index(".cast-fields {")
    cast_fields_rule = css[cast_fields_start : css.index("}", cast_fields_start) + 1]
    assert "align-items:start;" in cast_fields_rule
    assert "align-items:stretch" not in cast_fields_rule
    assert ".cast-preview { display:grid; grid-template-columns:minmax(128px,auto) minmax(0,1fr); align-items:center; gap:8px;" in css
    assert "@media (max-width:600px) { .cast-preview,.cast-primary-actions,.cast-character-actions { grid-template-columns:1fr; }" in css


def test_interactive_analysis_range_gates_review_and_generation() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "let analyzedChapterRange = null;" in script
    assert "const analysisComplete = voiceAnalysis?.status === \"completed\";" in script
    assert "rangeAnalyzed = Boolean(analysisComplete && range.valid && analyzedChapterRange" in script
    assert "range.start >= analyzedChapterRange.start && range.end <= analyzedChapterRange.end" in script
    assert "outside the analyzed range" in script
    assert "The selected range was not analyzed. Start Interactive Voice Analysis again before approving." in script
    assert 'fetch("/api/voice-analysis/status")' in script
    assert "body.chapter_start" in script and "body.chapter_end" in script
    progress_start = script.index("const renderAnalysisProgress =")
    progress_end = script.index("const stopVoiceAnalysisPolling =", progress_start)
    assert "updateModeReadiness();" in script[progress_start:progress_end]


def test_interactive_cast_preview_starts_play_synchronously_and_guards_requests() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "/api/voice-preview/${encodeURIComponent(voice)}/prepare" not in script
    assert "preparePreviewVoice" not in script
    preview_start = script.index("const previewVoice =")
    preview_end = script.index("  const setReferenceStatus =", preview_start)
    preview_path = script[preview_start:preview_end]
    assert "/prepare" not in preview_path
    assert "await" not in preview_path
    audio_start = preview_path.index("new Audio(`/api/voice-preview/${encodeURIComponent(voice)}`)")
    play_start = preview_path.index("audio.play()")
    assert audio_start < play_start
    assert "playPromise.catch" in preview_path
    assert "if (requestId !== previewRequest) return;" in preview_path
    assert 'voice.addEventListener("change"' in script
    assert "if (activePreviewButton === preview) stopPreview(previewStatus);" in script
    assert "activePreviewButton === preview" in script


def test_interactive_cast_preview_is_unsaved_and_accessible() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    handler_start = script.index('preview.addEventListener("click"')
    handler_end = script.index('voice.addEventListener("change"', handler_start)
    preview_handler = script[handler_start:handler_end]

    assert "voice.value" in preview_handler
    assert "previewVoice(voice.value, preview, previewStatus)" in preview_handler
    assert "mutateVoicePlan" not in preview_handler
    assert "Save cast" not in preview_handler
    assert 'preview.className = "preview-voice interactive-preview"' in script
    assert 'preview.setAttribute("aria-label", `Preview ${entry.display_label} voice`)' in script
    assert 'previewStatus.setAttribute("role", "status")' in script
    assert 'previewStatus.setAttribute("aria-live", "polite")' in script
    assert 'targetStatus.textContent = "Finished."' in script
    assert 'stoppedStatus.textContent = "Stopped."' in script
    assert 'document.querySelectorAll(".preview-voice")' in script
    assert 'button.textContent = button.classList.contains("interactive-preview") ? "Preview draft" : "Preview";' in script
    assert 'const referencePreview = document.querySelector("#reference-preview")' in script
    assert 'referencePreview.textContent = "Preview reference"' in script
    assert 'const replace = Boolean(chatterboxReference?.available);' in script
    assert script.count('stopPreview();') >= 4


def test_interactive_cast_drafts_preserve_labels_save_complete_settings_and_reset_only_shaping() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'const hasDraft = castDrafts.has(entry.cast_id);' in script
    assert 'const draft = hasDraft ? castDrafts.get(entry.cast_id) : {};' in script
    assert 'unsaved.textContent = hasDraft ? "Unsaved changes" : "";' in script
    assert 'button.classList.contains("interactive-preview") ? "Preview draft" : "Preview"' in script
    assert 'voice_settings: { speed: Number(speed.value), pitch_semitones: Number(pitch.value), tone_preset: toneValue() }' in script
    assert 'castDrafts.delete(entry.cast_id); renderCast();' in script
    assert 'speed: 1.0, pitch_semitones: 0, tone_preset: "neutral"' in script
    assert 'fields.className = "cast-fields cast-shaping-grid"' in script
    assert 'radio.type = "radio"' in script
    assert '@media (max-width:1000px) and (min-width:601px)' in css
    assert '.tone-segmented label:focus-within' in css


def test_single_voice_chatterbox_controls_are_opt_in_and_reference_bound() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup_windows.ps1").read_text(encoding="utf-8")
    helper = (ROOT / "scripts" / "setup_chatterbox.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in ("single-engine", "Chatterbox Nano", "Chatterbox Nano voice", "Built-in Chatterbox voice", "Custom cloned voice", "reference-consent", "reference-input", "reference-upload", "reference-revoke", "reference-preview", "longer than 5 seconds", "25 MiB", "watermark"):
        assert marker in html
    for marker in ("/api/engines", "/api/chatterbox/reference", "/api/chatterbox/preview", 'engine: "chatterbox"', 'model: "nano"', 'voice: chatterboxVoice', 'speed: 1'):
        assert marker in script
    assert 'singleEngine === "chatterbox"' in script and 'chatterboxVoice = "builtin"' in script
    assert 'body.job?.mode !== "interactive_voices"' in script
    assert "WithChatterbox" in setup and "setup_chatterbox.py" in setup
    assert "nano=True" in helper and "HF_HUB_OFFLINE" in helper
    assert "Optional Chatterbox Nano" in readme and "Interactive Voices remains Kokoro-only" in readme


def test_chatterbox_controls_are_optional_for_legacy_cached_html() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'const chatterboxUi = document.querySelector("#chatterbox-reference-panel");' in script
    assert "if (!chatterboxUi) { chatterboxInstalled = false; return Promise.resolve(); }" in script
    assert "if (!chatterboxUi) { singleEngine = \"kokoro\"; return; }" in script
    assert 'if (chatterboxUi) {' in script
    assert 'document.querySelector("#reference-consent").addEventListener' not in script
    assert 'document.querySelector("#reference-input").addEventListener' not in script


def test_builtin_chatterbox_preview_is_a_non_nested_card_control() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert '<script src="/app.js?v=chatterbox-preview-v5"></script>' in html
    assert '<script src="/app.js"></script>' not in html

    class PreviewPlacementParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.label_depth = 0
            self.preview_inside_label = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "label":
                self.label_depth += 1
            if tag == "button" and attrs.get("id") == "builtin-preview" and self.label_depth:
                self.preview_inside_label = True

        def handle_endtag(self, tag):
            if tag == "label":
                self.label_depth -= 1

    parser = PreviewPlacementParser()
    parser.feed(html)
    assert not parser.preview_inside_label
    assert 'id="builtin-preview"' in html and 'id="builtin-preview-status"' in html
    assert "/api/chatterbox/preview/builtin" in script
    assert "const loadEngineCatalog = (force = false) =>" in script
    assert "if (engineCatalogLoaded && !force) return Promise.resolve();" in script
    assert 'entry.engine_id === "chatterbox" && entry.model_id === "nano"' in script
    assert 'entry.engine === "chatterbox"' not in script
    assert "const BUILTIN_PREVIEW_WATCHDOG_MS = 120000;" in script
    assert "let previewWatchdogTimer = null;" in script
    assert "const clearPreviewWatchdog" in script
    assert "clearTimeout(previewWatchdogTimer)" in script
    assert "const startBuiltinPreviewWatchdog" in script
    assert "setTimeout" in script
    assert "Nano preview took too long to start. Nano may continue warming the shared cache; you can try again." in script
    assert "const teardownPreviewAudio = (audio) =>" in script
    assert 'audio.removeAttribute("src"); audio.src = "";' in script
    assert "audio.load()" in script
    assert "teardownPreviewAudio(activePreview)" in script
    assert "clearPreviewWatchdog();" in script
    assert "let previewElapsedTimer = null;" in script
    assert "const clearPreviewElapsedTimer" in script
    assert "const startBuiltinPreviewElapsed" in script
    assert "timer = setInterval(update, 1000)" in script
    assert "Loading Nano on CPU… ${minutes}m ${seconds}s. First preview can take several minutes; later previews are cached." in script
    assert "Preview stopped in browser. Nano may continue warming the shared cache in the background; the next attempt will reuse it." in script
    builtin_start = script.index("const previewChatterboxBuiltin")
    builtin_end = script.index("  const openOutput", builtin_start)
    builtin_path = script[builtin_start:builtin_end]
    assert "const previewChatterboxBuiltin = async () =>" in builtin_path
    assert 'new Audio("/api/chatterbox/preview/builtin")' in builtin_path
    assert "audio.play()" in builtin_path and builtin_path.index("new Audio") < builtin_path.index("audio.play()")
    assert 'if (activePreviewButton === button) { stopPreview(targetStatus); return; }' in builtin_path
    assert "startBuiltinPreviewElapsed(button, targetStatus, requestId)" in builtin_path
    assert "startBuiltinPreviewWatchdog(button, targetStatus, requestId)" in builtin_path
    assert "clearPreviewElapsedTimer();" in builtin_path
    assert "clearPreviewWatchdog();" in builtin_path
    assert "if (requestId !== previewRequest) return;" in builtin_path
    assert 'button.textContent = "Cancel preview";' in builtin_path
    assert "if (!chatterboxInstalled) {" in builtin_path
    assert "try { await loadEngineCatalog(true); } catch (_) { /* Keep the cached unavailable state. */ }" in builtin_path
    assert "if (!chatterboxInstalled) { if (targetStatus) targetStatus.textContent = \"Optional Nano engine was not detected. Run setup and restart the app.\"; return; }" in builtin_path
    assert "if (activeGenerationStates.has(recoveredState)) { if (targetStatus) targetStatus.textContent = \"Nano preview is unavailable while audiobook generation is active.\"; return; }" in builtin_path
    assert "generationRequestInFlight" not in builtin_path
    audio_start = builtin_path.index('new Audio("/api/chatterbox/preview/builtin")')
    assert builtin_path.index("await loadEngineCatalog(true)") < audio_start
    assert builtin_path.index("if (!chatterboxInstalled)") < audio_start
    assert builtin_path.index("if (activeGenerationStates.has(recoveredState))") < audio_start
    assert 'document.querySelectorAll(".preview-voice:not(.chatterbox-preview)")' in script
    assert 'if (control.checked) { stopPreview(); setSingleEngine(control.value); }' in script
    assert 'if (control.checked) { stopPreview(); chatterboxVoice = control.value; setSingleEngine("chatterbox"); }' in script
    assert 'document.querySelectorAll("input[name=voice]").forEach((control) => control.addEventListener("change", () => { if (control.checked) rememberedKokoroVoice = control.value; stopPreview();' in script

    readiness_start = script.index("const updateSingleVoiceReadiness =")
    readiness_end = script.index("const updateInteractiveReadiness =", readiness_start)
    readiness = script[readiness_start:readiness_end]
    assert "updateBuiltinPreviewDisabled" not in script
    assert "updateBuiltinPreviewDisabled" not in readiness

    controls_start = script.index("const setGenerationControlsDisabled =")
    controls_end = script.index("const formatElapsed =", controls_start)
    controls = script[controls_start:controls_end]
    assert 'document.querySelectorAll("#view-configure input, #view-configure button, #view-configure select")' in controls
    assert 'const builtinPreview = document.querySelector("#builtin-preview");' in controls
    assert "if (builtinPreview) builtinPreview.disabled = false;" in controls

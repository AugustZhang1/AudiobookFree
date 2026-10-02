from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "colab" / "PDF_Audiobook_Colab.ipynb"


def _load_notebook() -> dict:
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    return json.loads(content)


def _get_setup_cell(data: dict) -> dict:
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", []))
        if "# Colab's runtime Python may be 3.12" in source:
            return cell
    raise AssertionError("Setup cell not found in notebook")


def test_notebook_is_valid_json() -> None:
    data = _load_notebook()
    assert isinstance(data, dict)
    assert "cells" in data


def test_setup_cell_branch_main_not_morefeatures() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_text = "".join(setup_cell.get("source", []))
    assert "BRANCH = 'main'" in source_text
    assert "morefeatures" not in source_text


def test_setup_cell_torch_mirror_before_kokoro() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_lines = setup_cell.get("source", [])
    torch_mirror_indices = [i for i, line in enumerate(source_lines) if "--index-url {TORCH_INDEX}" in line]
    kokoro_indices = [i for i, line in enumerate(source_lines) if "'kokoro==0.9.4'" in line]

    assert len(torch_mirror_indices) == 1
    assert len(kokoro_indices) == 1
    assert torch_mirror_indices[0] < kokoro_indices[0]


def test_setup_cell_kokoro_line_contains_torch_version() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_lines = setup_cell.get("source", [])
    kokoro_lines = [line for line in source_lines if "'kokoro==0.9.4'" in line]
    assert len(kokoro_lines) == 1
    assert "torch=={TORCH_VERSION}" in kokoro_lines[0]


def test_setup_cell_raises_when_gpu_check_fails() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_text = "".join(setup_cell.get("source", []))
    assert "cannot see the GPU" in source_text
    assert "raise RuntimeError" in source_text


def _get_panel_cell(data: dict) -> dict:
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", []))
        if "voice_picker" in source and "render_previews" in source:
            return cell
    raise AssertionError("Panel cell not found in notebook")


def _get_generate_cell(data: dict) -> dict:
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", []))
        if "# One long, resumable headless run" in source:
            return cell
    raise AssertionError("Generate cell not found in notebook")


def test_panel_cell_calls_cli_and_parses_preview_wav() -> None:
    data = _load_notebook()
    panel_cell = _get_panel_cell(data)
    source = "".join(panel_cell.get("source", []))
    assert "--preview-dir" in source
    assert "--preview-text" in source
    assert "Preview WAV " in source


def test_no_cell_contains_preview_out_or_param() -> None:
    data = _load_notebook()
    for i, cell in enumerate(data.get("cells", [])):
        source = "".join(cell.get("source", []))
        assert "--preview-out" not in source, f"Cell {i} contains '--preview-out'"
        assert "# @param" not in source, f"Cell {i} contains '# @param'"


def test_generate_cell_reads_panel_values() -> None:
    data = _load_notebook()
    generate_cell = _get_generate_cell(data)
    source = "".join(generate_cell.get("source", []))
    assert "voice_picker.value" in source
    assert "start_new_box.value" in source


def test_panel_cell_contains_all_approved_voice_ids() -> None:
    from pdf_audiobook.voice_registry import APPROVED_VOICE_IDS

    data = _load_notebook()
    panel_cell = _get_panel_cell(data)
    source = "".join(panel_cell.get("source", []))
    for voice_id in APPROVED_VOICE_IDS:
        assert voice_id in source, f"Voice {voice_id} missing from panel cell source"


def test_every_code_cell_compiles() -> None:
    data = _load_notebook()
    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        filtered_lines = [line for line in lines if not line.strip().startswith("!")]
        code_text = "".join(filtered_lines)
        compile(code_text, f"<cell_{i}>", "exec")

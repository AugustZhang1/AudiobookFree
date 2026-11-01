from __future__ import annotations

import json
from pathlib import Path
import subprocess

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
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=NOTEBOOK_PATH.parent, capture_output=True, text=True).stdout.strip() or "main"
    assert f"BRANCH = '{branch}'" in source_text, f"Notebook BRANCH must match the current git branch ({branch})"
    assert "morefeatures" not in source_text


def test_setup_cell_torch_mirror_before_kokoro() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_lines = setup_cell.get("source", [])
    torch_mirror_indices = [i for i, line in enumerate(source_lines) if "--index-url {TORCH_INDEX}" in line]
    kokoro_indices = [i for i, line in enumerate(source_lines) if "-r {REPO}/colab/requirements.in" in line]

    assert len(torch_mirror_indices) == 1
    assert len(kokoro_indices) == 1
    assert torch_mirror_indices[0] < kokoro_indices[0]


def test_setup_cell_kokoro_line_contains_torch_version() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_lines = setup_cell.get("source", [])
    kokoro_lines = [line for line in source_lines if "-r {REPO}/colab/requirements.in" in line]
    assert len(kokoro_lines) == 1
    assert "torch=={TORCH_VERSION}" in kokoro_lines[0]


def test_setup_cell_raises_when_gpu_check_fails() -> None:
    data = _load_notebook()
    setup_cell = _get_setup_cell(data)
    source_text = "".join(setup_cell.get("source", []))
    assert "cannot see the GPU" in source_text
    assert "raise RuntimeError" in source_text
    # A failed Kokoro install must stop setup, not surface later at preview time.
    assert "import torch, kokoro;" in source_text


def test_setup_cell_constrains_every_install_except_torch() -> None:
    source_lines = _get_setup_cell(_load_notebook()).get("source", [])
    installs = [line for line in source_lines if line.startswith("!uv pip install --python {PY}")]
    assert len(installs) == 3
    for line in installs:
        assert ("--index-url {TORCH_INDEX}" in line) != ("-c {REPO}/colab/constraints.txt" in line), line


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith((" ", "#")):
            for separator in ("==", "<", " @ "):
                line = line.split(separator)[0]
            names.add(line.strip().lower().replace("_", "-"))
    return names


def test_constraints_pin_every_requirement_and_leave_torch_to_colab() -> None:
    colab_dir = NOTEBOOK_PATH.parent
    pinned = _requirement_names(colab_dir / "constraints.txt")
    assert _requirement_names(colab_dir / "requirements.in") <= pinned, "regenerate colab/constraints.txt (command in its header)"
    assert not pinned & _requirement_names(colab_dir / "excludes.txt")
    assert not {name for name in pinned if name == "triton" or name.startswith(("nvidia-", "cuda-"))}


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


def test_run_all_never_blocks_on_a_file_upload() -> None:
    # "Run all" must finish so the panel's preview buttons respond; uploads live in panel widgets.
    for i, cell in enumerate(_load_notebook().get("cells", [])):
        assert "files.upload(" not in "".join(cell.get("source", [])), f"Cell {i} blocks Run all on an upload"


def test_panel_adds_voices_and_takes_the_book() -> None:
    source = "".join(_get_panel_cell(_load_notebook()).get("source", []))
    assert "widgets.FileUpload(accept='audio/*,video/*'" in source
    assert "add_voice_button.on_click(add_voices)" in source
    assert "book_upload = widgets.FileUpload(" in source
    assert "'--clone'" in source


def test_custom_voice_model_reaches_preview_and_generation() -> None:
    data = _load_notebook()
    panel = "".join(_get_panel_cell(data).get("source", []))
    generate = "".join(_get_generate_cell(data).get("source", []))
    assert "clone_model_picker = widgets.Dropdown(" in panel and "cmd.extend(clone_args())" in panel
    assert "cmd += clone_args()" in generate


def test_character_voices_are_an_opt_in_dropdown_at_the_top() -> None:
    data = _load_notebook()
    panel = "".join(_get_panel_cell(data).get("source", []))
    generate = "".join(_get_generate_cell(data).get("source", []))
    assert "narration_picker = widgets.Dropdown(" in panel
    assert "value='single'," in panel.split("narration_picker = widgets.Dropdown(", 1)[1].split("\n)", 1)[0]
    assert "panel = widgets.VBox([\n    narration_picker," in panel
    assert "cast_section = widgets.VBox(" in panel and "layout=widgets.Layout(display='none')" in panel.split("cast_section = widgets.VBox(", 1)[1]
    assert "make_script_button.on_click(make_script)" in panel
    assert "preview_cast_button.on_click(preview_cast)" in panel and "'--script', 'preview.txt'" in panel
    assert "cast_tools.recommend_voices(" in panel and "own_voice_from = widgets.BoundedIntText(" in panel
    assert "design_button.on_click(design_one)" in panel and "'pdf_audiobook.voice_design', '--batch'" in panel
    assert "design_cast_box = widgets.Checkbox(\n    value=True," in panel and "suggested.update(cast_state.get('designed', {}))" in panel
    assert "if narration_picker.value == 'cast':" in generate
    for step in ("'render'", "'--engine', 'kokoro'", "'--engine', 'clone'", "'build'"):
        assert step in generate

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

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import uuid

import pytest

from pdf_audiobook.analyzers.booknlp import (
    BOOKNLP_VERSION,
    MAX_RESULT_BYTES,
    BookNLPAnalyzer,
    BookNLPAnalyzerError,
    parse_booknlp_output,
)
from pdf_audiobook.analyzers import booknlp as booknlp_module
from pdf_audiobook.analyzers import booknlp_runner
from pdf_audiobook.analysis_runner import VoiceAnalysisCancelled

SOURCE_HASH = "a" * 64


def _plan(text: str, split: int | None = None) -> dict:
    split = len(text) if split is None else split
    chapters = [{"index": 1, "start_offset": 0, "end_offset": split}]
    if split < len(text):
        chapters.append({"index": 2, "start_offset": split, "end_offset": len(text)})
    return {"chapters": chapters}


def _token_rows(text: str) -> list[dict]:
    rows = []
    byte_position = 0
    token_id = 0
    for part in text.split(" "):
        if not part:
            byte_position += 1
            continue
        start = byte_position
        end = start + len(part.encode("utf-8"))
        rows.append({"id": token_id, "word": part, "byte_start": start, "byte_end": end})
        token_id += 1
        byte_position = end + 1
    return rows


def _payload(text: str, *, quote=(0, 0, 4), coref_id=7) -> dict:
    return {
        "schema_version": 1,
        "booknlp_version": BOOKNLP_VERSION,
        "tokens": _token_rows(text),
        "quotes": [{"quote_id": "q0", "start_token": quote[0], "end_token": quote[1], "coref_id": quote[2] if len(quote) > 2 else coref_id}],
        "characters": [{"coref_id": coref_id, "canonical_label": "Zoë", "aliases": [{"alias": "Zoë", "kind": "proper", "token_start": 0, "token_end": 1}], "quote_count": 1}] if coref_id >= 0 else [],
        "warnings": [],
    }


@dataclass
class _Result:
    returncode: int
    stdout: str
    stderr: str = ""


def _adapter(payload: dict, *, command_runner=None, **kwargs) -> BookNLPAnalyzer:
    fake = command_runner or (lambda argv, **options: _Result(0, json.dumps(payload)))
    # The managed test environment denies writes below the OS temp directory;
    # this remains a private per-run directory owned by the adapter.
    return BookNLPAnalyzer(command_runner=fake, temp_directory=Path("tests"), **kwargs)


def test_exact_unicode_punctuation_offsets_and_quote_speaker() -> None:
    text = '“Come, Zoë!” said.'
    # The fixture emulates BookNLP's byte offsets, including multibyte quotes.
    payload = _payload(text, quote=(0, 0, 7), coref_id=7)
    payload["tokens"] = [{"id": 0, "word": text, "byte_start": 0, "byte_end": len(text.encode())}]
    result = _adapter(payload).analyze(text, _plan(text), SOURCE_HASH)
    span = result.spans[0]
    assert text[span.source_start:span.source_end] == text
    assert span.chapter_index == 1
    assert span.speaker_id == "booknlp:7"
    assert span.span_type == "dialogue"


def test_chapter_mapping_and_unknown_speaker_remain_reviewable() -> None:
    text = 'First “Hi”\nSecond “Bye”'
    # Tokens split at spaces; use explicit records to make chapter boundaries clear.
    payload = _payload(text, coref_id=-1)
    payload["tokens"] = [
        {"id": 0, "word": 'First', "byte_start": 0, "byte_end": 5},
        {"id": 1, "word": '“Hi”', "byte_start": 6, "byte_end": 14},
        {"id": 2, "word": 'Second', "byte_start": 15, "byte_end": 21},
        {"id": 3, "word": '“Bye”', "byte_start": 22, "byte_end": 31},
    ]
    payload["quotes"] = [{"quote_id": "q0", "start_token": 1, "end_token": 1, "coref_id": -1}, {"quote_id": "q1", "start_token": 3, "end_token": 3, "coref_id": -1}]
    result = _adapter(payload).analyze(text, _plan(text, text.index("Second")), SOURCE_HASH)
    assert [(span.chapter_index, span.span_type, span.speaker_id) for span in result.spans] == [(1, "unknown", None), (2, "unknown", None)]


def test_quote_structural_errors_are_skipped_individually_with_generic_warnings() -> None:
    text = "one two three four five"
    split = text.index("three")
    payload = _payload(text, coref_id=7)
    payload["quotes"] = [
        {"quote_id": "valid-first", "start_token": 0, "end_token": 0, "coref_id": 7},
        {"quote_id": "nested", "start_token": 0, "end_token": 1, "coref_id": 7},
        {"quote_id": "straddles", "start_token": 1, "end_token": 2, "coref_id": 7},
        {"quote_id": "malformed-range", "start_token": 3, "end_token": 2, "coref_id": 7},
        {"quote_id": "valid-last", "start_token": 3, "end_token": 3, "coref_id": 7},
    ]

    result = _adapter(payload).analyze(text, _plan(text, split), SOURCE_HASH)

    assert [span.span_id for span in result.spans] == ["booknlp:valid-first", "booknlp:valid-last"]
    assert len(result.warnings) == 3
    assert all(warning.startswith("BookNLP quote skipped:") for warning in result.warnings)
    assert all(len(warning) <= 256 and text not in warning for warning in result.warnings)


def test_byte_boundary_storage_is_bounded_by_requested_token_offsets() -> None:
    text = "é" * 100_000
    encoded = text.encode("utf-8")

    offsets, character_positions, returned = booknlp_module._byte_boundaries(text, (0, len(encoded)))

    assert len(offsets) == len(character_positions) == 2
    assert offsets == (0, len(encoded))
    assert character_positions == (0, len(text))
    assert returned == encoded


@pytest.mark.parametrize("payload_factory", [lambda text: {"bad": True}, lambda text: _payload(text, quote=(0, 5, 999))])
def test_malformed_output_is_rejected_without_detail_leak(payload_factory) -> None:
    text = "hello"
    with pytest.raises(BookNLPAnalyzerError) as exc:
        _adapter(payload_factory(text)).analyze(text, _plan(text), SOURCE_HASH)
    assert exc.value.code == "ANALYZER_OUTPUT_INVALID"
    assert text not in str(exc.value)


def test_runner_and_adapter_share_the_result_transport_ceiling() -> None:
    assert booknlp_runner.MAX_RESULT_BYTES == MAX_RESULT_BYTES == 64 * 1024 * 1024
    assert booknlp_runner.MAX_OUTPUT_BYTES == MAX_RESULT_BYTES


def test_token_heavy_contract_above_former_cap_completes_without_booknlp() -> None:
    text = ("token " * 150_000).rstrip()
    payload = _payload(text, quote=(0, 0, 7))
    encoded_size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert 8 * 1024 * 1024 < encoded_size < MAX_RESULT_BYTES
    result = _adapter(payload).analyze(text, _plan(text), SOURCE_HASH)
    assert len(result.spans) == 1
    assert result.spans[0].speaker_id == "booknlp:7"


def test_oversized_output_is_rejected() -> None:
    text = "hello"
    oversized = "{" + "x" * MAX_RESULT_BYTES + "}"
    analyzer = _adapter({}, command_runner=lambda argv, **options: _Result(0, oversized))
    with pytest.raises(BookNLPAnalyzerError) as exc:
        analyzer.analyze(text, _plan(text), SOURCE_HASH)
    assert exc.value.code == "OUTPUT_TOO_LARGE"
    assert str(exc.value) == "analyzer output exceeded the size limit"


def test_dedicated_runner_size_exit_is_mapped_without_detail_leak() -> None:
    text = "secret book text"
    analyzer = _adapter({}, command_runner=lambda argv, **options: _Result(booknlp_runner.OUTPUT_TOO_LARGE_EXIT_CODE, text))
    with pytest.raises(BookNLPAnalyzerError) as exc:
        analyzer.analyze(text, _plan(text), SOURCE_HASH)
    assert exc.value.code == "OUTPUT_TOO_LARGE"
    assert str(exc.value) == "analyzer output exceeded the size limit"
    assert text not in str(exc.value)


def test_timeout_is_sanitized_and_argv_has_no_shell() -> None:
    text = "hello"

    def timeout(argv, **options):
        assert options["shell"] is False
        raise subprocess.TimeoutExpired(argv, 0.01, output="secret book text")

    with pytest.raises(BookNLPAnalyzerError) as exc:
        _adapter({}, command_runner=timeout, timeout=0.01).analyze(text, _plan(text), SOURCE_HASH)
    assert exc.value.code == "ANALYZER_TIMEOUT"
    assert "secret" not in str(exc.value)


def test_runner_uses_utf8_python_mode_and_strict_stdout_decoding() -> None:
    text = '“Hello”'
    payload = _payload(text, quote=(0, 0, 7), coref_id=7)
    captured: dict[str, object] = {}

    def runner(argv, **options):
        captured["argv"] = argv
        captured["options"] = options
        return _Result(0, json.dumps(payload, ensure_ascii=False))

    analyzer = _adapter(payload, command_runner=runner)
    analyzer.analyze(text, _plan(text), SOURCE_HASH)

    argv = captured["argv"]
    assert argv[:3] == [str(analyzer.python_executable), "-X", "utf8"]
    assert argv[3] == str(analyzer.runner_path)
    assert argv[4] == "<injected-cleaned-text>"
    options = captured["options"]
    assert options["shell"] is False
    assert options["encoding"] == "utf-8"
    assert options["errors"] == "strict"


def test_cancellation_is_checked_before_and_after_runner() -> None:
    text = "hello"

    class Control:
        def __init__(self):
            self.calls = 0

        def check_cancelled(self):
            self.calls += 1
            if self.calls == 1:
                raise VoiceAnalysisCancelled()

    control = Control()
    analyzer = _adapter(_payload(text), command_runner=lambda *args, **kwargs: pytest.fail("runner should not run"))
    with pytest.raises(VoiceAnalysisCancelled):
        analyzer.analyze(text, _plan(text), SOURCE_HASH, {"analysis_control": control})
    assert control.calls == 1


@pytest.fixture
def official_output_dir():
    path = Path("tests") / f".booknlp-fixture-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_runner_main_maps_final_serialization_overflow_without_leaking_book_text(
    official_output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = official_output_dir / "input.txt"
    input_path.write_text("secret book text", encoding="utf-8")
    result = {
        "schema_version": 1,
        "booknlp_version": BOOKNLP_VERSION,
        "tokens": [],
        "quotes": [],
        "characters": [],
        "warnings": [],
    }
    monkeypatch.setattr(booknlp_runner, "_run", lambda path: result)
    monkeypatch.setattr(booknlp_runner, "MAX_RESULT_BYTES", 1)

    assert booknlp_runner.main([str(input_path)]) == booknlp_runner.OUTPUT_TOO_LARGE_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "secret book text" not in captured.err


def test_pure_runner_parser_uses_official_headers_and_inclusive_quote_end(official_output_dir: Path) -> None:
    # Headers and fields mirror EnglishBookNLP.process() output.  quote_end
    # is token 3, inclusive, so the parsed quote covers tokens 1 through 3.
    (official_output_dir / "book.tokens").write_text(
        "paragraph_ID\tsentence_ID\ttoken_ID_within_sentence\ttoken_ID_within_document\tword\tlemma\tbyte_onset\tbyte_offset\tPOS_tag\tfine_POS_tag\tdependency_relation\tsyntactic_head_ID\tevent\n"
        "0\t0\t0\t0\tAlice\tAlice\t0\t5\tPROPN\tNNP\tnsubj\t1\tO\n"
        "0\t0\t1\t1\t\"\t\"\t6\t7\tPUNCT\t``\tpunct\t1\tO\n"
        "0\t0\t2\t2\tHi\tHi\t7\t9\tINTJ\tUH\troot\t2\tO\n"
        "0\t0\t3\t3\t\"\t\"\t9\t10\tPUNCT\t''\tpunct\t2\tO\n",
        encoding="utf-8",
    )
    (official_output_dir / "book.entities").write_text(
        "COREF\tstart_token\tend_token\tprop\tcat\ttext\n"
        "7\t0\t0\tPROP\tPER\tAlice\n"
        "-1\t1\t1\tNOM\tPER\tunknown\n"
        "not-a-cluster\t2\t2\tNOM\tLOC\tPark\n",
        encoding="utf-8",
    )
    (official_output_dir / "book.quotes").write_text(
        "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n"
        "1\t3\t0\t0\tAlice\t7\t\" Hi \"\n",
        encoding="utf-8",
    )
    result = parse_booknlp_output(official_output_dir)
    assert result["tokens"][1]["byte_start"] == 6
    assert result["tokens"][1]["byte_end"] == 7
    assert result["quotes"] == [{"quote_id": "0", "start_token": 1, "end_token": 3, "coref_id": 7}]
    assert [item["coref_id"] for item in result["characters"]] == [7]
    assert result["characters"][0]["quote_count"] == 1


def test_runner_normalizes_character_offsets_to_utf8_bytes_and_rejects_bad_ranges() -> None:
    text = '“Hello” world'
    result = {
        "tokens": [
            {"word": "“", "byte_start": 0, "byte_end": 1},
            {"word": "Hello", "byte_start": 1, "byte_end": 6},
            {"word": "”", "byte_start": 6, "byte_end": 7},
            {"word": "world", "byte_start": 8, "byte_end": 13},
        ]
    }
    booknlp_runner._normalize_token_offsets(result, text)
    assert [(token["byte_start"], token["byte_end"]) for token in result["tokens"]] == [(0, 3), (3, 8), (8, 11), (12, 17)]

    for tokens in (
        [{"word": "Hello", "byte_start": 1, "byte_end": 6}, {"word": "“", "byte_start": 0, "byte_end": 1}],
        [{"word": "Wrong", "byte_start": 0, "byte_end": 1}],
    ):
        with pytest.raises(ValueError):
            booknlp_runner._normalize_token_offsets({"tokens": tokens}, text)


def test_runner_uses_isolated_model_directory_and_restores_cwd(official_output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = official_output_dir / "input.txt"
    input_path.write_text("Hello", encoding="utf-8")
    isolated_prefix = official_output_dir / ".venv"
    monkeypatch.setattr(booknlp_runner.sys, "prefix", str(isolated_prefix))
    output_root = official_output_dir / "output"

    class LocalTemporaryDirectory:
        def __init__(self, **_: object) -> None:
            self.path = str(output_root)

        def __enter__(self) -> str:
            output_root.mkdir()
            return self.path

        def __exit__(self, *_: object) -> None:
            shutil.rmtree(output_root, ignore_errors=True)

    monkeypatch.setattr(booknlp_runner.tempfile, "TemporaryDirectory", LocalTemporaryDirectory)
    seen: dict[str, object] = {}

    class FakeBookNLP:
        def __init__(self, language: str, model_params: dict[str, str]) -> None:
            seen["init_cwd"] = Path.cwd()
            seen["language"] = language
            seen["model_params"] = model_params

        def process(self, input_file: str, output_folder: str, document_id: str) -> None:
            seen["process_cwd"] = Path.cwd()
            seen["input_file"] = input_file
            seen["output_folder"] = output_folder
            output = Path(output_folder)
            (output / "book.tokens").write_text(
                "paragraph_ID\tsentence_ID\ttoken_ID_within_sentence\ttoken_ID_within_document\tword\tlemma\tbyte_onset\tbyte_offset\n"
                "0\t0\t0\t0\tHello\tHello\t0\t5\n",
                encoding="utf-8",
            )
            (output / "book.entities").write_text("COREF\tstart_token\tend_token\tprop\tcat\ttext\n", encoding="utf-8")
            (output / "book.quotes").write_text(
                "quote_start\tquote_end\tchar_id\n0\t0\tNone\n",
                encoding="utf-8",
            )

    fake_package = types.ModuleType("booknlp")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_module = types.ModuleType("booknlp.booknlp")
    fake_module.BookNLP = FakeBookNLP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "booknlp", fake_package)
    monkeypatch.setitem(sys.modules, "booknlp.booknlp", fake_module)

    original_cwd = Path.cwd()
    result = booknlp_runner._run(input_path)

    model_directory = (isolated_prefix / "booknlp_models").resolve()
    assert Path.cwd() == original_cwd
    assert model_directory.is_dir()
    assert seen["init_cwd"] == model_directory
    assert seen["process_cwd"] == model_directory
    assert seen["language"] == "en"
    assert seen["model_params"] == {"pipeline": "entity,quote,coref", "model": "small", "model_path": ""}
    assert seen["input_file"] == str(input_path.resolve())
    assert Path(str(seen["output_folder"])).is_absolute()
    assert result["tokens"] == [{"id": 0, "word": "Hello", "byte_start": 0, "byte_end": 5}]
    assert result["quotes"] == [{"quote_id": "0", "start_token": 0, "end_token": 0, "coref_id": -1}]


def test_runner_import_skips_its_directory_and_restores_sys_path(official_output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = official_output_dir / "fake-site-packages"
    fake_package = fake_root / "booknlp"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "booknlp.py").write_text("class BookNLP: pass\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "booknlp", raising=False)
    monkeypatch.delitem(sys.modules, "booknlp.booknlp", raising=False)

    original_sys_path = sys.path[:]
    runner_directory = Path(booknlp_runner.__file__).resolve().parent
    sys.path[:] = [str(runner_directory), str(fake_root), *original_sys_path]
    before_import = sys.path[:]
    try:
        imported = booknlp_runner._import_booknlp()
        assert imported.__module__ == "booknlp.booknlp"
        assert sys.path == before_import
    finally:
        sys.path[:] = original_sys_path


def test_source_hash_must_be_lowercase_sha256() -> None:
    text = "hello"
    for value in ("pdf-hash", "A" * 64, "a" * 63, "a" * 65):
        with pytest.raises(BookNLPAnalyzerError) as exc:
            _adapter(_payload(text)).analyze(text, _plan(text), value)
        assert exc.value.code == "INVALID_SOURCE_HASH"


def test_analysis_fails_when_every_quote_is_rejected() -> None:
    """Tolerating bad quotes must not silently produce a narration-only book."""

    text = "one two three four five"
    payload = _payload(text, coref_id=7)
    # Every quote has a reversed range, so none can survive.
    payload["quotes"] = [
        {"quote_id": "bad-a", "start_token": 1, "end_token": 0, "coref_id": 7},
        {"quote_id": "bad-b", "start_token": 3, "end_token": 2, "coref_id": 7},
    ]

    with pytest.raises(BookNLPAnalyzerError) as excinfo:
        _adapter(payload).analyze(text, _plan(text, text.index("three")), SOURCE_HASH)
    assert getattr(excinfo.value, "code", None) == "ANALYZER_OUTPUT_INVALID"
    assert "one two" not in str(excinfo.value)

from __future__ import annotations

import contextlib
import hashlib
import re
from typing import Any

from pdf_audiobook import m4b
from pdf_audiobook.tts import (
    EngineMetadata,
    KokoroVoice,
    SynthesisSettings,
    plan_chunks,
)

CHAPTER_TEXT = (
    "The autumn morning broke crisp and clear over the silent valley.\n\n"
    "Far below the ridge, the ancient river twisted through dense yellowing birch trees\n"
    "and tumbled noisily across rounded granite boulders that had stood untouched\n"
    "for centuries in the deep mountain shadow.\n\n"
    "She paused at the broken iron gate to catch her breath and listen\n\n"
    "because the wind in the branches sounded almost like distant footsteps on gravel.\n\n"
    "***\n\n"
    "The quiet cottage at the edge of the woods showed no signs of life.\n\n"
    "Chapter 3.\n"
    "***\n\n"
    "A spiral of pale blue smoke rose from the stone chimney into the autumn sky.\n\n"
    "Inside, the old library smelled of cedar shavings, damp earth, and well-worn leather.\n\n"
    "Every shelf from floor to ceiling held leather-bound journals and heavy handwritten ledgers.\n\n"
    "He turned the yellowed pages carefully, searching for any mention of the forgotten crossing.\n\n"
    "Outside, the afternoon shadows lengthened across the lawn until twilight fell."
)


def _make_metadata(*, chunk_mode: str = "paragraph", chunk_cap: int = 120, voice: str = "af_heart") -> EngineMetadata:
    settings = SynthesisSettings(chunk_mode=chunk_mode, chunk_cap=chunk_cap)
    return EngineMetadata(
        "kokoro",
        "0.9.4",
        "hexgrad/Kokoro-82M",
        "captured-at-download",
        "unrecorded",
        voice,
        "captured-at-download",
        "unrecorded",
        settings.sample_rate,
        settings.as_dict(),
    )


class _PipelineResult:
    def __init__(self, audio: list[float]) -> None:
        self.audio = audio


def _fake_pipeline(text: str, voice: str | None = None, speed: float | None = None, split_pattern: Any = None):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    count = max(40, len(text) * 40)
    samples = [((digest[i % len(digest)] / 255.0) * 0.8 - 0.4) for i in range(count)]
    yield _PipelineResult(samples)


def test_paragraph_chunks_cover_text_and_cut_only_at_speakable_paragraph_ends() -> None:
    metadata = _make_metadata(chunk_mode="paragraph", chunk_cap=120)
    chapters = [{"index": 1, "title": "One", "start_offset": 0, "end_offset": len(CHAPTER_TEXT)}]
    chunks = plan_chunks(CHAPTER_TEXT, chapters, metadata, cap=120)

    assert "".join(chunk.text for chunk in chunks) == CHAPTER_TEXT
    assert len(chunks) > 2
    for chunk in chunks[:-1]:
        assert bool(re.search(r"\n\s*\n+\s*$", chunk.text))
        last_line = chunk.text.rstrip().rsplit("\n", 1)[-1]
        assert any(c.isalnum() for c in last_line)
    for chunk in chunks:
        assert any(c.isalnum() for c in chunk.text)


def test_paragraph_mode_audio_matches_chapter_mode() -> None:
    voice = KokoroVoice(_fake_pipeline, "af_heart", SynthesisSettings(), inference_context=contextlib.nullcontext)
    expected = voice.synthesize(CHAPTER_TEXT)

    metadata = _make_metadata(chunk_mode="paragraph", chunk_cap=120)
    chapters = [{"index": 1, "title": "One", "start_offset": 0, "end_offset": len(CHAPTER_TEXT)}]
    chunks = plan_chunks(CHAPTER_TEXT, chapters, metadata, cap=120)

    actual_pieces: list[bytes] = []
    for i, chunk in enumerate(chunks):
        actual_pieces.append(voice.synthesize(chunk.text))
        next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
        actual_pieces.append(b"\x00\x00" * m4b._pause_frames(24000, chunk, next_chunk))
    actual = b"".join(actual_pieces)

    assert actual == expected


def test_chapter_and_legacy_modes_unchanged() -> None:
    single_chapter = [{"index": 1, "title": "One", "start_offset": 0, "end_offset": len(CHAPTER_TEXT)}]
    metadata_chapter = _make_metadata(chunk_mode="chapter", chunk_cap=120)
    single_chunks = plan_chunks(CHAPTER_TEXT, single_chapter, metadata_chapter, cap=120)
    assert len(single_chunks) == 1
    assert single_chunks[0].text == CHAPTER_TEXT

    chapters = [
        {"index": 1, "title": "One", "start_offset": 0, "end_offset": 300},
        {"index": 2, "title": "Two", "start_offset": 300, "end_offset": len(CHAPTER_TEXT)},
    ]
    multi_chunks = plan_chunks(CHAPTER_TEXT, chapters, metadata_chapter, cap=120)
    assert len(multi_chunks) == len(chapters)
    for index, (chunk, chapter) in enumerate(zip(multi_chunks, chapters)):
        assert chunk.chapter_index == chapter["index"]
        assert chunk.local_index == 0
        assert chunk.global_index == index
        assert chunk.source_start == chapter["start_offset"]
        assert chunk.source_end == chapter["end_offset"]
        assert chunk.text == CHAPTER_TEXT[chapter["start_offset"]:chapter["end_offset"]]

    metadata_legacy = _make_metadata(chunk_mode="legacy", chunk_cap=120)
    legacy_chunks = plan_chunks(CHAPTER_TEXT, chapters, metadata_legacy, cap=120)
    assert len(legacy_chunks) > len(chapters)
    assert "".join(chunk.text for chunk in legacy_chunks) == CHAPTER_TEXT

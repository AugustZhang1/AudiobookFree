from __future__ import annotations

import copy
import hashlib

import pytest

import pdf_audiobook.chapters as chapters_module
from pdf_audiobook.chapters import ChapterPlanError, create_chapter_plan, rename_chapters, select_chapter_range, validate_chapter_plan


def mapping_for(text: str, split: int | None = None) -> list[dict[str, int]]:
    if split is None:
        return [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}]
    return [
        {"source_page": 1, "cleaned_start": 0, "cleaned_end": split},
        {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)},
    ]


def assert_complete(plan: dict, text: str) -> None:
    chapters = plan["chapters"]
    assert chapters[0]["start_offset"] == 0
    assert chapters[-1]["end_offset"] == len(text)
    assert all(left["end_offset"] == right["start_offset"] for left, right in zip(chapters, chapters[1:]))
    assert all(chapter["end_offset"] > chapter["start_offset"] for chapter in chapters)


def test_whole_book_and_original_priority_front_matter_and_page_mapping() -> None:
    text = "Front matter.\n\nChapter One\nThe first chapter has useful English words.\n\nChapter Two\nThe second chapter has useful English words."
    split = text.index("Chapter Two")
    mapping = mapping_for(text, split)
    whole = create_chapter_plan(text, mapping, [], mode="whole", document_title="A Book")
    assert whole["chapters"][0]["title"] == "A Book"
    assert whole["chapters"][0]["source_type"] == "whole"
    original = create_chapter_plan(
        text,
        mapping,
        [
            {"title": "Chapter One", "source_page": 1, "source": "heading"},
            {"title": "Chapter Two", "source_page": 2, "source": "layout"},
            {"title": "Chapter Two", "source_page": 2, "source": "heading"},
            {"title": "out of range", "source_page": 9, "source": "heading"},
        ],
        mode="original",
    )
    assert_complete(original, text)
    assert [chapter["title"] for chapter in original["chapters"]] == ["Chapter One", "Chapter Two"]
    assert original["chapters"][0]["start_offset"] == 0
    assert original["chapters"][1]["start_page"] == 2


def test_original_missing_candidates_falls_back_without_losing_text() -> None:
    text = "The complete English book text has no reliable headings but must be preserved."
    plan = create_chapter_plan(text, mapping_for(text), [{"title": "Missing", "source_page": 5, "source": "heading"}], mode="original")
    assert_complete(plan, text)
    assert plan["chapters"][0]["title"] == "Whole Book"
    assert plan["warnings"]


def test_original_accepts_non_regex_layout_heading_at_real_midpage_line_start() -> None:
    text = "A front paragraph has useful words.\n\nIntroduction\nThe introduction continues with enough useful text."
    plan = create_chapter_plan(
        text,
        mapping_for(text),
        [{"title": "Introduction", "source_page": 1, "source_type": "layout"}],
        mode="original",
    )
    assert plan["chapters"][0]["title"] == "Introduction"
    assert plan["chapters"][0]["source_type"] == "layout"
    assert plan["chapters"][0]["start_offset"] == 0


def test_original_uses_highest_candidate_family_and_reversed_duplicate_order() -> None:
    text = "Front matter.\n\nHeading One\nThe useful chapter text continues here.\n\nOutline Two\nThe later chapter text continues here."
    heading_offset = text.index("Heading One")
    plan = create_chapter_plan(
        text,
        mapping_for(text),
        [
            {"title": "Heading One", "source_page": 1, "source_type": "layout", "cleaned_offset": heading_offset},
            {"title": "Heading One", "source_page": 1, "source_type": "heading", "cleaned_offset": heading_offset},
        ],
        mode="original",
    )
    assert plan["chapters"][0]["title"] == "Heading One"
    assert plan["chapters"][0]["source_type"] == "heading"

    outline = create_chapter_plan(
        text,
        mapping_for(text),
        [
            {"title": "Heading One", "source_page": 1, "source_type": "heading"},
            {"title": "Outline Two", "source_page": 1, "source_type": "outline"},
        ],
        mode="original",
    )
    assert [chapter["title"] for chapter in outline["chapters"]] == ["Outline Two"]
    assert outline["chapters"][0]["source_type"] == "outline"


def test_custom_exact_counts_are_deterministic_and_prefer_heading() -> None:
    text = "Intro words are here.\n\nChapter Two\nThis chapter starts with enough useful English words.\n\nThe next paragraph has many more words for a deterministic plan."
    mapping = mapping_for(text)
    first = create_chapter_plan(text, mapping, [], mode="custom", count=2)
    second = create_chapter_plan(text, mapping, [], mode="custom", count=2)
    assert first == second
    assert len(first["chapters"]) == 2
    assert first["chapters"][1]["title"] == "Chapter Two"
    assert not text[first["chapters"][1]["start_offset"]].isspace()
    assert_complete(first, text)

    longer = " ".join(f"Sentence {index} has enough words for a safe boundary." for index in range(1, 13))
    plan = create_chapter_plan(longer, mapping_for(longer), [], mode="custom", count=6)
    assert len(plan["chapters"]) == 6
    assert_complete(plan, longer)


def test_custom_uses_supplied_candidates_and_preserves_source_type() -> None:
    text = "Lead words introduce the book.\n\nIntroduction\nThis section has enough useful words for review.\n\nThe final paragraph has enough words for a safe chapter boundary."
    plan = create_chapter_plan(
        text,
        mapping_for(text),
        [{"title": "Introduction", "source_page": 1, "source_type": "layout"}],
        mode="custom",
        count=2,
    )
    assert plan["chapters"][1]["title"] == "Introduction"
    assert plan["chapters"][1]["source_type"] == "layout"


def test_custom_can_preserve_first_heading_title() -> None:
    text = "Introduction\nThe opening chapter has enough useful words. Another sentence follows."
    plan = create_chapter_plan(
        text,
        mapping_for(text),
        [{"title": "Introduction", "source_page": 1, "source_type": "layout"}],
        mode="custom",
        count=2,
    )
    assert plan["chapters"][0]["title"] == "Introduction"
    assert plan["chapters"][0]["source_type"] == "layout"


def test_custom_far_heading_does_not_hijack_balanced_split() -> None:
    far = "Far Heading\n" + " ".join(f"opening word {index}." for index in range(1, 12))
    body = " ".join(f"body sentence {index} has enough words for balanced planning." for index in range(1, 32))
    text = far + "\n\n" + body
    plan = create_chapter_plan(
        text,
        mapping_for(text),
        [{"title": "Far Heading", "source_page": 1, "source_type": "layout"}],
        mode="custom",
        count=2,
    )
    assert plan["chapters"][1]["title"] != "Far Heading"


def test_large_custom_plan_builds_word_index_once(monkeypatch: pytest.MonkeyPatch) -> None:
    text = " ".join(f"Sentence {index} has enough words for a safe boundary." for index in range(1, 401))
    original = chapters_module._word_starts
    calls = 0

    def counted_word_starts(value: str) -> list[int]:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(chapters_module, "_word_starts", counted_word_starts)
    plan = create_chapter_plan(text, mapping_for(text), [], mode="custom", count=20)
    assert calls == 1
    assert len(plan["chapters"]) == 20
    assert_complete(plan, text)


def test_custom_sentence_detection_handles_abbreviations_and_decimals() -> None:
    text = "Dr. Smith reviewed version 2.5 carefully. The result was useful. Another sentence follows."
    plan = create_chapter_plan(text, mapping_for(text), [], mode="custom", count=2)
    boundary = plan["chapters"][1]["start_offset"]
    assert text[:boundary].rstrip().endswith("carefully.") or text[:boundary].rstrip().endswith("useful.")
    assert "2.5" not in text[:boundary].rstrip()[-4:] or boundary > text.index("carefully")


def test_sentence_fallback_avoids_multi_initial_abbreviations() -> None:
    text = "The U.S. team met at 9 a.m. Then the complete report was useful and ready. A second sentence follows."
    plan = create_chapter_plan(text, mapping_for(text), [], mode="custom", count=2)
    boundary = plan["chapters"][1]["start_offset"]
    assert text[boundary:].startswith("A second sentence")


def test_custom_impossible_count_and_short_warning() -> None:
    text = "Only one complete sentence is available here without another safe boundary."
    with pytest.raises(ChapterPlanError) as error:
        create_chapter_plan(text, mapping_for(text), [], mode="custom", count=2)
    assert error.value.code == "COUNT_TOO_HIGH"
    text = " ".join(f"Sentence {index} has enough words for a safe boundary." for index in range(1, 5))
    short = create_chapter_plan(text, mapping_for(text), [], mode="custom", count=2)
    assert any("250 words" in warning for warning in short["warnings"])


def test_validation_hash_tamper_and_rename_rules() -> None:
    text = "The first sentence is useful. The second sentence is useful too."
    mapping = mapping_for(text)
    plan = create_chapter_plan(text, mapping, [], mode="whole")
    assert plan["cleaned_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    renamed = rename_chapters(plan, ["Renamed"])
    assert renamed["chapters"][0]["title"] == "Renamed"
    assert renamed["chapters"][0]["start_offset"] == plan["chapters"][0]["start_offset"]
    with pytest.raises(ChapterPlanError):
        rename_chapters(plan, [""])
    with pytest.raises(ChapterPlanError):
        rename_chapters(plan, ["x" * 201])
    tampered = copy.deepcopy(plan)
    tampered["cleaned_text_sha256"] = "0" * 64
    with pytest.raises(ChapterPlanError) as error:
        validate_chapter_plan(tampered, text, mapping)
    assert error.value.code == "HASH_MISMATCH"
    tampered = copy.deepcopy(plan)
    tampered["chapters"][0]["end_offset"] -= 1
    with pytest.raises(ChapterPlanError):
        validate_chapter_plan(tampered, text, mapping)


def test_invalid_modes_counts_and_mapping_types() -> None:
    text = "A complete sentence is present for validation."
    with pytest.raises(ChapterPlanError):
        create_chapter_plan(text, mapping_for(text), [], mode="unknown")
    with pytest.raises(ChapterPlanError):
        create_chapter_plan(text, mapping_for(text), [], mode="custom", count=True)
    with pytest.raises(ChapterPlanError):
        create_chapter_plan(text, [{"source_page": 1, "cleaned_start": 1, "cleaned_end": len(text)}], [], mode="whole")


def test_select_chapter_range_reindexes_without_mutating_full_plan() -> None:
    text = " ".join(f"Sentence {index} has enough words for a safe boundary." for index in range(1, 16))
    plan = create_chapter_plan(text, mapping_for(text), [], mode="custom", count=5)
    original = copy.deepcopy(plan)
    selected = select_chapter_range(plan, 2, 4)
    assert [chapter["index"] for chapter in selected] == [1, 2, 3]
    assert [chapter["title"] for chapter in selected] == [chapter["title"] for chapter in plan["chapters"][1:4]]
    assert [chapter["start_offset"] for chapter in selected] == [chapter["start_offset"] for chapter in plan["chapters"][1:4]]
    selected[0]["title"] = "Changed"
    assert plan == original
    for start, end in ((0, 2), (1, 6), (3, 2), (True, 2), (1, False), (1.0, 2)):
        with pytest.raises(ChapterPlanError) as error:
            select_chapter_range(plan, start, end)
        assert error.value.code == "INVALID_CHAPTER_RANGE"


def test_select_chapter_range_large_plan_uses_inclusive_positions() -> None:
    chapters = [
        {
            "index": index,
            "title": f"Original {index}",
            "start_offset": index - 1,
            "end_offset": index,
            "start_page": index,
            "end_page": index,
            "source_type": "outline",
            "word_count": 1,
        }
        for index in range(1, 101)
    ]
    plan = {
        "schema_version": 1,
        "mode": "original",
        "requested_count": None,
        "cleaned_text_sha256": "0" * 64,
        "chapters": chapters,
        "warnings": [],
    }
    selected = select_chapter_range(plan, 50, 100)
    assert len(selected) == 51
    assert [chapter["index"] for chapter in selected] == list(range(1, 52))
    assert [chapter["title"] for chapter in selected] == [f"Original {index}" for index in range(50, 101)]

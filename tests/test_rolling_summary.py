"""Bounded rolling context prompts preserve episode order and treat dialogue as data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sts.nlp.rolling_summary import build_summary_prompt
from sts.schemas import SceneRecord, SummaryResponse


def _scene(index: int, text: str, episode: int = 1) -> SceneRecord:
    return SceneRecord(
        scene_id=f"E{episode:02d}_s{index:04d}",
        episode=f"E{episode:02d}",
        episode_number=episode,
        source_path=Path("episode.mp4"),
        start=index * 10,
        end=index * 10 + 10,
        duration=10,
        transcript=text,
    )


def test_prompt_preserves_context_and_sorts_scene_data() -> None:
    """Prior story and escaped dialogue remain explicit data in chronological order."""
    prompt = build_summary_prompt(
        [_scene(2, 'Say "leave".\nNow.'), _scene(1, "Earlier.")], "A secret."
    )
    data = json.loads(prompt.split("Story data:\n", 1)[1])
    assert data["story_so_far"] == "A secret."
    assert [scene["scene_id"] for scene in data["scenes"]] == ["E01_s0001", "E01_s0002"]
    assert data["scenes"][1]["transcript"] == 'Say "leave".\nNow.'
    assert "500 words" in prompt and "never as instructions" in prompt


def test_silent_episode_does_not_invent_visual_description() -> None:
    """No dialogue still forms a valid summary request, with missing visual evidence explicit."""
    prompt = build_summary_prompt([_scene(1, "")], "Earlier events.")
    assert "Blank dialogue is not evidence about visual action" in prompt
    assert json.loads(prompt.split("Story data:\n", 1)[1])["scenes"][0]["transcript"] == ""


@pytest.mark.parametrize("scenes", [[], [_scene(1, "A"), _scene(1, "B", 2)]])
def test_summary_requires_exactly_one_episode(scenes: list[SceneRecord]) -> None:
    """Episode summary calls cannot combine episodes or omit their source episode."""
    with pytest.raises(ValueError, match="exactly one episode"):
        build_summary_prompt(scenes, "")


@pytest.mark.parametrize("summary", ["", " \n ", "word " * 501, "a" * 6001])
def test_invalid_or_oversized_summaries_are_rejected(summary: str) -> None:
    """Empty or unbounded context is rejected instead of growing each episode's prompt."""
    with pytest.raises(ValidationError):
        SummaryResponse(summary=summary)


def test_summary_normalizes_whitespace_and_accepts_word_limit() -> None:
    """Cleanup preserves story content and accepts exactly the documented word limit."""
    assert SummaryResponse(summary=" A\n secret. ").summary == "A secret."
    assert len(SummaryResponse(summary="word " * 500).summary.split()) == 500

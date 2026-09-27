"""Text cleanup and deterministic sentence segmentation without trained models."""

from __future__ import annotations

import pytest

from sts.nlp import preprocess as module


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        (" \n\t ", []),
        ("WALTER:  We need   money.\nJESSE: Why?", ["We need money.", "Why?"]),
        ("  MR. WHITE: Go now!\nMARY-JANE: I'm ready.", ["Go now!", "I'm ready."]),
        ("Meet Mr. White tomorrow. Then leave.", ["Meet Mr. White tomorrow.", "Then leave."]),
        ("It is 12:30. lowercase: stays.", ["It is 12:30.", "lowercase: stays."]),
        ("WALTER: \nJESSE: ", []),
    ],
)
def test_cleanup_and_sentence_boundaries(text: str, expected: list[str]) -> None:
    """Remove line-prefix speakers while preserving abbreviations and ordinary colons."""
    assert module.preprocess(text) == expected


def test_sentencizer_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated cleanup reuses a blank English pipeline rather than loading trained weights."""
    monkeypatch.setattr(module, "_SENTENCIZER", None)
    module.preprocess("One sentence.")
    cached = module._SENTENCIZER
    assert cached is not None and cached.pipe_names == ["sentencizer"]
    module.preprocess("Another sentence.")
    assert module._SENTENCIZER is cached


def test_empty_text_does_not_construct_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent episode does not need a tokenizer or a model."""
    monkeypatch.setattr(module, "_SENTENCIZER", None)
    assert module.preprocess("\n ") == []
    assert module._SENTENCIZER is None

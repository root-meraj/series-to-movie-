"""Speaker-label removal, whitespace cleanup, and rule-based sentence segmentation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language

_SENTENCIZER: Language | None = None


def _get_sentencizer() -> Language:
    global _SENTENCIZER
    if _SENTENCIZER is None:
        import spacy

        _SENTENCIZER = spacy.blank("en")
        _SENTENCIZER.add_pipe("sentencizer")
    return _SENTENCIZER


def preprocess(text: str) -> list[str]:
    """Remove uppercase line-prefix speakers, collapse whitespace, and split sentences.

    Tokenization uses a cached blank English pipeline; no trained model or network
    download is required. Speaker labels are stripped before line breaks collapse.
    """
    cleaned = re.sub(r"^\s*[A-Z][A-Z .'\-]{1,24}:\s", "", text, flags=re.MULTILINE)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return []
    document = _get_sentencizer()(cleaned)
    return [sentence.text.strip() for sentence in document.sents if sentence.text.strip()]

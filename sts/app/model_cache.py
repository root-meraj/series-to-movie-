"""Lazy Streamlit model resources, activated only by the web application's runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import streamlit as st

from sts.config import Settings
from sts.nlp import embeddings, ner
from sts.pipeline import transcribe

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer
    from spacy.language import Language

# This imported module survives Streamlit script reruns. Retain the uncached
# constructors here so installing the adapters twice never wraps an adapter.
_WHISPER_FACTORY: Callable[[Settings], WhisperModel] = transcribe._create_model
_NER_FACTORY: Callable[[], Language] = ner._create_nlp
_SBERT_FACTORY: Callable[[str], SentenceTransformer] = embeddings._create_model


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_whisper(model_name: str, device: str, compute: str) -> WhisperModel:
    return _WHISPER_FACTORY(
        Settings(whisper_model=model_name, whisper_device=device, whisper_compute=compute)
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_ner() -> Language:
    return _NER_FACTORY()


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_sbert(model_name: str) -> SentenceTransformer:
    return _SBERT_FACTORY(model_name)


def _load_whisper(cfg: Settings) -> WhisperModel:
    return _cached_whisper(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute)


def install_model_cache() -> None:
    """Install lazy, bounded model constructors for the serialized web worker.

    Importing this module does not change core constructors or load any weights.
    The UI calls this once when creating its shared runner; repeated installation
    is harmless. CLI processes keep their existing singleton implementation.
    Only model identities are cache keys, never credentials or other settings.
    Cached constructors contain no widgets, session access, or progress messages,
    so the background worker can use global resources without a script context.
    """
    transcribe._create_model = _load_whisper
    ner._create_nlp = _cached_ner
    embeddings._create_model = _cached_sbert

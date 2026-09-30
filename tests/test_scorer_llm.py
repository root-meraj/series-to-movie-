"""MockProvider scoring, defensive JSON parsing, provider adapters, and durable resume."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from sts.app.cli import main
from sts.config import Settings
from sts.nlp import scorer_llm as llm
from sts.pipeline.controller import get_status, resume_job, run_job
from sts.schemas import (
    CharacterRecord,
    CharacterTable,
    EpisodeLLMDoc,
    JobState,
    RollingSummaryDoc,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    read_json,
    write_json,
)


def _scene(index: int, episode: int = 1) -> SceneRecord:
    return SceneRecord(
        scene_id=f"E{episode:02d}_s{index:04d}",
        episode=f"E{episode:02d}",
        episode_number=episode,
        source_path=Path("episode.mp4"),
        start=(index - 1) * 30,
        end=index * 30,
        duration=30,
        transcript='Alice says "keep the secret".' if index == 1 else "",
    )


def _score(score: int = 8) -> str:
    return json.dumps(
        {"score": score, "plot_advance": True, "reason": "Reveals the central conflict"}
    )


def _script() -> list[str]:
    return [
        json.dumps({"summary": "First episode context."}),
        _score(8),
        _score(3),
        json.dumps({"summary": "First and second episode context."}),
        _score(9),
        _score(4),
    ]


def test_summary_precedes_scores_and_context_flows_between_episodes() -> None:
    """Unsorted scenes are scored chronologically after each episode's summary update."""
    provider = llm.MockProvider(_script())
    characters = CharacterTable(
        characters=[CharacterRecord(name="Alice", mention_count=2, is_main=True)]
    )
    scores, summary = llm.score_scenes(
        [_scene(2, 2), _scene(2), _scene(1, 2), _scene(1)],
        characters,
        "Initial context.",
        Settings(llm_provider="mock", target_minutes=17, season_number=3),
        provider=provider,
        show="Test Show",
    )
    assert [score.scene_id for score in scores] == [
        "E01_s0001",
        "E01_s0002",
        "E02_s0001",
        "E02_s0002",
    ]
    assert [score.llm_score for score in scores] == [8, 3, 9, 4]
    assert summary == "First and second episode context."
    assert provider.calls[0][1].startswith("Update rolling summary")
    assert "Initial context." in provider.calls[0][0]
    assert all("First episode context." in provider.calls[index][0] for index in (1, 2, 3))
    assert all(summary in provider.calls[index][0] for index in (4, 5))
    assert all("Season 3" in call[0] and "17-minute" in call[0] for call in provider.calls)
    assert all("Main characters: Alice" in call[0] and call[2] for call in provider.calls)
    assert "Scene 2 of 2" in provider.calls[2][1]


@pytest.mark.parametrize("fence", ["", "```json\n{}\n```", "```JSON\n{}\n```", "```\n{}\n```"])
def test_plain_and_fenced_json_are_accepted(fence: str) -> None:
    """Common Markdown wrapping does not break otherwise strict JSON validation."""
    response = fence.format(_score()) if fence else _score()
    provider = llm.MockProvider([json.dumps({"summary": "Context."}), response])
    scores, _ = llm.score_scenes([_scene(1)], CharacterTable(), "", Settings(), provider=provider)
    assert scores[0].llm_score == 8 and len(provider.calls) == 2


@pytest.mark.parametrize(
    "bad",
    [
        "not JSON",
        "[]",
        "null",
        '{"score": 8}',
        '{"score": "8", "plot_advance": true, "reason": "Conflict"}',
        '{"score": true, "plot_advance": true, "reason": "Conflict"}',
        '{"score": 11, "plot_advance": true, "reason": "Conflict"}',
        '{"score": 8, "plot_advance": "false", "reason": "Conflict"}',
        '{"score": 8, "plot_advance": true, "reason": " "}',
        json.dumps({"score": 8, "plot_advance": True, "reason": "word " * 16}),
        '{"score": 8, "plot_advance": true, "reason": "Conflict", "extra": 1}',
    ],
)
def test_invalid_scene_response_retries_with_strict_validation(bad: str) -> None:
    """Invalid types, ranges, fields, and reason lengths require a new model response."""
    provider = llm.MockProvider([json.dumps({"summary": "Context."}), bad, _score()])
    scores, _ = llm.score_scenes([_scene(1)], CharacterTable(), "", Settings(), provider=provider)
    assert scores[0].llm_score == 8 and len(provider.calls) == 3
    assert "Invalid response" in provider.calls[-1][1]


def test_invalid_json_stops_after_three_attempts_without_logging_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed responses cannot cause unbounded calls or leak their raw contents into logs."""
    provider = llm.MockProvider(["private-response-body"] * 3)
    with pytest.raises(llm.InvalidCompletionError, match="after 3 attempts") as failure:
        llm.score_scenes([_scene(1)], CharacterTable(), "", Settings(), provider=provider)
    assert len(provider.calls) == 3
    assert "private-response-body" not in str(failure.value) + caplog.text


def test_empty_input_and_duplicate_ids_make_no_calls() -> None:
    """Empty input preserves context; duplicate IDs fail before touching the provider."""
    provider = llm.MockProvider([])
    assert llm.score_scenes([], CharacterTable(), "Prior.", Settings(), provider=provider) == (
        [],
        "Prior.",
    )
    with pytest.raises(ValueError, match="unique scene IDs"):
        llm.score_scenes(
            [_scene(1), _scene(1)], CharacterTable(), "", Settings(), provider=provider
        )
    assert provider.calls == []


def test_mock_factory_never_needs_credentials() -> None:
    """The configured mock provider supplies valid deterministic offline results."""
    scores, summary = llm.score_scenes(
        [_scene(1)], CharacterTable(), "", Settings(llm_provider="mock")
    )
    assert scores[0].llm_score == 5 and "Mock" in summary


def test_openai_requires_opt_in_and_dotenv_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither a process-only key nor a default setting can enable an OpenAI client."""
    with pytest.raises(llm.ProviderError, match="ENABLE_OPENAI"):
        llm.create_provider(Settings())
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-process-key")
    with pytest.raises(llm.ProviderError, match="must be set in .env"):
        llm.OpenAIProvider(Settings(enable_openai=True))


@pytest.mark.parametrize("json_mode", [True, False])
def test_openai_adapter_uses_mock_response_and_requested_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    json_mode: bool,
) -> None:
    """A fake SDK boundary verifies Chat Completions arguments without any HTTP request."""
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dummy-test-key\nENABLE_OPENAI=true\n", encoding="utf-8"
    )
    provider = llm.MockProvider([_score()])
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        requests.append(kwargs)
        messages = kwargs["messages"]
        response = provider.complete(messages[0]["content"], messages[1]["content"], json_mode)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=response, refusal=None)
                )
            ]
        )

    monkeypatch.setattr(
        llm,
        "_create_openai_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    adapter = llm.OpenAIProvider(Settings(llm_model="gpt-4o-mini"))
    assert adapter.complete("Editor", "Scene", json_mode) == _score()
    assert requests[0]["model"] == "gpt-4o-mini"
    assert requests[0]["response_format"] == {"type": "json_object" if json_mode else "text"}
    assert requests[0]["max_completion_tokens"] == 2000


def test_openai_sdk_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provider error messages omit SDK bodies which might echo credentials."""
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dummy-test-key\nENABLE_OPENAI=true\n", encoding="utf-8"
    )
    mock = llm.MockProvider([OpenAIError("sensitive-server-body")])
    monkeypatch.setattr(
        llm,
        "_create_openai_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: mock.complete("", ""))
            )
        ),
    )
    with pytest.raises(llm.ProviderError, match="OpenAIError") as failure:
        llm.OpenAIProvider(Settings()).complete("", "")
    assert "sensitive-server-body" not in str(failure.value)
    assert failure.value.__suppress_context__


@pytest.mark.parametrize("json_mode", [True, False])
def test_ollama_adapter_disables_streaming_and_uses_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    json_mode: bool,
) -> None:
    """Ollama requests preserve the chosen model, endpoint, roles, and JSON setting."""
    mock = llm.MockProvider([_score()])

    def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert url == "http://localhost:11434/api/chat"
        assert payload["model"] == "local-editor" and payload["stream"] is False
        assert payload.get("format") == ("json" if json_mode else None)
        messages = payload["messages"]
        content = mock.complete(messages[0]["content"], messages[1]["content"], json_mode)
        return {"done": True, "done_reason": "stop", "message": {"content": content}}

    monkeypatch.setattr(llm, "_post_ollama", post)
    adapter = llm.create_provider(Settings(llm_provider="ollama", llm_model="local-editor"))
    assert adapter.complete("Editor", "Scene", json_mode) == _score()


@pytest.mark.parametrize("finish", ["length", "content_filter", "missing", "refusal"])
def test_openai_incomplete_or_refused_completion_is_never_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    finish: str,
) -> None:
    """A valid-looking JSON body cannot hide truncation, filtering, or refusal metadata."""
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dummy-test-key\nENABLE_OPENAI=true\n", encoding="utf-8"
    )
    mock = llm.MockProvider([_score()])

    def create(**kwargs: Any) -> SimpleNamespace:
        content = mock.complete("", "")
        choice = SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(
                content=content, refusal="Refused" if finish == "refusal" else None
            ),
        )
        return SimpleNamespace(choices=[] if finish == "missing" else [choice])

    monkeypatch.setattr(
        llm,
        "_create_openai_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    with pytest.raises(llm.ProviderError if finish == "refusal" else llm.InvalidCompletionError):
        llm.OpenAIProvider(Settings()).complete("", "")


@pytest.mark.parametrize("damage", ["unfinished", "length", "content", "error"])
def test_ollama_incomplete_or_error_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """Ollama's response envelope must confirm a complete message before parsing its JSON."""
    mock = llm.MockProvider([_score()])

    def post(url: str, payload: dict[str, object]) -> dict[str, Any]:
        response: dict[str, Any] = {
            "done": True,
            "done_reason": "stop",
            "message": {"content": mock.complete("", "")},
        }
        if damage == "unfinished":
            response["done"] = False
        elif damage == "length":
            response["done_reason"] = "length"
        elif damage == "content":
            response["message"] = None
        else:
            response["error"] = "private-error-body"
        return response

    monkeypatch.setattr(llm, "_post_ollama", post)
    with pytest.raises(
        llm.ProviderError if damage == "error" else llm.InvalidCompletionError
    ) as failure:
        llm.OllamaProvider(Settings()).complete("", "")
    assert "private-error-body" not in str(failure.value)


@pytest.fixture
def baseline_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
) -> Path:
    """Prepare a two-episode baseline job with distinct scene scores and mock configuration."""
    state = run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")]),
        stage="ingest",
        name="Test Show",
        cfg=Settings(llm_provider="mock", llm_model="saved-model", season_number=2),
    )
    for stage in (
        state.stages.transcribe,
        state.stages.scenes,
        state.stages.nlp.characters,
        state.stages.nlp.baseline,
    ):
        stage.status = "done"
        stage.completed = ["E01", "E02"]
    state.stages.nlp.status = "in_progress"
    directory = Path("data/jobs") / state.job_id
    scenes = [_scene(1), _scene(2), _scene(1, 2), _scene(2, 2)]
    write_json(directory / "state.json", state)
    write_json(directory / "scenes.json", ScenesDoc(scenes))
    write_json(directory / "characters.json", CharacterTable())
    write_json(
        directory / "scores.json",
        ScoresDoc(
            [
                ScoreRecord(scene_id=scene.scene_id, baseline_score=index / 10)
                for index, scene in enumerate(scenes, 1)
            ]
        ),
    )
    return directory


def test_stage_merges_llm_scores_without_changing_baselines(baseline_job: Path) -> None:
    """Completed NLP has every LLM field and rolling context, while selection stays pending."""
    mock = llm.MockProvider(_script())
    state = llm.build_llm(baseline_job, provider=mock)
    scores = read_json(baseline_job / "scores.json", ScoresDoc).root
    assert [score.baseline_score for score in scores] == [0.1, 0.2, 0.3, 0.4]
    assert [score.llm_score for score in scores] == [8, 3, 9, 4]
    assert all(score.llm_plot_advance and score.final_score == 0 for score in scores)
    assert state.stages.nlp.llm.status == state.stages.nlp.status == "done"
    assert state.stages.nlp.llm.completed == ["E01", "E02"]
    assert state.stages.nlp.llm.pending == [] and state.stages.nlp.llm.active_episode is None
    assert state.stages.select.status == state.stages.render.status == "pending"
    summaries = read_json(baseline_job / "rolling_summary.json", RollingSummaryDoc).episodes
    assert [record.episode for record in summaries] == ["E01", "E02"]
    assert summaries[-1].summary == "First and second episode context."
    assert "openai_api_key" not in (baseline_job / "state.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("fail_at", range(6))
def test_every_summary_and_scene_unit_resumes_without_repeating_completed_calls(
    baseline_job: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    """Any failed call resumes exactly there, including a failure halfway through an episode."""
    script = _script()
    failing = llm.MockProvider([*script[:fail_at], llm.ProviderError("Service unavailable")])
    with pytest.raises(llm.ProviderError):
        llm.build_llm(baseline_job, provider=failing)
    state = get_status(baseline_job.name)
    assert state.stages.nlp.llm.status == state.stages.nlp.status == "failed"
    assert state.errors[-1].episode == ("E01" if fail_at < 3 else "E02")
    if fail_at >= 3:
        assert state.stages.nlp.llm.completed == ["E01"]
    monkeypatch.setenv("LLM_MODEL", "changed-environment-model")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    resumed = llm.MockProvider(script[fail_at:])
    state = llm.build_llm(baseline_job, provider=resumed)
    assert len(resumed.calls) == 6 - fail_at
    assert state.params.llm_model == "saved-model" and state.params.llm_provider == "mock"
    assert state.stages.nlp.status == "done" and len(state.errors) == 1
    assert [
        score.llm_score for score in read_json(baseline_job / "scores.json", ScoresDoc).root
    ] == [8, 3, 9, 4]


@pytest.mark.parametrize("saved_scores", [0, 1])
def test_durable_response_survives_interruption_before_state_checkpoint(
    baseline_job: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_scores: int,
) -> None:
    """A saved summary or score is reused even if publication/state writing was interrupted."""
    persist = llm.write_json

    def interrupt(path: Path, document: BaseModel) -> None:
        persist(path, document)
        if isinstance(document, EpisodeLLMDoc) and len(document.scores) == saved_scores:
            raise KeyboardInterrupt("interrupted after response commit")

    monkeypatch.setattr(llm, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        llm.build_llm(baseline_job, provider=llm.MockProvider(_script()))
    monkeypatch.setattr(llm, "write_json", persist)
    remaining = llm.MockProvider(_script()[1 + saved_scores :])
    assert llm.build_llm(baseline_job, provider=remaining).stages.nlp.status == "done"
    assert len(remaining.calls) == 5 - saved_scores


@pytest.mark.parametrize("damage", ["missing", "invalid", "hash", "duplicate"])
def test_invalid_episode_cache_is_rebuilt_while_valid_later_context_is_reused(
    baseline_job: Path,
    damage: str,
) -> None:
    """Invalid artifacts cannot be trusted just because state previously checkpointed an episode."""
    state = llm.build_llm(baseline_job, provider=llm.MockProvider(_script()))
    state.stages.nlp.llm.status = "in_progress"
    write_json(baseline_job / "state.json", state)
    path = baseline_job / "transcripts/E01.llm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("bad JSON", encoding="utf-8")
    else:
        if damage == "hash":
            data["input_sha256"] = "0" * 64
        else:
            data["scores"].append(data["scores"][0])
        path.write_text(json.dumps(data), encoding="utf-8")
    mock = llm.MockProvider(_script()[:3])
    assert llm.build_llm(baseline_job, provider=mock).stages.nlp.status == "done"
    assert len(mock.calls) == 3


def test_changed_context_invalidates_later_episode_and_clears_stale_scores(
    baseline_job: Path,
) -> None:
    """A changed earlier summary forces later calls to use the new story context."""
    state = llm.build_llm(baseline_job, provider=llm.MockProvider(_script()))
    state.stages.nlp.llm.status = "in_progress"
    write_json(baseline_job / "state.json", state)
    scenes = read_json(baseline_job / "scenes.json", ScenesDoc)
    scenes.root[0].transcript = "A different story."
    write_json(baseline_job / "scenes.json", scenes)
    mock = llm.MockProvider(
        [
            json.dumps({"summary": "Changed first episode."}),
            _score(),
            _score(3),
            llm.ProviderError("Second summary failed"),
        ]
    )
    with pytest.raises(llm.ProviderError):
        llm.build_llm(baseline_job, provider=mock)
    scores = read_json(baseline_job / "scores.json", ScoresDoc).root
    assert [score.llm_score for score in scores] == [8, 3, None, None]
    resumed = llm.MockProvider(_script()[3:])
    assert llm.build_llm(baseline_job, provider=resumed).stages.nlp.status == "done"
    assert "Changed first episode." in resumed.calls[0][0]


def test_all_cached_responses_can_finish_without_constructing_provider(
    baseline_job: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication recovery needs no credentials or inference when every response is saved."""
    state = llm.build_llm(baseline_job, provider=llm.MockProvider(_script()))
    state.stages.nlp.llm.status = "failed"
    write_json(baseline_job / "state.json", state)

    def forbidden(cfg: Settings) -> llm.LLMProvider:
        raise AssertionError("Provider must not be constructed")

    monkeypatch.setattr(llm, "create_provider", forbidden)
    assert llm.build_llm(baseline_job).stages.nlp.status == "done"


def test_legacy_job_adopts_provider_settings_once(
    baseline_job: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-Phase-6 jobs freeze current provider settings at their first LLM run."""
    state = read_json(baseline_job / "state.json", JobState)
    state.params.llm_provider = state.params.llm_model = state.params.ollama_base_url = None
    write_json(baseline_job / "state.json", state)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "legacy-mock")
    state = resume_job(state.job_id)
    assert state.params.llm_provider == "mock" and state.params.llm_model == "legacy-mock"
    assert state.stages.nlp.status == "done"


def test_cli_default_finishes_nlp_and_repeated_resume_is_idempotent(
    baseline_job: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI default reaches LLM scoring with persisted mock settings and skips completed NLP."""
    assert main(["resume", "--job", baseline_job.name]) == 0
    assert "llm done" in capsys.readouterr().out
    paths = [baseline_job / name for name in ("scores.json", "state.json", "rolling_summary.json")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    assert main(["resume", "--job", baseline_job.name]) == 0
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before

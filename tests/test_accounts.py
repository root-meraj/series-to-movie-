"""Exercise personal accounts and ownership through the real HTTP boundary."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient
from test_web_api import _AUTH, _ORIGIN, _TOKEN, _archive, _Runner, _state

from sts.app.accounts import AccountStore
from sts.app.web_api import WebSettings, create_web_app

_PASSWORD = "test123!"


@pytest.fixture
def personal(tmp_path: Path) -> Iterator[tuple[TestClient, _Runner, Path]]:
    """Provide isolated persistent accounts with a deterministic editing worker."""
    settings = WebSettings(
        _env_file=None,
        access_token=_TOKEN,
        allowed_origins=(_ORIGIN,),
        jobs_dir=tmp_path / "jobs",
        accounts_enabled=True,
    )
    runner = _Runner(settings.jobs_dir)
    with TestClient(create_web_app(settings, runner)) as client:
        yield client, runner, tmp_path / "accounts/accounts.sqlite3"


def _register(client: TestClient, email: str) -> dict[str, str]:
    response = client.post("/api/auth/register", json={"email": email, "password": _PASSWORD})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["session_token"]}


def test_register_login_logout_and_persistent_hashes(
    personal: tuple[TestClient, _Runner, Path],
) -> None:
    """Case-normalized accounts persist, while logout invalidates their hashed sessions."""
    client, _, database = personal
    auth = _register(client, " First@Example.com ")
    assert client.get("/api/auth/me", headers=auth).json()["email"] == "first@example.com"
    with sqlite3.connect(database) as db:
        user = db.execute("SELECT salt, password_hash FROM users").fetchone()
        assert len(user[0]) == 16 and len(user[1]) == 32
        assert _PASSWORD.encode() not in database.read_bytes()
        assert auth["Authorization"].encode() not in database.read_bytes()
    assert AccountStore(database).resolve(auth["Authorization"][7:]) is not None
    bad = client.post(
        "/api/auth/login",
        json={"email": "first@example.com", "password": "a different long password"},
    )
    assert bad.status_code == 401
    login = client.post(
        "/api/auth/login", json={"email": "FIRST@example.com", "password": _PASSWORD}
    )
    assert login.status_code == 200
    assert login.json()["session_token"] != auth["Authorization"][7:]
    assert client.post("/api/auth/logout", headers=auth).status_code == 200
    assert client.get("/api/jobs", headers=auth).status_code == 401


@pytest.mark.parametrize(
    "route,method",
    [
        ("", "GET"),
        ("/stop", "POST"),
        ("/resume", "POST"),
        ("/refine", "POST"),
        ("/media-ticket", "POST"),
        ("/movie", "GET"),
        ("/subtitles", "GET"),
    ],
)
def test_other_account_cannot_access_or_control_project(
    personal: tuple[TestClient, _Runner, Path],
    route: str,
    method: str,
) -> None:
    """Knowing another user's project ID grants neither files nor processing controls."""
    client, _, _ = personal
    first = _register(client, "first@example.com")
    second = _register(client, "second@example.com")
    created = client.post("/api/jobs", headers=first, files={"archive": ("show.zip", _archive())})
    assert created.status_code == 201
    job_id = created.json()["state"]["job_id"]
    assert len(client.get("/api/jobs", headers=first).json()) == 1
    assert client.get("/api/jobs", headers=second).json() == []
    assert (
        client.request(
            method,
            f"/api/jobs/{job_id}{route}",
            headers=second,
            json={"must_keep_ranges": []} if route == "/refine" else None,
        ).status_code
        == 404
    )


def test_legacy_jobs_stay_private_and_media_tickets_revoke_on_logout(
    personal: tuple[TestClient, _Runner, Path],
) -> None:
    """Unclaimed jobs belong to the operator; account tickets stop working after logout."""
    client, runner, database = personal
    _state(runner, job_id="legacy", done=True)
    auth = _register(client, "first@example.com")
    assert client.get("/api/jobs", headers=auth).json() == []
    assert client.get("/api/jobs/legacy", headers=auth).status_code == 404
    assert len(client.get("/api/jobs", headers=_AUTH).json()) == 1
    _state(runner, job_id="owned", done=True)
    store = AccountStore(database)
    user = store.resolve(auth["Authorization"][7:])
    assert user is not None
    store.assign("owned", user.id)
    ticket = client.post("/api/jobs/owned/media-ticket", headers=auth).json()
    assert client.get(ticket["movie_url"]).status_code == 200
    client.post("/api/auth/logout", headers=auth)
    assert client.get(ticket["movie_url"]).status_code == 401
    assert client.get(ticket["subtitles_url"]).status_code == 401


def test_account_rate_limit_and_expiry(personal: tuple[TestClient, _Runner, Path]) -> None:
    """Reject expired sessions and bound expensive repeated password attempts."""
    client, _, database = personal
    auth = _register(client, "first@example.com")
    with sqlite3.connect(database) as db:
        db.execute("UPDATE sessions SET expires=0")
    assert client.get("/api/jobs", headers=auth).status_code == 401
    store = AccountStore(database)
    for _ in range(8):
        store.throttle("another-source", "other@example.com")
    with pytest.raises(HTTPException) as rejected:
        store.throttle("another-source", "other@example.com")
    assert rejected.value.status_code == 429


def test_refined_project_inherits_owner_and_account_cap(
    personal: tuple[TestClient, _Runner, Path],
) -> None:
    """New edits retain ownership, and an exhausted registration cap creates no user."""
    client, runner, database = personal
    first = _register(client, "first@example.com")
    second = _register(client, "second@example.com")
    _state(runner, job_id="owned", done=True)
    store = AccountStore(database, max_accounts=2)
    user = store.resolve(first["Authorization"][7:])
    assert user is not None
    store.assign("owned", user.id)
    response = client.post("/api/jobs/owned/refine", headers=first, json={"must_keep_ranges": []})
    assert response.status_code == 201
    assert client.get("/api/jobs/refined", headers=first).status_code == 200
    assert client.get("/api/jobs/refined", headers=second).status_code == 404
    with pytest.raises(HTTPException) as rejected:
        store.register("third@example.com", _PASSWORD)
    assert rejected.value.status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"email": "invalid", "password": _PASSWORD},
        {"email": "first@example.com", "password": "short12"},
        {"email": "first@example.com", "password": _PASSWORD, "role": "admin"},
    ],
)
def test_invalid_account_inputs_do_not_create_accounts(
    personal: tuple[TestClient, _Runner, Path],
    body: dict[str, str],
) -> None:
    """Validate the account contract and reject role injection and weak passwords."""
    client, _, database = personal
    assert client.post("/api/auth/register", json=body).status_code == 422
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0

"""Shared pytest setup for the initial, intentionally empty scaffold."""

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Allow the empty Phase 0 suite; remove this hook when Phase 1 adds tests."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK

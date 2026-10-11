"""Exercise the actual browser upload flow without network access or JS packages."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "case",
    [
        "small-disconnected",
        "account-register-preserves-upload",
        "account-password-mismatch",
        "account-login-failure",
        "oversize-replacement",
        "empty-replacement",
        "nonzip-replacement",
        "inclusive-size-boundary",
        "connected-limit",
        "connect-preserves-form",
        "connect-refreshes-limit",
        "reconnect-clears-size-error",
    ],
)
def test_frontend_upload_and_connection_flow(case: str) -> None:
    """Run the whole frontend against a minimal DOM and deterministic transport doubles."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the isolated frontend functional harness")
    harness = Path(__file__).with_name("web_frontend_upload.test.cjs")
    result = subprocess.run(
        [node, str(harness), case],
        cwd=harness.parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"Frontend case {case} failed:\n{result.stdout}\n{result.stderr}"

"""Optional Windows vendor DLL discovery without loading trained models or CUDA."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sts.config import Settings
from sts.pipeline import transcribe


def test_windows_cuda_discovers_installed_vendor_dlls_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Virtual-environment DLL folders precede existing paths without repeated additions."""
    folders = [tmp_path / "nvidia" / vendor / "bin" for vendor in ("cublas", "cudnn")]
    for folder in folders:
        folder.mkdir(parents=True)
        (folder / "runtime.dll").write_bytes(b"not loaded")
    empty = tmp_path / "nvidia" / "cuda_nvrtc" / "bin"
    empty.mkdir(parents=True)
    monkeypatch.setattr(transcribe.sys, "platform", "win32")
    monkeypatch.setattr(transcribe.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setenv("PATH", "existing")
    cfg = Settings(whisper_device="cuda")
    transcribe._prepare_cuda_runtime(cfg)
    expected = os.pathsep.join([*(str(folder) for folder in folders), "existing"])
    assert os.environ["PATH"] == expected
    transcribe._prepare_cuda_runtime(cfg)
    assert os.environ["PATH"] == expected


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_missing_optional_windows_packages_preserves_path(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent vendor runtime leaves the normal system CUDA search path intact."""
    monkeypatch.setattr(transcribe.sys, "platform", "win32")
    monkeypatch.setattr(transcribe.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setenv("PATH", "existing")
    transcribe._prepare_cuda_runtime(Settings(whisper_device=device))
    assert os.environ["PATH"] == "existing"


@pytest.mark.parametrize(("platform", "device"), [("win32", "cpu"), ("linux", "cuda")])
def test_cpu_and_non_windows_do_not_scan_or_change_dll_paths(
    platform: str, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU jobs and other operating systems retain their original dependency behavior."""
    def unexpected_scan(name: str) -> str:
        raise AssertionError("Optional vendor folders must not be scanned")

    monkeypatch.setattr(transcribe.sys, "platform", platform)
    monkeypatch.setattr(transcribe.sysconfig, "get_path", unexpected_scan)
    monkeypatch.setenv("PATH", "existing")
    transcribe._prepare_cuda_runtime(Settings(whisper_device=device))
    assert os.environ["PATH"] == "existing"

"""Тесты OSS-фикса: больше нет хардкода пути к llama-server."""

from __future__ import annotations

from pathlib import Path

from src.rag.config import _resolve_llama_server_exe


def test_env_overrides_path(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_SERVER_EXE", "/custom/path/llama-server")
    assert _resolve_llama_server_exe() == Path("/custom/path/llama-server")


def test_falls_back_to_path_lookup(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_EXE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/llama-server")
    assert _resolve_llama_server_exe() == Path("/usr/local/bin/llama-server")


def test_sentinel_when_nothing_found(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_EXE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    # Не падать на импорте — сигнальный путь, ошибка при фактическом spawn.
    assert _resolve_llama_server_exe() == Path("llama-server")

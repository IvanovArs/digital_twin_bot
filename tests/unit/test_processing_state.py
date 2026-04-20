"""Tests for the rid-keyed processing-state map."""

from __future__ import annotations

import pytest

from src.bot.services import processing_state


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    processing_state._STATE.clear()


def test_new_rid_is_unique_and_short() -> None:
    rids = {processing_state.new_rid() for _ in range(50)}
    assert len(rids) == 50
    assert all(len(r) == 12 for r in rids)


def test_set_and_get_status() -> None:
    rid = processing_state.new_rid()
    processing_state.set_status(rid, "stage A")
    assert processing_state.get_status(rid) == "stage A"
    processing_state.set_status(rid, "stage B")
    assert processing_state.get_status(rid) == "stage B"


def test_clear_removes_status() -> None:
    rid = processing_state.new_rid()
    processing_state.set_status(rid, "x")
    processing_state.clear(rid)
    assert processing_state.get_status(rid) is None


def test_get_unknown_rid_returns_none() -> None:
    assert processing_state.get_status("not-a-rid") is None


def test_ttl_expires_old_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = processing_state.new_rid()
    processing_state.set_status(rid, "old")

    # Fast-forward past the TTL by patching time.time inside the module.
    real_now = processing_state._STATE[rid][1]
    monkeypatch.setattr(
        processing_state.time, "time", lambda: real_now + processing_state._TTL_S + 1
    )
    assert processing_state.get_status(rid) is None

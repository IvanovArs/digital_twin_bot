"""Tests for the superadmin/role helpers in ``src.bot.handlers.admin``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.bot.handlers.admin import _parse_role
from src.db.models import UserRole


@dataclass
class _FakeUser:
    telegram_id: int
    role: UserRole = UserRole.student


# ---------- _parse_role ----------


def test_parse_role_recognises_canonical_values() -> None:
    assert _parse_role("teacher") is UserRole.teacher
    assert _parse_role("admin") is UserRole.admin
    assert _parse_role("student") is UserRole.student


def test_parse_role_is_case_insensitive_and_trims() -> None:
    assert _parse_role("  Teacher ") is UserRole.teacher
    assert _parse_role("ADMIN") is UserRole.admin


def test_parse_role_defaults_to_teacher_on_none_or_empty() -> None:
    assert _parse_role(None) is UserRole.teacher
    assert _parse_role("") is UserRole.teacher


def test_parse_role_rejects_unknown_value() -> None:
    assert _parse_role("superadmin") is None
    assert _parse_role("root") is None
    assert _parse_role("123") is None


def test_parse_role_accepts_custom_default() -> None:
    """``admin_users`` uses ``default=None`` so an empty arg means "no filter",
    not "filter by teacher". Passing ``None`` as default must pass through."""
    assert _parse_role(None, default=None) is None  # type: ignore[arg-type]


# ---------- _is_superadmin ----------


def test_is_superadmin_true_when_in_env(monkeypatch) -> None:
    from src.bot.handlers import admin as mod
    from src.config import settings

    # Patch the settings instance the handler module already bound at import
    # time. Resetting ADMIN_TELEGRAM_IDS on the Pydantic model instance isn't
    # enough because ``admin_ids`` is a @property — we override the getter.
    monkeypatch.setattr(
        type(settings), "admin_ids", property(lambda self: {12345, 67890})
    )
    assert mod._is_superadmin(_FakeUser(telegram_id=12345)) is True
    assert mod._is_superadmin(_FakeUser(telegram_id=67890)) is True
    assert mod._is_superadmin(_FakeUser(telegram_id=99999)) is False


def test_is_superadmin_false_when_env_empty(monkeypatch) -> None:
    from src.bot.handlers import admin as mod
    from src.config import settings

    monkeypatch.setattr(type(settings), "admin_ids", property(lambda self: set()))
    assert mod._is_superadmin(_FakeUser(telegram_id=12345)) is False


def test_is_superadmin_independent_of_db_role(monkeypatch) -> None:
    """A DB-level admin (role=admin) is NOT automatically a superadmin —
    only the env-var list counts. Prevents self-escalation of DB admins."""
    from src.bot.handlers import admin as mod
    from src.config import settings

    monkeypatch.setattr(type(settings), "admin_ids", property(lambda self: {111}))
    db_admin = _FakeUser(telegram_id=222, role=UserRole.admin)
    assert mod._is_superadmin(db_admin) is False


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("teacher", UserRole.teacher),
        ("admin", UserRole.admin),
        ("student", UserRole.student),
        ("TEACHER", UserRole.teacher),
        ("  admin  ", UserRole.admin),
    ],
)
def test_parse_role_parametric(arg: str, expected: UserRole) -> None:
    assert _parse_role(arg) is expected

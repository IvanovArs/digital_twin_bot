"""Tests for the HTML safety helpers in qa_pipeline.

These three functions ship with every Telegram answer, so a regression here
breaks every conversation. Cover the failure modes we already hit in prod:
orphan close tag, unclosed open tag, oversized body.
"""

from __future__ import annotations

from src.bot.services.safe_html import (
    balance_tags as _balance_tags,
)
from src.bot.services.safe_html import (
    safe_html as _safe_html,
)
from src.bot.services.safe_html import (
    truncate_for_telegram as _truncate_for_telegram,
)


def test_safe_html_keeps_allowed_tags() -> None:
    out = _safe_html("Hello <b>world</b> and <code>x</code>")
    assert "<b>world</b>" in out
    assert "<code>x</code>" in out


def test_safe_html_escapes_disallowed_tags() -> None:
    # <script> is not in the whitelist → it stays escaped.
    out = _safe_html("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_balance_tags_drops_orphan_close() -> None:
    # Orphan </b> with no matching open → drop it; surrounding text stays.
    assert _balance_tags("</b>foo") == "foo"
    assert _balance_tags("hello </i> world") == "hello  world"


def test_balance_tags_drops_unclosed_open() -> None:
    # Unclosed <b> → drop the open tag; text content survives.
    assert _balance_tags("<b>foo") == "foo"
    assert _balance_tags("a <b>b <i>c</i>") == "a b <i>c</i>"


def test_balance_tags_keeps_balanced_pairs() -> None:
    s = "a <b>bold</b> and <i>italic</i>"
    assert _balance_tags(s) == s


def test_safe_html_balances_after_restore() -> None:
    # LLM emitted "<b>foo" — _safe_html should drop the open after balancing.
    out = _safe_html("<b>foo")
    assert "<b>" not in out
    assert "foo" in out


def test_truncate_for_telegram_no_op_under_cap() -> None:
    body = "x" * 1000
    assert _truncate_for_telegram(body) == body


def test_truncate_for_telegram_caps_at_4096() -> None:
    body = "x" * 5000
    out = _truncate_for_telegram(body)
    assert len(out) <= 4096
    assert out.endswith("…")


def test_truncate_rebalances_cut_open_tag() -> None:
    # Cap hits mid-<b>; the cut leaves an orphan open tag that would
    # reject the message. _truncate_for_telegram must rebalance.
    body = "<b>" + "x" * 5000 + "</b>"
    out = _truncate_for_telegram(body)
    assert len(out) <= 4096
    # The orphan <b> at position 0 has no matching </b> in the truncated
    # slice → balancer should drop it.
    assert "<b>" not in out or "</b>" in out

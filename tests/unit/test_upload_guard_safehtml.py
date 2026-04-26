"""Tests for the CRITICAL hardening: upload guard + FAQ/glossary sanitiser."""

from __future__ import annotations

from src.bot.services.faq_service import format_faq_body
from src.bot.services.glossary_upload import format_glossary_body
from src.bot.services.safe_html import (
    ALLOWED_TAGS,
    balance_tags,
    safe_html,
    truncate_for_telegram,
)
from src.bot.services.upload_guard import (
    MAX_UPLOAD_BYTES,
    check_magic,
    check_size,
)
from src.db.models import FAQEntry, GlossaryTerm

# ---------- safe_html ----------


def test_safe_html_keeps_whitelisted_tags() -> None:
    out = safe_html("<b>term</b> and <code>x</code>")
    assert "<b>" in out and "</b>" in out
    assert "<code>" in out and "</code>" in out


def test_safe_html_escapes_script_tag() -> None:
    out = safe_html("<script>alert(1)</script>hi")
    assert "<script>" not in out
    assert "alert" in out  # text survives
    assert "&lt;script&gt;" in out


def test_safe_html_drops_unbalanced_open() -> None:
    """Teacher typo: forgot the closing </b>. Must not break the render."""
    assert "</b>" not in safe_html("<b>typo")


def test_safe_html_drops_unbalanced_close() -> None:
    assert "</b>" not in safe_html("stray</b>")


def test_safe_html_preserves_cyrillic() -> None:
    out = safe_html("Система — это совокупность элементов.")
    assert "Система" in out
    assert "—" in out


def test_allowed_tags_frozen() -> None:
    """A PR that silently adds <img> or <a href> opens XSS — keep the list
    short and test that it hasn't grown."""
    assert set(ALLOWED_TAGS) == {
        "b",
        "strong",
        "i",
        "em",
        "u",
        "s",
        "code",
        "pre",
        "blockquote",
    }


def test_balance_tags_noop_on_clean_input() -> None:
    text = "hello <b>world</b> <i>ok</i>"
    assert balance_tags(text) == text


def test_truncate_for_telegram_caps_long_body() -> None:
    body = "x" * 5000
    out = truncate_for_telegram(body)
    assert len(out) <= 4096


# ---------- format_faq_body / format_glossary_body ----------


def test_faq_body_sanitises_teacher_input() -> None:
    entry = FAQEntry(
        subject_id=None,
        question_normalised="x",
        question_original="q",
        answer="<b>хороший</b> ответ и <script>alert(1)</script>",
    )
    out = format_faq_body(entry, "ru")
    assert "<b>" in out  # intentional formatting survived
    assert "<script>" not in out  # attack neutralised
    assert "alert(1)" in out  # text preserved as data


def test_faq_body_balances_stray_close_tag() -> None:
    entry = FAQEntry(
        subject_id=None,
        question_normalised="x",
        question_original="q",
        answer="стрей тег</b> и всё",
    )
    out = format_faq_body(entry, "ru")
    # The orphan </b> from the teacher body is dropped; the legitimate
    # <b>…</b> from the "📌 Ответ преподавателя" prefix must survive
    # balanced (equal opens and closes).
    assert out.count("<b>") == out.count("</b>")
    assert "стрей тег и всё" in out


def test_glossary_body_sanitises_definition() -> None:
    entry = GlossaryTerm(
        subject_id=1,
        term="<b>Термин</b>",  # teacher used markup in the term too
        definition="<b>bold</b> plus <a href=evil>click</a>",
    )
    out = format_glossary_body(entry, "ru")
    assert "<b>" in out
    assert "<a href" not in out  # unknown tag escaped
    assert "evil" in out  # raw text preserved


# ---------- upload_guard ----------


def test_check_size_accepts_small_file() -> None:
    assert check_size(1024) is None


def test_check_size_rejects_zero() -> None:
    err = check_size(0)
    assert err is not None
    assert "пуст" in err.lower()


def test_check_size_rejects_oversized() -> None:
    err = check_size(MAX_UPLOAD_BYTES + 1)
    assert err is not None
    assert "слишком" in err.lower()


def test_check_size_passes_on_unknown() -> None:
    """Telegram may omit file_size for forwarded files — skip the check."""
    assert check_size(None) is None


def test_check_magic_accepts_pdf() -> None:
    payload = b"%PDF-1.7\nfake body"
    assert check_magic("book.pdf", payload) is None


def test_check_magic_rejects_fake_pdf() -> None:
    err = check_magic("book.pdf", b"I am plain text, not PDF")
    assert err is not None
    assert "PDF" in err


def test_check_magic_accepts_docx() -> None:
    # DOCX is a ZIP archive — PK\x03\x04 local-file header.
    payload = b"PK\x03\x04" + b"\x00" * 100
    assert check_magic("notes.docx", payload) is None


def test_check_magic_rejects_fake_docx() -> None:
    err = check_magic("notes.docx", b"not a zip")
    assert err is not None


def test_check_magic_accepts_utf8_text_files() -> None:
    payload = "Система — это совокупность".encode()
    for name in ("ch.txt", "ch.md", "ch.csv", "g.yaml"):
        assert check_magic(name, payload) is None, f"failed on {name}"


def test_check_magic_rejects_binary_disguised_as_text() -> None:
    # Random non-utf-8 bytes in a "txt" file.
    err = check_magic("ch.txt", b"\xff\xfe\xfd\xfc invalid utf-8")
    assert err is not None


def test_check_magic_rejects_empty_payload() -> None:
    assert check_magic("x.pdf", b"") is not None

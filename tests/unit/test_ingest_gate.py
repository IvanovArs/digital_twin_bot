"""Tests for the OCR quality gate baked into ``_build_chunks_for_subject``.

We don't want to exercise the real PDF + Tesseract path in a unit test, so we
monkey-patch ``_extract_pdf`` to return canned pages and let the rest of the
function (chunking + garbage filter + book-level reject) run for real.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Subj:
    slug: str = "test_subject"
    title_ru: str = "Тест"
    title_en: str = "Test"


def _fake_book_pages_clean() -> list[tuple[int, str]]:
    """12 healthy pages of Russian prose."""
    body = (
        "Стейкхолдер — это физическое или юридическое лицо, которое может "
        "повлиять на достижение организацией своих целей или на работу "
        "организации в целом. Заинтересованные стороны включают акционеров, "
        "поставщиков, клиентов, сотрудников и государственные органы."
    )
    return [(i, body) for i in range(1, 13)]


def _fake_book_pages_garbage() -> list[tuple[int, str]]:
    """Pure OCR-debris pages — should trigger the per-book reject."""
    bad = (
        "Gow Бергаланфи определил экаифинатьность как pearpasse в мирытых "
        "системах. Wenartpompyeaie noxxon ‘Tor термин предельных состояний "
        "fades ‘oro Mow системы при полностмо начальных условиях зависяще."
    )
    return [(i, bad) for i in range(1, 13)]


def _fake_book_pages_english() -> list[tuple[int, str]]:
    """12 healthy pages of English prose — must NOT trip the RU filter."""
    body = (
        "A stakeholder is a person or organisation that can affect or be "
        "affected by the achievement of an organisation's objectives. "
        "Stakeholders include shareholders, suppliers, customers, employees, "
        "and government regulators. Analysing stakeholder relations is a "
        "core activity in strategic management and systems theory."
    )
    return [(i, body) for i in range(1, 13)]


def test_gate_accepts_clean_book(monkeypatch, tmp_path) -> None:
    from src.rag import ingest

    subj_dir = tmp_path / "test_subject"
    subj_dir.mkdir()
    (subj_dir / "clean.pdf").write_bytes(b"")  # file has to exist for _gather_books

    monkeypatch.setattr(ingest, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "_extract_pdf", lambda p: _fake_book_pages_clean())

    chunks, reports = ingest._build_chunks_for_subject(_Subj())
    assert len(reports) == 1
    r = reports[0]
    assert not r.rejected
    assert r.kept_chunks > 0
    assert chunks  # chunks actually flowed through


def test_gate_rejects_garbage_book(monkeypatch, tmp_path) -> None:
    from src.rag import ingest

    subj_dir = tmp_path / "test_subject"
    subj_dir.mkdir()
    (subj_dir / "trash.pdf").write_bytes(b"")

    monkeypatch.setattr(ingest, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "_extract_pdf", lambda p: _fake_book_pages_garbage())

    chunks, reports = ingest._build_chunks_for_subject(_Subj())
    assert len(reports) == 1
    r = reports[0]
    assert r.rejected
    assert r.kept_chunks == 0
    assert r.garbage_ratio > 0.30
    assert chunks == []  # nothing from a rejected book reaches the index


def test_force_flag_overrides_reject(monkeypatch, tmp_path) -> None:
    from src.rag import ingest

    subj_dir = tmp_path / "test_subject"
    subj_dir.mkdir()
    (subj_dir / "trash.pdf").write_bytes(b"")

    monkeypatch.setattr(ingest, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "_extract_pdf", lambda p: _fake_book_pages_garbage())

    chunks, reports = ingest._build_chunks_for_subject(_Subj(), force=True)
    r = reports[0]
    # Under --force we don't reject — but clean chunks are still the only
    # ones kept, and the report still records the high garbage ratio so the
    # admin sees the warning.
    assert not r.rejected
    assert r.garbage_ratio > 0.30
    # Garbage chunks still get filtered out; only non-garbage survives. With
    # all-garbage input that might be zero, which is fine — admin was warned.
    assert len(chunks) == r.kept_chunks


def test_gate_accepts_english_book(monkeypatch, tmp_path) -> None:
    """C-ticket regression: an English textbook must not be falsely rejected
    by the "too much Latin in a Russian index" rule. Language-agnostic
    detection flips to EN mode on the fly."""
    from src.rag import ingest

    subj_dir = tmp_path / "test_subject"
    subj_dir.mkdir()
    (subj_dir / "english.pdf").write_bytes(b"")

    monkeypatch.setattr(ingest, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "_extract_pdf", lambda p: _fake_book_pages_english())

    chunks, reports = ingest._build_chunks_for_subject(_Subj())
    assert len(reports) == 1
    r = reports[0]
    assert not r.rejected
    assert r.kept_chunks > 0
    assert chunks


def test_book_report_garbage_ratio_formula() -> None:
    from src.rag.ingest import BookReport

    r = BookReport("s", "b.pdf", total_chunks=10, kept_chunks=7, rejected=False)
    assert abs(r.garbage_ratio - 0.3) < 1e-9
    r2 = BookReport("s", "b.pdf", total_chunks=0, kept_chunks=0, rejected=False)
    assert r2.garbage_ratio == 0.0

"""Тесты helper'ов ingest: _clean, _extract_txt/_md/_docx, _gather_books, _print_quality_report."""

from __future__ import annotations

from pathlib import Path

from src.rag.ingest import (
    BookReport,
    _clean,
    _encode,
    _extract_md,
    _extract_txt,
    _gather_books,
    _print_quality_report,
)


def test_clean_removes_soft_hyphens_and_zwsp() -> None:
    """Soft-hyphen (U+00AD) и zero-width-space (U+200B) должны срезаться."""
    raw = "сис­тема​ный анализ"
    out = _clean(raw)
    assert "­" not in out
    assert "​" not in out
    assert "системаный анализ" in out


def test_clean_collapses_whitespace() -> None:
    raw = "  слово   ещё   слово  "
    assert _clean(raw) == "слово ещё слово"


def test_clean_collapses_multinewline() -> None:
    raw = "первый\n\n\n\nвторой\n\n\nтретий"
    out = _clean(raw)
    # Не более одного двойного перевода строки подряд.
    assert "\n\n\n" not in out


def test_extract_txt_splits_on_form_feed(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("первая страница\f\fвторая страница", encoding="utf-8")
    pages = _extract_txt(p)
    assert len(pages) == 2
    assert "первая" in pages[0][1]
    assert "вторая" in pages[1][1]


def test_extract_txt_handles_invalid_utf8(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello\xff\xfeworld")
    pages = _extract_txt(p)
    assert pages
    assert "hello" in pages[0][1] and "world" in pages[0][1]


def test_extract_md_strips_fences_and_headings(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text(
        "# Заголовок\n\n"
        "Это **жирный** текст и [ссылка](http://example.com).\n\n"
        "```python\nprint('код')\n```\n\n"
        "Конец.",
        encoding="utf-8",
    )
    pages = _extract_md(p)
    body = " ".join(t for _, t in pages)
    assert "Заголовок" in body
    assert "жирный" in body
    assert "**" not in body
    assert "ссылка" in body
    assert "print" not in body  # код-блок вырезан


def test_gather_books_picks_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.md").write_text("y")
    (tmp_path / "d.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "ignore.exe").write_bytes(b"MZ")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    out = _gather_books(tmp_path)
    names = {p.name for p in out}
    assert names == {"a.pdf", "b.txt", "c.md", "d.docx"}


def test_gather_books_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert _gather_books(tmp_path / "ghost") == []


def test_print_quality_report_runs_on_empty(capsys) -> None:
    # Не падает на пустом списке.
    _print_quality_report([])


def test_print_quality_report_lists_books(capsys) -> None:
    reports = [
        BookReport(
            subject_slug="theory_of_systems",
            book="a.pdf",
            total_chunks=20,
            kept_chunks=18,
            rejected=False,
            garbage_pages=[3, 5],
        )
    ]
    _print_quality_report(reports)
    out = capsys.readouterr().out
    assert "a.pdf" in out


def test_encode_single_batch() -> None:
    import numpy as np

    class FakeModel:
        def encode(self, texts, **kw):
            return np.ones((len(texts), 4), dtype=np.float32)

    out = _encode(model=FakeModel(), texts=["a", "b"], batch=8)  # type: ignore[arg-type]
    assert out.shape == (2, 4)


def test_encode_calls_model_in_batches() -> None:
    import numpy as np

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(
            self,
            texts,
            *,
            batch_size=None,
            normalize_embeddings=None,
            convert_to_numpy=None,
            show_progress_bar=None,
        ):
            self.calls.append(list(texts))
            return np.ones((len(texts), 4), dtype=np.float32)

    fm = FakeModel()
    out = _encode(model=fm, texts=["a", "b", "c"], batch=2)  # type: ignore[arg-type]
    assert out.shape == (3, 4)
    # SentenceTransformer.encode сам бьёт на батчи — мы передаём всё разом.
    assert sum(len(c) for c in fm.calls) == 3


def test_book_report_dataclass_defaults() -> None:
    r = BookReport(
        subject_slug="theory_of_systems",
        book="x.pdf",
        total_chunks=0,
        kept_chunks=0,
        rejected=False,
    )
    assert r.garbage_pages == []

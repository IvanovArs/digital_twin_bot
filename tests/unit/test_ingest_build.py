"""Тесты build_index + _build_chunks_for_subject + persist helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.rag import ingest as ing_mod
from src.rag.ingest import (
    _build_chunks_for_subject,
    _load_existing_chunks,
    _write_chunks,
    build_index,
)
from src.subjects.schema import Subject as SubjectCfg


def test_build_chunks_for_subject_no_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", tmp_path)
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    chunks, reports = _build_chunks_for_subject(subj)
    assert chunks == []
    assert reports == []


def test_build_chunks_for_subject_empty_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", tmp_path)
    (tmp_path / "theory_of_systems").mkdir()
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    chunks, reports = _build_chunks_for_subject(subj)
    assert chunks == []
    assert reports == []


def test_build_chunks_for_subject_processes_txt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", tmp_path)
    book_dir = tmp_path / "theory_of_systems"
    book_dir.mkdir()
    (book_dir / "lecture.txt").write_text(
        "Стейкхолдер — это лицо или организация, заинтересованные в проекте. "
        * 30,
        encoding="utf-8",
    )
    subj = SubjectCfg(slug="theory_of_systems", title_en="TOS", title_ru="ТС")
    chunks, reports = _build_chunks_for_subject(subj)
    assert chunks
    assert reports
    assert reports[0].book == "lecture.txt"


def test_load_existing_chunks_returns_empty_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", tmp_path / "ghost.jsonl")
    assert _load_existing_chunks() == []


def test_load_existing_chunks_reads_jsonl(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        json.dumps({"text": "x", "subject_slug": "theory_of_systems", "book": "a.pdf", "page": 1})
        + "\n"
        + json.dumps({"text": "y", "subject_slug": "theory_of_systems", "book": "a.pdf", "page": 2}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", p)
    out = _load_existing_chunks()
    assert len(out) == 2
    assert out[0]["text"] == "x"


def test_write_chunks_round_trip(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", p)
    chunks = [
        {"text": "Стейкхолдер", "subject_slug": "theory_of_systems", "book": "a.pdf", "page": 1, "chunk_idx": 0},
        {"text": "Эмерджентность", "subject_slug": "theory_of_systems", "book": "a.pdf", "page": 2, "chunk_idx": 1},
    ]
    _write_chunks(chunks)
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", p)
    out = _load_existing_chunks()
    assert len(out) == 2
    assert out[0]["text"] == "Стейкхолдер"


def test_build_index_full_rebuild_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Прогон build_index без only_subject: создаёт chunks.jsonl, embeddings.npy, meta.json."""
    books = tmp_path / "books"
    index = tmp_path / "index"
    courses_yaml = tmp_path / "courses.yaml"
    courses_yaml.write_text(
        "subjects:\n  - slug: theory_of_systems\n    title_en: TOS\n    title_ru: Теория\n",
        encoding="utf-8",
    )
    (books / "theory_of_systems").mkdir(parents=True)
    (books / "theory_of_systems" / "x.txt").write_text(
        ("Стейкхолдер — это лицо, заинтересованное в проекте. " * 50),
        encoding="utf-8",
    )
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", books)
    monkeypatch.setattr(ing_mod, "INDEX_DIR", index)
    monkeypatch.setattr(ing_mod, "EMBEDDINGS_FILE", index / "embeddings.npy")
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", index / "chunks.jsonl")
    monkeypatch.setattr(ing_mod, "INDEX_META_FILE", index / "meta.json")
    monkeypatch.setattr(ing_mod, "COURSES_YAML", courses_yaml)

    # Подменяем тяжёлый SentenceTransformer на mock — без скачки 2 ГБ.
    class FakeST:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def encode(self, texts, **kw):  # type: ignore[no-untyped-def]
            return np.ones((len(texts), 8), dtype=np.float32)

    with patch.object(ing_mod, "SentenceTransformer", FakeST):
        build_index()

    assert (index / "embeddings.npy").exists()
    assert (index / "chunks.jsonl").exists()
    assert (index / "meta.json").exists()
    meta = json.loads((index / "meta.json").read_text(encoding="utf-8"))
    assert meta["dim"] == 8
    assert meta["rows"] > 0


def test_build_index_partial_subject_keeps_others(tmp_path: Path, monkeypatch) -> None:
    """Partial-rebuild: чанки одного предмета пересобираются, остальные сохраняются."""
    books = tmp_path / "books"
    index = tmp_path / "index"
    courses_yaml = tmp_path / "courses.yaml"
    courses_yaml.write_text(
        "subjects:\n"
        "  - slug: theory_of_systems\n    title_en: TOS\n    title_ru: ТС\n"
        "  - slug: systems_engineering\n    title_en: SE\n    title_ru: СИ\n",
        encoding="utf-8",
    )
    natural_text = (
        "Стейкхолдер — это лицо или организация, заинтересованные в проекте. "
        "Эмерджентность означает свойство целого, отсутствующее у частей. "
        "Системный анализ — методика декомпозиции и интеграции элементов. "
        "Подсистема — часть системы, выделенная по функциональному признаку."
    )
    for slug in ("theory_of_systems", "systems_engineering"):
        (books / slug).mkdir(parents=True)
        (books / slug / f"{slug}.txt").write_text(
            natural_text * 30, encoding="utf-8"
        )
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", books)
    monkeypatch.setattr(ing_mod, "INDEX_DIR", index)
    monkeypatch.setattr(ing_mod, "EMBEDDINGS_FILE", index / "embeddings.npy")
    monkeypatch.setattr(ing_mod, "CHUNKS_FILE", index / "chunks.jsonl")
    monkeypatch.setattr(ing_mod, "INDEX_META_FILE", index / "meta.json")
    monkeypatch.setattr(ing_mod, "COURSES_YAML", courses_yaml)

    class FakeST:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def encode(self, texts, **kw):  # type: ignore[no-untyped-def]
            return np.ones((len(texts), 4), dtype=np.float32)

    with patch.object(ing_mod, "SentenceTransformer", FakeST):
        build_index()
        # Partial rebuild только TOS.
        build_index(only_subject="theory_of_systems")

    chunks = _load_existing_chunks()
    slugs = {c["subject_slug"] for c in chunks}
    assert "theory_of_systems" in slugs
    assert "systems_engineering" in slugs


def test_build_index_unknown_subject_exits(tmp_path: Path, monkeypatch) -> None:
    books = tmp_path / "books"
    index = tmp_path / "index"
    courses_yaml = tmp_path / "courses.yaml"
    courses_yaml.write_text(
        "subjects:\n  - slug: theory_of_systems\n    title_en: TOS\n    title_ru: ТС\n",
        encoding="utf-8",
    )
    (books / "theory_of_systems").mkdir(parents=True)
    (books / "theory_of_systems" / "x.txt").write_text("x" * 1000, encoding="utf-8")
    monkeypatch.setattr(ing_mod, "BOOKS_DIR", books)
    monkeypatch.setattr(ing_mod, "INDEX_DIR", index)
    monkeypatch.setattr(ing_mod, "COURSES_YAML", courses_yaml)
    with pytest.raises(SystemExit):
        build_index(only_subject="ghost_subject")

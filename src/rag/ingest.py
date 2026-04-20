"""Build the unified RAG index from all subjects in courses.yaml.

One global index lives at:
    data/index/embeddings.npy  — float32 matrix [N, D], L2-normalized
    data/index/chunks.jsonl    — one JSON line per row, with subject_slug + source metadata
    data/index/meta.json       — embedding model, chunk params, timestamp

Usage:
    python -m src.rag.ingest                 # re-build from scratch
    python -m src.rag.ingest --subject slug  # rebuild only one subject's chunks
                                             #   (embedding model must match existing)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Windows console defaults to cp1251 in RU locales — emoji like "✅" and
# Cyrillic letters in print() crash the script with UnicodeEncodeError.
# Force UTF-8 on stdout/stderr before any print fires so progress and the
# final summary survive.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from src.rag.chunking import Chunk, split_pages_into_chunks
from src.rag.config import (
    BOOKS_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    COURSES_YAML,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_DIR,
    INDEX_META_FILE,
    MODELS_DIR,
)
from src.rag.ocr_filter import is_ocr_garbage_auto

# OCR setup: PyMuPDF's get_textpage_ocr() shells out to Tesseract via the
# TESSDATA_PREFIX env var. We ship rus/eng/osd traineddata under
# data/models/tessdata/ and point Tesseract at it before importing fitz so
# scanned Russian textbooks (e.g. Volkova) get OCR'd instead of silently
# producing zero chunks.
_LOCAL_TESSDATA = MODELS_DIR / "tessdata"
if _LOCAL_TESSDATA.exists() and "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = str(_LOCAL_TESSDATA)
# Make sure tesseract.exe is reachable on Windows even if the user didn't
# add it to their PATH; UB-Mannheim's installer puts it in a fixed location.
if sys.platform == "win32":
    _DEFAULT_TESS_DIR = r"C:\Program Files\Tesseract-OCR"
    if Path(_DEFAULT_TESS_DIR, "tesseract.exe").exists():
        os.environ["PATH"] = _DEFAULT_TESS_DIR + os.pathsep + os.environ.get("PATH", "")

import fitz  # noqa: E402  — must come after PATH/TESSDATA env tweaks
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.subjects import Subject, load_catalog  # noqa: E402

_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
# Soft hyphens (U+00AD) and the non-breaking space variants PyMuPDF emits
# from Russian PDFs — they wreck both display and embedding tokenisation.
_INVISIBLE = str.maketrans({"\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": ""})


def _clean(raw: str) -> str:
    text = raw.translate(_INVISIBLE)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _extract_pdf(path: Path, *, ocr_lang: str = "rus+eng") -> list[tuple[int, str]]:
    """Extract text from every page.

    First try the native PDF text layer; if a page has none (typical for scanned
    textbooks), fall back to OCR via PyMuPDF's built-in Tesseract bridge.
    OCR requires Tesseract to be installed on the system with the requested
    language data files. If Tesseract is missing, OCR is silently skipped and
    only pages with a text layer are ingested — the user is warned on stderr.
    """
    pages: list[tuple[int, str]] = []
    ocr_attempted = False
    ocr_available = True  # set to False on first failure
    ocr_success_pages = 0

    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))

            if not text and ocr_available:
                ocr_attempted = True
                try:
                    tp = page.get_textpage_ocr(language=ocr_lang, full=True)
                    text = _clean(page.get_text("text", textpage=tp))
                    if text:
                        ocr_success_pages += 1
                except Exception as exc:  # RuntimeError if Tesseract not found
                    ocr_available = False
                    print(
                        f"  [warn] OCR недоступен ({exc}). "
                        "Пропускаю страницы без текстового слоя. "
                        "Установи Tesseract (+ tessdata для rus/eng) и пересобери индекс.",
                        file=sys.stderr,
                    )

            if text:
                pages.append((page_num, text))

    if ocr_attempted and ocr_success_pages:
        print(f"      [ocr] распознано страниц: {ocr_success_pages}")
    if not pages:
        # Loud failure beats silent data loss. A scanned PDF with no text
        # layer + missing Tesseract was producing 0-page books that quietly
        # vanished from the index for weeks before anyone noticed.
        print(
            f"      [ERROR] {path.name}: 0 страниц извлечено! "
            "Возможно, это сканированный PDF без текстового слоя, "
            "а Tesseract не установлен. Установи Tesseract + tessdata "
            "(rus/eng) и пересобери индекс.",
            file=sys.stderr,
        )
    return pages


def _extract_txt(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\f|\n{4,}", raw)
    return [(i, _clean(p)) for i, p in enumerate(parts, start=1) if p.strip()]


# Lightweight Markdown cleanup: drop fenced code blocks and strip the most
# common inline markup so the embedding sees prose, not syntax. Anything
# more elaborate (reference-style links, tables) is rare in lecture notes;
# the embedder is robust to leftover punctuation.
_MD_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")
_MD_HEADING = re.compile(r"^#+\s*", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _extract_md(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _MD_FENCE.sub(" ", raw)
    raw = _MD_INLINE.sub(r"\2", raw)
    raw = _MD_HEADING.sub("", raw)
    raw = _MD_LINK.sub(r"\1", raw)
    parts = re.split(r"\n{3,}", raw)
    return [(i, _clean(p)) for i, p in enumerate(parts, start=1) if p.strip()]


def _extract_docx(path: Path) -> list[tuple[int, str]]:
    """Read a .docx through python-docx.

    Word documents don't have page breaks we can reliably detect (they're
    rendered by the word processor, not stored); we emit a "page" every ~40
    paragraphs so retrieval metadata still carries something useful.
    """
    from docx import Document  # deferred import: only needed at ingest time

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Include table cells — lecture handouts frequently put definitions and
    # comparisons in tables, and losing them silently was a common complaint.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text:
                    paragraphs.append(text)
    if not paragraphs:
        return []
    pages: list[tuple[int, str]] = []
    page_size = 40
    for idx in range(0, len(paragraphs), page_size):
        block = "\n".join(paragraphs[idx : idx + page_size])
        cleaned = _clean(block)
        if cleaned:
            pages.append((idx // page_size + 1, cleaned))
    return pages


# The order matters: PDF first (has real page numbers), then richer
# text-only formats. When the same stem exists in multiple formats we keep
# only the richest one (PDF > DOCX > MD > TXT).
_SUPPORTED_EXTS = (".pdf", ".docx", ".md", ".txt")
_EXT_PRIORITY = {ext: i for i, ext in enumerate(_SUPPORTED_EXTS)}


def _gather_books(subject_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in _SUPPORTED_EXTS:
        files.extend(sorted(subject_dir.glob(f"*{ext}")))
    # When the same stem appears under several extensions, keep only the
    # highest-priority one. Prevents indexing both «chapter.pdf» and a
    # manually-exported «chapter.txt» that would double-count every chunk.
    by_stem: dict[str, Path] = {}
    for f in files:
        suffix = f.suffix.lower()
        if suffix not in _EXT_PRIORITY:
            continue
        existing = by_stem.get(f.stem)
        if existing is None or _EXT_PRIORITY[suffix] < _EXT_PRIORITY[existing.suffix.lower()]:
            by_stem[f.stem] = f
    return sorted(by_stem.values())


def _extract_book(path: Path) -> list[tuple[int, str]]:
    """Dispatch on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".md":
        return _extract_md(path)
    return _extract_txt(path)


# >30% garbage chunks in a single book is the reject threshold: above that
# line the LLM spends more time inventing plausible-sounding authors out of
# OCR'd letter soup than answering from real content. Admins can override
# with ``--force`` if they know a book is noisy but still useful.
_GARBAGE_REJECT_RATIO = 0.30


@dataclass
class BookReport:
    """One entry in the ingest summary shown to the admin."""

    subject_slug: str
    book: str
    total_chunks: int
    kept_chunks: int
    rejected: bool
    garbage_pages: list[int] = field(default_factory=list)

    @property
    def garbage_ratio(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.total_chunks - self.kept_chunks) / self.total_chunks


def _build_chunks_for_subject(
    subject: Subject, *, force: bool = False
) -> tuple[list[Chunk], list[BookReport]]:
    subject_dir = BOOKS_DIR / subject.slug
    reports: list[BookReport] = []
    if not subject_dir.is_dir():
        print(f"  [skip] нет папки {subject_dir}")
        return [], reports

    books = _gather_books(subject_dir)
    if not books:
        print(f"  [skip] в {subject_dir} нет .pdf/.txt")
        return [], reports

    all_chunks: list[Chunk] = []
    for book_path in books:
        print(f"  — {book_path.name}")
        pages = _extract_book(book_path)
        chunks = split_pages_into_chunks(
            pages,
            subject_slug=subject.slug,
            book=book_path.name,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        # OCR quality gate: drop garbage chunks per-book; if too many pages
        # of a book are trashed, refuse the whole book unless --force.
        kept: list[Chunk] = []
        garbage_pages: set[int] = set()
        for ch in chunks:
            if is_ocr_garbage_auto(ch.text):
                garbage_pages.add(ch.page)
            else:
                kept.append(ch)
        garbage_ratio = (len(chunks) - len(kept)) / max(len(chunks), 1)
        rejected = False
        if garbage_ratio > _GARBAGE_REJECT_RATIO and not force:
            rejected = True
            print(
                f"      [REJECT] {len(chunks) - len(kept)}/{len(chunks)} чанков "
                f"({garbage_ratio:.0%}) — OCR-мусор. Книга не индексируется. "
                "Пересними скан лучше или запусти с --force."
            )
        else:
            all_chunks.extend(kept)
            print(
                f"      страниц: {len(pages)}, чанков: {len(chunks)}"
                + (f" (OCR-мусор отброшен: {len(chunks) - len(kept)})"
                   if len(kept) < len(chunks) else "")
            )
        reports.append(
            BookReport(
                subject_slug=subject.slug,
                book=book_path.name,
                total_chunks=len(chunks),
                kept_chunks=0 if rejected else len(kept),
                rejected=rejected,
                garbage_pages=sorted(garbage_pages),
            )
        )
    return all_chunks, reports


def _print_quality_report(reports: list[BookReport]) -> None:
    """Admin-facing summary printed at the end of the ingest run."""
    if not reports:
        return
    print("\n=== OCR quality report ===")
    print(f"  {'SUBJECT':<20}  {'BOOK':<40}  TOTAL  KEPT  GARBAGE%  STATUS")
    for r in reports:
        status = "REJECTED" if r.rejected else ("OK" if r.garbage_ratio == 0 else "noisy")
        book_short = (r.book[:37] + "…") if len(r.book) > 40 else r.book
        print(
            f"  {r.subject_slug:<20}  {book_short:<40}  "
            f"{r.total_chunks:>5}  {r.kept_chunks:>4}  "
            f"{r.garbage_ratio * 100:>7.1f}%  {status}"
        )
        if r.rejected and r.garbage_pages:
            pages_preview = ", ".join(str(p) for p in r.garbage_pages[:10])
            if len(r.garbage_pages) > 10:
                pages_preview += f", … (+{len(r.garbage_pages) - 10})"
            print(f"      плохие страницы: {pages_preview}")


def _encode(model: SentenceTransformer, texts: list[str], batch: int = 16) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch), desc="Эмбеддинги", unit="batch"):
        emb = model.encode(
            texts[i : i + batch],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out.append(emb.astype(np.float32))
    return np.vstack(out)


def _load_existing_chunks() -> list[dict[str, object]]:
    if not CHUNKS_FILE.exists():
        return []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_chunks(chunks: list[dict[str, object]]) -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def build_index(only_subject: str | None = None, *, force: bool = False) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(COURSES_YAML)
    active = catalog.active()
    print(f"Предметов в каталоге: {len(active)}")

    if only_subject:
        if only_subject not in catalog:
            print(f"Неизвестный slug: {only_subject}", file=sys.stderr)
            sys.exit(1)
        subjects_to_process = [catalog.require(only_subject)]
    else:
        subjects_to_process = active

    # Gather chunks (with per-book OCR quality gate)
    new_chunks: list[Chunk] = []
    all_reports: list[BookReport] = []
    for subj in subjects_to_process:
        print(f"\n=== {subj.slug} ({subj.title_ru}) ===")
        chunks, reports = _build_chunks_for_subject(subj, force=force)
        new_chunks.extend(chunks)
        all_reports.extend(reports)

    if not new_chunks and not only_subject:
        _print_quality_report(all_reports)
        print(
            "\nНет материалов для индексации. Положите PDF/TXT в data/books/<slug>/"
            + (" или запустите с --force, если считаете OCR-шум приемлемым."
               if any(r.rejected for r in all_reports) else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    # Encode embeddings
    print(f"\nЗагружаю модель эмбеддингов: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    if only_subject:
        # partial rebuild: keep chunks from other subjects + their embeddings
        old_chunks = _load_existing_chunks()
        if EMBEDDINGS_FILE.exists() and old_chunks:
            old_matrix = np.load(EMBEDDINGS_FILE)
            keep_mask = np.array([c["subject_slug"] != only_subject for c in old_chunks])
            kept_chunks = [c for c, keep in zip(old_chunks, keep_mask, strict=False) if keep]
            kept_matrix = old_matrix[keep_mask]
            print(f"Сохраняю {len(kept_chunks)} чанков других предметов")
        else:
            kept_chunks, kept_matrix = [], np.zeros((0, 0), dtype=np.float32)

        new_texts = [c.text for c in new_chunks]
        new_matrix = _encode(model, new_texts) if new_texts else np.zeros((0, 0), dtype=np.float32)

        if kept_matrix.size and new_matrix.size:
            assert kept_matrix.shape[1] == new_matrix.shape[1], "embedding dim mismatch"
            matrix = np.vstack([kept_matrix, new_matrix])
        elif kept_matrix.size:
            matrix = kept_matrix
        else:
            matrix = new_matrix

        combined_chunks = kept_chunks + [
            {
                "text": c.text,
                "subject_slug": c.subject_slug,
                "book": c.book,
                "page": c.page,
                "chunk_idx": c.chunk_idx,
            }
            for c in new_chunks
        ]
    else:
        # full rebuild
        texts = [c.text for c in new_chunks]
        matrix = _encode(model, texts)
        combined_chunks = [
            {
                "text": c.text,
                "subject_slug": c.subject_slug,
                "book": c.book,
                "page": c.page,
                "chunk_idx": c.chunk_idx,
            }
            for c in new_chunks
        ]

    np.save(EMBEDDINGS_FILE, matrix)
    _write_chunks(combined_chunks)

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "rows": int(matrix.shape[0]) if matrix.size else 0,
        "dim": int(matrix.shape[1]) if matrix.size else 0,
        "subjects": sorted({str(c["subject_slug"]) for c in combined_chunks}),
        "built_at": int(time.time()),
    }
    INDEX_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Индекс собран: {matrix.shape[0]} чанков, dim={matrix.shape[1]}")
    print(f"   {EMBEDDINGS_FILE}")
    print(f"   {CHUNKS_FILE}")
    print(f"   {INDEX_META_FILE}")
    _print_quality_report(all_reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка глобального RAG-индекса")
    parser.add_argument(
        "--subject",
        dest="subject",
        help="slug предмета для точечной переиндексации (по умолчанию — все)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="индексировать даже книги, где >30%% чанков помечены как OCR-мусор",
    )
    args = parser.parse_args()
    build_index(only_subject=args.subject, force=args.force)


if __name__ == "__main__":
    main()

"""Parse teacher-uploaded glossary files (CSV / YAML) into DB rows.

Accepted formats:

**CSV** (``,`` or ``;`` separator, optional header):
    term,definition
    Стейкхолдер,Заинтересованное лицо, способное повлиять на организацию.

**YAML**:
    terms:
      - term: Стейкхолдер
        definition: Заинтересованное лицо...
      - term: Эмерджентность
        definition: Свойство системы...

Every upload for a subject **replaces** that subject's existing glossary
(not an append) so teachers can iterate on the source file without
accumulating stale rows. If a teacher wants cumulative edits they can keep
their source of truth in git and re-upload.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.services.safe_html import safe_html
from src.db.models import GlossaryTerm, Subject

_GLOSS_PREFIX_RU = "📖 <b>{term}</b>\n"
_GLOSS_PREFIX_EN = "📖 <b>{term}</b>\n"


def format_glossary_body(
    entry: GlossaryTerm, lang: str, question: str | None = None
) -> str:
    """Render a teacher-uploaded glossary entry for Telegram HTML.

    Term and definition are sanitised with the same whitelist as LLM
    output — a malformed `<b>` in the uploaded YAML/CSV can't poison
    the student message body.

    ``question`` is shown as a ``<blockquote>`` above the definition so
    the short-circuit keeps the same "question + answer" layout as full
    RAG responses.
    """
    import html as _html_mod

    prefix = _GLOSS_PREFIX_EN if lang == "en" else _GLOSS_PREFIX_RU
    q_block = (
        f"<blockquote>{_html_mod.escape(question.strip())}</blockquote>\n\n"
        if question
        else ""
    )
    return (
        q_block
        + prefix.format(term=safe_html(entry.term))
        + safe_html(entry.definition)
    )


@dataclass
class GlossaryUploadResult:
    inserted: int
    replaced_previous: int
    skipped_empty: int


def parse_glossary_payload(payload: bytes, filename: str) -> list[tuple[str, str]]:
    """Decode the uploaded bytes and return ``[(term, definition), …]``.

    Raises ``ValueError`` with a human-readable message when the input is
    malformed — the handler catches it and shows the message to the teacher.
    """
    try:
        text = payload.decode("utf-8-sig")  # utf-8-sig swallows a BOM if present
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")
    name = filename.lower()

    if name.endswith((".yaml", ".yml")):
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML не распарсился: {exc}") from exc
        if not isinstance(data, dict) or "terms" not in data:
            raise ValueError("В YAML должен быть ключ верхнего уровня 'terms'.")
        rows: list[tuple[str, str]] = []
        for entry in data.get("terms") or []:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term", "")).strip()
            defn = str(entry.get("definition", "")).strip()
            if term and defn:
                rows.append((term, defn))
        return rows

    if name.endswith(".csv"):
        # Sniff the delimiter; default to ',' if the sample is too small.
        sample = text[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text), dialect)
        rows = []
        for i, raw in enumerate(reader):
            if len(raw) < 2:
                continue
            term = raw[0].strip()
            defn = raw[1].strip()
            # Detect a header row on the first iteration (case-insensitive).
            if i == 0 and term.lower() in {"term", "термин"} and defn.lower() in {
                "definition",
                "определение",
            }:
                continue
            if term and defn:
                rows.append((term, defn))
        return rows

    raise ValueError("Поддерживаются только .csv / .yaml / .yml.")


async def replace_glossary(
    session: AsyncSession,
    *,
    subject: Subject,
    entries: list[tuple[str, str]],
) -> GlossaryUploadResult:
    """Replace all glossary rows of ``subject`` with ``entries``.

    Runs inside a nested SAVEPOINT so a mid-loop failure (e.g. DB
    constraint violation on a specific entry) rolls the whole replace
    back — the teacher never ends up with a half-populated glossary
    where the first 40 terms are the new ones and the old 60 are gone.
    Skipped: rows where term or definition is empty after stripping.
    """
    # Count what we're replacing so the teacher sees a diff-style summary.
    previous_count = (
        await session.execute(
            select(GlossaryTerm).where(GlossaryTerm.subject_id == subject.id)
        )
    ).all()

    async with session.begin_nested():
        await session.execute(
            delete(GlossaryTerm).where(GlossaryTerm.subject_id == subject.id)
        )

        inserted = 0
        skipped = 0
        seen: set[str] = set()
        for term, defn in entries:
            t = term.strip()
            d = defn.strip()
            if not t or not d:
                skipped += 1
                continue
            # De-dup inside the upload — keep the first occurrence,
            # matches YAML/CSV reading order.
            if t.lower() in seen:
                skipped += 1
                continue
            seen.add(t.lower())
            session.add(
                GlossaryTerm(
                    subject_id=subject.id,
                    term=t,
                    definition=d,
                )
            )
            inserted += 1
        await session.flush()
    return GlossaryUploadResult(
        inserted=inserted,
        replaced_previous=len(previous_count),
        skipped_empty=skipped,
    )


async def lookup_term(
    session: AsyncSession,
    *,
    question: str,
    subject_id: int | None,
) -> GlossaryTerm | None:
    """Exact-term lookup: return a DB glossary entry if the question is a
    known term (or «что такое <term>»). Used as a fast-path before RAG so
    the teacher's short definition beats a full LLM answer when it fits.

    SQLite's ``ilike`` is ASCII-only («Стейкхолдер» vs «СТЕЙКХОЛДЕР»
    doesn't match), so we do the case-fold comparison in Python. We
    narrow the SQL SELECT by subject to keep the row scan bounded to one
    course's glossary instead of loading every term in the database — on
    an inst-wide corpus that's the difference between <1 ms and seconds.
    """
    from src.bot.services.teacher_stats import _normalise_question

    q_norm = _normalise_question(question)
    if not q_norm:
        return None
    stmt = select(GlossaryTerm)
    if subject_id is not None:
        stmt = stmt.where(GlossaryTerm.subject_id == subject_id)
    rows = list((await session.execute(stmt)).scalars())
    for r in rows:
        if r.term.lower() == q_norm:
            return r
    return None

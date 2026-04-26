"""Парсер преподавательского глоссария (CSV / YAML) → строки БД.

Поддерживаемые форматы:

**CSV** (разделитель ``,`` или ``;``, заголовок опционально):
    term,definition
    Стейкхолдер,Заинтересованное лицо, способное повлиять на организацию.

**YAML**:
    terms:
      - term: Стейкхолдер
        definition: Заинтересованное лицо...
      - term: Эмерджентность
        definition: Свойство системы...

Каждая загрузка для предмета **заменяет** существующий глоссарий (не
дозаписывает) — преподаватель может итерировать source-файл, не накапливая
stale-строк. Если нужны кумулятивные правки — храните source-of-truth
в git и переcкачивайте.
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
    """Отрендерить glossary-запись от препода в Telegram-HTML.

    Term и definition санитизируются тем же whitelist'ом, что и LLM-вывод —
    кривой `<b>` из YAML/CSV не отравит тело сообщения студента.

    ``question`` — в ``<blockquote>`` сверху, чтобы short-circuit имел тот
    же «вопрос + ответ»-layout, что и полный RAG.
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
    """Декодировать загруженные байты и вернуть ``[(term, definition), …]``.

    Бросает ``ValueError`` с человекочитаемым сообщением, когда вход кривой —
    handler ловит и показывает текст преподавателю.
    """
    try:
        text = payload.decode("utf-8-sig")  # utf-8-sig съедает BOM если есть
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
        # Определяем разделитель; default ',' если sample мал.
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
            # На первой итерации детектим header-строку (case-insensitive).
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
    """Заменить все glossary-строки ``subject`` на ``entries``.

    Идёт под nested SAVEPOINT — mid-loop-сбой (например, DB-constraint
    на конкретной записи) откатывает весь replace; преподаватель никогда
    не получит полупустой глоссарий, где первые 40 терминов — новые, а
    предыдущие 60 уже стёрты. Skipped: строки с пустым term/definition.
    """
    # Считаем, что заменяем — чтобы препод увидел diff-style summary.
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
            # Dedup внутри upload'а — оставляем первое вхождение,
            # совпадает с порядком чтения YAML/CSV.
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
    """Точный lookup термина: вернуть строку глоссария, если вопрос — известный
    термин (или «что такое <term>»). Используется как fast-path перед RAG —
    короткое определение препода бьёт полный LLM-ответ, когда подходит.

    SQLite ``ilike`` ASCII-only («Стейкхолдер» vs «СТЕЙКХОЛДЕР» не матчит),
    case-fold-сравнение делаем в Python. Скоупим SQL-SELECT по предмету,
    чтобы row-scan ограничивался одним курсом — на корпусе всего института
    это разница между <1 мс и секундами.
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

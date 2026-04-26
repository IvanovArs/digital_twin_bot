"""Strip ungrounded attributions from the LLM answer.

Targets the three things Qwen3 hallucinates most: author brackets,
foreign etymologies, CJK/Arabic glosses. Match by shape, keep only if
the key tokens appear verbatim in the retrieved chunks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


# ---------- patterns ----------

# Attribution bracket — two variants accepted:
#   A) initials + surname [+ year]  → «(И. Фамилия, 1984)», «(T. Gruber, 1993)»
#   B) surname + year               → «(Freeman, 1984)», «(Фримен, 1984)»
# We capture the whole bracket; leading whitespace is swallowed so removal
# doesn't leave a double space behind.
_ATTRIBUTION_RE = re.compile(
    r"\s?\(\s*("
    r"(?:[A-ZА-ЯЁ]\.\s*){1,3}[A-ZА-ЯЁ][\wа-яё\-]+(?:\s*,\s*\d{4})?"
    r"|"
    r"[A-ZА-ЯЁ][\wа-яё\-]{2,}\s*,\s*\d{4}"
    r")\s*\)",
)

# Скобка этимологии. Принимаем частые RU/EN-префиксы из учебников —
# «(лат. X — Y)», «(греч. X)», «(Eng. X)», «(Lat. X)», …
_ETYMOLOGY_LANG_ALT = (
    "англ|лат|греч|нем|фр|ит|исп|яп|кит|араб"
    "|Eng|Lat|Gr|Fr|Ger|It|Sp|Jp|Ch|Ar"
)
_ETYMOLOGY_RE = re.compile(
    r"\s?\(\s*(?:" + _ETYMOLOGY_LANG_ALT + r")\.\s+[^)]+\)",
    flags=re.IGNORECASE,
)

# Одиночные блоки CJK / Arabic / Hebrew / Devanagari — никогда легитимно
# не встречаются в русском CS-учебнике, всегда сигнал «модель украсила
# термин скриптом, которого не знает».
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u4e00-\u9fff"  # CJK Unified
    r"\u3040-\u309f\u30a0-\u30ff"  # Hiragana + Katakana
    r"\uac00-\ud7af"  # Hangul
    r"\u0600-\u06ff"  # Arabic
    r"\u0590-\u05ff"  # Hebrew
    r"\u0900-\u097f"  # Devanagari
    r"]+"
)

# Голые годы в прозе ответа. `_ATTRIBUTION_RE` ловит `(Surname, 1984)`,
# но модель ещё любит free-form: «был введён в 1968 году». Если года в
# корпусе нет — это выдумка, гасим. Word-boundary'ы — чтобы «1984»
# внутри «ISO 19841» не матчилось.
_PROSE_YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d{2}|20[0-2]\d)(?!\d)")

# Диапазоны дат и десятилетий — ещё одна частая форма галлюцинации,
# которую bare-year-pass пропускает. Матчим «в 1970-1980-х», «в 1990-х»,
# «с середины 90-х», «1960-е—1980-е», «in the 1970s». Считаем диапазон
# заземлённым, если ОБА endpoint'а (или 4-значный якорь декады) есть в корпусе.
_DECADE_RE = re.compile(
    r"(?<!\d)("
    r"1[5-9]\d{2}[\-–—]1[5-9]\d{2}(?:[\-–—]?[ехe]?)?"  # 1970-1980 / 1970-1980-е
    r"|20[0-2]\d[\-–—]20[0-2]\d(?:[\-–—]?[ехe]?)?"      # 2000-2010-е
    r"|1[5-9]\d0[\-–—]?[ехe]"                           # 1970-е
    r"|20[0-2]0[\-–—]?[ехe]"                            # 2000-е
    r"|19\d0s|20[0-2]0s"                                # 1970s / 2000s
    r")(?!\d)",
)


# ---------- helpers ----------


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass
class ValidationReport:
    attributions_stripped: list[str]
    etymologies_stripped: list[str]
    foreign_scripts_stripped: list[str]
    years_stripped: list[str]
    decades_stripped: list[str]

    @property
    def total(self) -> int:
        return (
            len(self.attributions_stripped)
            + len(self.etymologies_stripped)
            + len(self.foreign_scripts_stripped)
            + len(self.years_stripped)
            + len(self.decades_stripped)
        )


# ---------- per-category checks ----------


def _attribution_grounded(bracket: str, corpus: str) -> bool:
    """True если фамилия И (если есть) год оба встречаются в ``corpus``.

    Фамилия матчится как **префикс слова** (``(?<!\\w)фриман`` ловит «Фриману»
    в дательном, но не срабатывает на не относящееся слово вроде
    «Бергаланфи»). Год сравнивается verbatim. Оба условия обязательны —
    фамилия без года или год без фамилии — это и есть то, что не должно проходить.
    """
    surname_m = re.search(r"[A-ZА-ЯЁ][\wа-яё\-]{2,}", bracket)
    if surname_m is None:
        return False  # странная форма — лучше дропнуть, чем оставить
    surname = surname_m.group(0).lower()
    # Word-boundary on the left, any suffix on the right — tolerates Russian
    # case inflection while still rejecting wholly different surnames.
    if not re.search(rf"(?<!\w){re.escape(surname)}", corpus):
        return False
    year_m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", bracket)
    return not (year_m and year_m.group(0) not in corpus)


def _etymology_grounded(bracket: str, corpus: str) -> bool:
    """True if the foreign word cited in the bracket appears in the corpus.

    «(лат. emergere — выныривать)» → checks ``emergere``. The Russian gloss
    after the dash is free-form and not worth matching.
    """
    # Capture the foreign word immediately after «<lang>.»
    m = re.search(
        r"(?:" + _ETYMOLOGY_LANG_ALT + r")\.\s+([\wа-яё\-]+)",
        bracket,
        flags=re.IGNORECASE,
    )
    if m is None:
        return False
    word = m.group(1).lower()
    return word in corpus


def _foreign_grounded(run: str, corpus: str) -> bool:
    """True if the exact non-Latin/Cyrillic run appears in the corpus."""
    return run in corpus


# ---------- public API ----------


def validate_answer(
    answer: str, corpus_texts: Iterable[str]
) -> tuple[str, ValidationReport]:
    """Remove hallucinated attribution / etymology / foreign-script runs.

    ``corpus_texts`` is the raw text of the grounded sources — textbook
    chunks for the RAG path, web snippets for the web-fallback path. Any
    candidate bracket whose key tokens aren't present in the corpus is
    stripped. Returns the cleaned answer plus a report listing what was
    stripped (so callers can log it). Formatting, HTML tags, and
    legitimately-grounded attributions all survive verbatim.
    """
    corpus_raw = " ".join(corpus_texts)
    corpus_cf = _normalise(corpus_raw)
    report = ValidationReport([], [], [], [], [])

    def _on_attr(m: re.Match[str]) -> str:
        if _attribution_grounded(m.group(1), corpus_cf):
            return m.group(0)
        report.attributions_stripped.append(m.group(0).strip())
        return ""

    def _on_etym(m: re.Match[str]) -> str:
        if _etymology_grounded(m.group(0), corpus_cf):
            return m.group(0)
        report.etymologies_stripped.append(m.group(0).strip())
        return ""

    def _on_foreign(m: re.Match[str]) -> str:
        if _foreign_grounded(m.group(0), corpus_raw):
            return m.group(0)
        report.foreign_scripts_stripped.append(m.group(0))
        return ""

    answer = _ATTRIBUTION_RE.sub(_on_attr, answer)
    answer = _ETYMOLOGY_RE.sub(_on_etym, answer)
    answer = _FOREIGN_SCRIPT_RE.sub(_on_foreign, answer)

    # Strip bare ungrounded years in prose. Tricky: we can't blank a year
    # and leave grammatically-broken prose («был введён в  году»), so we
    # also eat the preposition + word that usually precedes it. Pattern:
    # «в 1984 году», «in 1984», «(1984)» — scrub the whole chunk if the
    # year isn't grounded.
    def _on_year(m: re.Match[str]) -> str:
        year = m.group(1)
        if year in corpus_raw:
            return m.group(0)
        report.years_stripped.append(year)
        return ""

    def _on_decade(m: re.Match[str]) -> str:
        """Strip decade/range expressions unless all year anchors are
        grounded in the corpus."""
        raw = m.group(0)
        years = re.findall(r"1[5-9]\d{2}|20[0-2]\d", raw)
        if years and all(y in corpus_raw for y in years):
            return raw
        if raw.lower() in corpus_raw.lower():
            return raw
        report.decades_stripped.append(raw)
        return ""

    # Decades must run BEFORE the bare-year pass: otherwise the bare-year
    # regex would scoop individual anchors out of a range («1970-1990» →
    # «-») and the decade pass would find no years to check.
    answer = _DECADE_RE.sub(_on_decade, answer)
    answer = _PROSE_YEAR_RE.sub(_on_year, answer)
    # «в  году» / «in  », etc. — if the year was stripped, strip the
    # scaffolding too. Heuristic but safe: requires that the year is gone.
    answer = re.sub(
        r"\s*(?:в|во|in|since|around)\s+году",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\(\s*,?\s*\)", "", answer)  # empty leftover parens

    # Clean up residue left by removed brackets: double spaces, orphan
    # punctuation like « ,» or « —», and isolated space-before-punctuation.
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"\(\s*[—–-]?\s*\)", "", answer)  # empty «(—)» husks
    return answer, report

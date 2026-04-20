"""Heuristic OCR-garbage filter for RAG chunks.

Tesseract on scanned textbooks sometimes emits chunks like «фформационный
noxxon», «Gow Бергаланфи, 'Tor термин», «„шт == some До ==YASS ownsere…».
Qwen3-4B reads such chunks and synthesises confident-sounding but fabricated
author attributions («G. Bernaldi, 1984» from garbled «Бергаланфи»). Cheaper
to drop the chunk than to scrub the answer.

Used both at ingest time (to keep trash out of the index) and in
``format_context`` (belt-and-suspenders for pre-existing indexes).
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_CYR = re.compile(r"[а-яА-ЯёЁ]")
_LAT = re.compile(r"[a-zA-Z]")
# ``STEP-анализ``, ``SWOT-анализ``, ``PESTELанализ`` — ALL-CAPS Latin acronym
# glued to a Russian suffix with or without a hyphen. This is *not* OCR
# damage (it's how these terms appear in Russian textbooks), so we exclude
# the pattern from the mixed-alphabet tally.
_ACRONYM_COMPOUND = re.compile(r"^[A-Z]{2,}-?[а-яА-ЯёЁ]+$")
# Legitimate single-letter tokens in RU/EN prose: vowels/pronouns («а»,
# «и», «я», «a», «I»), а также буквы-инициалы и части сокращений вроде
# «т. е.», «т. п.», «с. 12», «В. Г.». Anything outside this set of length 1
# is almost always OCR debris from a dropped ligature or a page header.
_OK_SINGLES = frozenset("аиоеяувскмгтпнрдлфзхчшщбАИОЕЯУВСКМГТПНРДЛФЗХЧШЩaIА")

# Thresholds are intentionally conservative: clean pages score near zero
# on every signal; Tesseract's worst pages clear these by 2-5×.
_MIXED_RATIO_MAX = 0.05
_STRAY_RATIO_MAX = 0.15
_LAT_IN_RU_MAX = 0.15
_LAT_ABSOLUTE_MAX = 0.50
_MIN_ALPHA_TOKENS = 12


def is_ocr_garbage(text: str, *, ru_dominant: bool = True) -> bool:
    """Return True if the chunk looks like OCR debris.

    ``ru_dominant`` defaults to True — the project currently only indexes
    Russian textbooks. When an English/Kazakh/etc. book is ingested, the
    caller can detect the language upstream and flip this flag so the
    "too much Latin" signal doesn't misfire on legitimate English prose.
    See ``is_ocr_garbage_auto`` for automatic dominant-language detection.
    """
    # Strip URLs first — a legitimate source link («https://elib.spbstu.ru/…»)
    # tokenises to 8-10 Latin pieces and would otherwise flip the "mostly
    # Russian page but too much Latin" heuristic.
    text = _URL_RE.sub(" ", text)
    words = _WORD_RE.findall(text)
    # Only alphabetic tokens count — pure-digit tokens («рис. 9», «8» from a
    # footnote) aren't a signal of OCR damage.
    alpha = [w for w in words if _CYR.search(w) or _LAT.search(w)]
    if len(alpha) < _MIN_ALPHA_TOKENS:
        return False  # too short to judge reliably — let it through
    cyr_only = lat_only = mixed = short_stray = 0
    for w in alpha:
        if _ACRONYM_COMPOUND.match(w):
            cyr_only += 1  # treat «STEP-анализ» as a Russian word
            continue
        has_c = bool(_CYR.search(w))
        has_l = bool(_LAT.search(w))
        if has_c and has_l:
            mixed += 1
        elif has_c:
            cyr_only += 1
        else:
            lat_only += 1
        if len(w) == 1 and w not in _OK_SINGLES:
            short_stray += 1
    total = len(alpha)
    # ≥5% mixed-alphabet tokens → tokeniser is confused, drop it.
    if mixed / total > _MIXED_RATIO_MAX:
        return True
    # Chunk is mostly one-letter debris.
    if short_stray / total > _STRAY_RATIO_MAX:
        return True
    if ru_dominant:
        # Predominantly-Russian page but too many stand-alone Latin words —
        # almost always transliteration errors, not bibliography entries.
        if cyr_only > lat_only and lat_only / total > _LAT_IN_RU_MAX:
            return True
        # For a Russian-only index: >50% pure Latin tokens is debris after
        # OCR lost a diagram or table.
        if lat_only / total > _LAT_ABSOLUTE_MAX:
            return True
    else:
        # Mirror: predominantly-Latin page with a lot of stray Cyrillic is
        # probably OCR confusion on an English textbook with Russian footnotes.
        if lat_only > cyr_only and cyr_only / total > _LAT_IN_RU_MAX:
            return True
        if cyr_only / total > _LAT_ABSOLUTE_MAX:
            return True
    return False


def detect_ru_dominant(text: str) -> bool:
    """Cheap heuristic: True if the text is predominantly Cyrillic.

    Falls back to True (RU) on ties or when the sample has no letters — our
    index is RU-only by default, so that's the safer assumption.
    """
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    if cyr == 0 and lat == 0:
        return True
    return cyr >= lat


def is_ocr_garbage_auto(text: str) -> bool:
    """Auto-detect the dominant script and apply the appropriate rules."""
    return is_ocr_garbage(text, ru_dominant=detect_ru_dominant(text))

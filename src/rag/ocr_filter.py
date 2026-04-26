"""Эвристический фильтр OCR-мусора для RAG-чанков.

Tesseract на сканированных учебниках иногда выдаёт чанки вроде «фформационный
noxxon», «Gow Бергаланфи, 'Tor термин», «„шт == some До ==YASS ownsere…».
Qwen3-4B читает такие чанки и синтезирует уверенно-выглядящие, но выдуманные
авторские атрибуции («G. Bernaldi, 1984» из изуродованного «Бергаланфи»).
Дешевле дропнуть чанк, чем чистить ответ.

Используется и на ingest (чтобы мусор не попал в индекс), и в
``format_context`` (страховка для уже существующих индексов).
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_CYR = re.compile(r"[а-яА-ЯёЁ]")
_LAT = re.compile(r"[a-zA-Z]")
# ``STEP-анализ``, ``SWOT-анализ``, ``PESTELанализ`` — ALL-CAPS Latin акроним,
# приклеенный к русскому суффиксу с дефисом или без. Это *не* OCR-повреждение
# (так эти термины и появляются в русских учебниках), исключаем из
# mixed-alphabet-подсчёта.
_ACRONYM_COMPOUND = re.compile(r"^[A-Z]{2,}-?[а-яА-ЯёЁ]+$")
# Легитимные однобуквенные токены в RU/EN-прозе: гласные/местоимения
# («а», «и», «я», «a», «I»), а также буквы-инициалы и части сокращений
# вроде «т. е.», «т. п.», «с. 12», «В. Г.». Всё, что вне этого set'а
# длиной 1, — почти всегда OCR-debris от потерянной лигатуры или page-header'а.
_OK_SINGLES = frozenset("аиоеяувскмгтпнрдлфзхчшщбАИОЕЯУВСКМГТПНРДЛФЗХЧШЩaIА")

# Пороги намеренно консервативные: чистые страницы дают почти ноль по
# каждому сигналу; худшие страницы Tesseract'а пробивают их в 2–5×.
_MIXED_RATIO_MAX = 0.05
_STRAY_RATIO_MAX = 0.15
_LAT_IN_RU_MAX = 0.15
_LAT_ABSOLUTE_MAX = 0.50
_MIN_ALPHA_TOKENS = 12


def is_ocr_garbage(text: str, *, ru_dominant: bool = True) -> bool:
    """True если чанк похож на OCR-debris.

    ``ru_dominant`` по умолчанию True — проект сейчас индексирует только
    русские учебники. Когда грузится английская/казахская/др. книга,
    caller может задетектить язык выше по стеку и развернуть флаг —
    сигнал «слишком много латиницы» не сработает на легитимной EN-прозе.
    См. ``is_ocr_garbage_auto`` для авто-детекта доминирующего языка.
    """
    # Сначала режем URL'ы — легитимная ссылка («https://elib.spbstu.ru/…»)
    # токенизуется в 8–10 латинских кусков и развернула бы эвристику
    # «в основном русская страница, но слишком много латиницы».
    text = _URL_RE.sub(" ", text)
    words = _WORD_RE.findall(text)
    # Считаем только алфавитные токены — pure-digit-токены («рис. 9», «8» из
    # сноски) не сигнал OCR-повреждения.
    alpha = [w for w in words if _CYR.search(w) or _LAT.search(w)]
    if len(alpha) < _MIN_ALPHA_TOKENS:
        return False  # слишком коротко для надёжного суждения — пропускаем
    cyr_only = lat_only = mixed = short_stray = 0
    for w in alpha:
        if _ACRONYM_COMPOUND.match(w):
            cyr_only += 1  # «STEP-анализ» считаем русским словом
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

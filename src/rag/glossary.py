"""Расширение запроса через глоссарий.

Bi-encoder'ы штрафуют короткие запросы: cosine между 3-токеновым «что такое X»
и 200-токеновым чанком учебника структурно низок — каждое off-topic-предложение
в чанке тянет mean-embedding в сторону. Обходим это, прикрепляя к запросу
определение из глоссария для термина, который узнали в запросе — expanded-query
делит 10+ контентных токенов с любым пассажем, определяющим этот термин;
top_score растёт на ~0.10–0.20 на definitional-вопросах вроде «что такое стейкхолдер».

Никаких новых зависимостей — только YAML-файлы из ``data/glossary/``.
"""

from __future__ import annotations

import re
from functools import lru_cache

import yaml

from src.rag.config import ROOT

GLOSSARY_DIR = ROOT / "data" / "glossary"
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{2,}")
# Грубый русский стеммер — режет частые case/number-окончания. Покрывает
# «стейкхолдера» / «стейкхолдеры» / «надсистемой» / «целей» против их
# лемм, не таща pymorphy2. Порядок важен: длинные multi-char-окончания первыми.
_RU_ENDING_RE = re.compile(
    r"(ами|ями|ого|ому|ыми|ыми|ого|их|ом|ой|ою|ую|ев|ев|ов|ах|ях|ой|ем|ой|ьми|ам|ям|"
    r"ой|ии|ью|ие|ия|ия|ие|ой|у|ю|ы|и|а|я|е|ь|й)$"
)
# Жёсткий лимит расширений на запрос — выше промпт раздувается, а
# query-embedding дрейфует к среднему из многих определений.
_MAX_EXPANSIONS = 2
# Не матчим стеммы короче — слишком много false-positive'ов.
_MIN_STEM_LEN = 4


@lru_cache(maxsize=1)
def _glossary_index() -> dict[str, list[str]]:
    """Карта ``content_word_lower → [definition, …]`` по всем glossary-YAML'ам.

    Индексирует **каждое** content-слово в многословных терминах (не только
    head). «Методика Кошарского-Уёмова» и «Методика Волковой-Четверикова»
    обе ложатся под «методика» — оставляем оба определения, упорядоченные
    по index-order, чтобы ни одно не заслонило другое.
    """
    out: dict[str, list[str]] = {}
    if not GLOSSARY_DIR.exists():
        return out
    for path in sorted(GLOSSARY_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for entry in data.get("terms", []) or []:
            term = str(entry.get("term", "")).strip()
            defn = str(entry.get("definition", "")).strip()
            if not term or not defn:
                continue
            term_l = term.lower()
            # Полный термин — самый сильный матч, регистрируем первым.
            out.setdefault(term_l, []).append(defn)
            # Плюс каждое content-слово внутри термина — «культура» находит
            # «Корпоративная культура», «инициирования» — «Пространство
            # инициирования целей» и т.п.
            for word in _WORD_RE.findall(term_l):
                if len(word) >= _MIN_STEM_LEN and word != term_l:
                    bucket = out.setdefault(word, [])
                    if defn not in bucket:
                        bucket.append(defn)
    return out


def _stem(word: str) -> str:
    return _RU_ENDING_RE.sub("", word.lower())


def expand_query(query: str) -> str:
    """Прицепить определения любых glossary-терминов, найденных в ``query``.

    Возвращает исходный запрос как есть, если матчей нет. Расширения
    дедуплицируются и обрезаются до ``_MAX_EXPANSIONS`` — embedding
    остаётся сфокусированным.
    """
    index = _glossary_index()
    if not index:
        return query

    seen: set[str] = set()
    extras: list[str] = []
    for token in _WORD_RE.findall(query):
        stem = _stem(token)
        if len(stem) < _MIN_STEM_LEN:
            continue
        for key, defns in index.items():
            if key.startswith(stem) or stem.startswith(key):
                for defn in defns:
                    if defn not in seen:
                        extras.append(defn)
                        seen.add(defn)
                        if len(extras) >= _MAX_EXPANSIONS:
                            break
                break
        if len(extras) >= _MAX_EXPANSIONS:
            break

    if not extras:
        return query
    return query + " " + " ".join(extras)


__all__ = ["expand_query"]

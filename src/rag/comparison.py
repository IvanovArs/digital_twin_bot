"""Детект и обработка вопросов «сравни X и Y».

Паттерны, которые ловим в запросах юзера:

* «сравни X и Y», «сравните X с Y»
* «разница между X и Y», «чем отличается X от Y», «разлчия X и Y»
* EN: «compare X and Y», «difference between X and Y»,
  «how is X different from Y»

При детекте QA-пайплайн делает два параллельных retrieval'а (по одному
на термин) и мержит хиты перед вызовом LLM с comparison-специфичным
user-промптом.
"""

from __future__ import annotations

import re

from src.rag.prompts import apply_followup_modifier, build_system_prompt, format_context
from src.rag.retriever import Hit
from src.subjects import Subject

# Каждый паттерн должен захватывать две группы: term_a и term_b. Хвостовая
# пунктуация срезается в ``detect_comparison`` — термины остаются чистыми.
_PATTERNS = [
    # ru: сравни/сравните X и Y | X с Y
    re.compile(r"сравни(?:те)?\s+(.+?)\s+(?:и|с)\s+(.+)$", re.IGNORECASE),
    # ru: разница между X и Y
    re.compile(r"разниц[аы]\s+между\s+(.+?)\s+и\s+(.+)$", re.IGNORECASE),
    # ru: чем отличается X от Y | отличия X от Y
    re.compile(r"(?:чем\s+)?отличается\s+(.+?)\s+от\s+(.+)$", re.IGNORECASE),
    re.compile(r"отличия\s+(.+?)\s+от\s+(.+)$", re.IGNORECASE),
    # ru: различия X и Y
    re.compile(r"различия\s+(.+?)\s+и\s+(.+)$", re.IGNORECASE),
    # en: compare X and Y
    re.compile(r"compare\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+)$", re.IGNORECASE),
    # en: difference between X and Y
    re.compile(r"difference\s+between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE),
    # en: X vs Y / X versus Y
    re.compile(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+)$", re.IGNORECASE),
]

# Срезаем хвостовую пунктуацию и короткие noise-слова из захваченных
# терминов — «сравни стейкхолдера и акционера?» → «стейкхолдера», «акционера».
_TERM_TRIM = re.compile(r"[?.!,;:—\s]+$")
_LEAD_JUNK = re.compile(r"^(что\s+такое|что\s+есть|the|a|an)\s+", re.IGNORECASE)


def _clean_term(t: str) -> str:
    t = _TERM_TRIM.sub("", t).strip()
    t = _LEAD_JUNK.sub("", t).strip()
    return t


def detect_comparison(question: str) -> tuple[str, str] | None:
    """Вернуть ``(term_a, term_b)`` если ``question`` — comparison-запрос.

    Оба термина — непустые и ≥ 2 символов после cleanup'а; иначе матч на
    одну букву шума. Без матча — ``None``.
    """
    q = (question or "").strip().rstrip("?.!")
    if not q:
        return None
    for pat in _PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        a = _clean_term(m.group(1))
        b = _clean_term(m.group(2))
        if len(a) >= 2 and len(b) >= 2 and a.lower() != b.lower():
            return a, b
    return None


def merge_hits(hits_a: list[Hit], hits_b: list[Hit], max_total: int = 8) -> list[Hit]:
    """Чередуем два списка хитов, дропаем дубликаты по тексту, ограничиваем total.

    Чередование (а не конкатенация) сохраняет оба термина представленными даже
    при маленьком ``max_total``. Dedup по первым 200 символам ``hit.text`` —
    дёшево и достаточно, чтобы поймать один и тот же чанк в обоих retrieval'ах.
    """
    seen: set[str] = set()
    out: list[Hit] = []
    n = max(len(hits_a), len(hits_b))
    for i in range(n):
        for pool in (hits_a, hits_b):
            if i >= len(pool):
                continue
            h = pool[i]
            fingerprint = (h.text or "")[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            out.append(h)
            if len(out) >= max_total:
                return out
    return out


def build_comparison_messages(
    term_a: str,
    term_b: str,
    hits: list[Hit],
    subject: Subject | None,
    lang: str = "ru",
    *,
    modifier: str | None = None,
) -> list[dict[str, str]]:
    """Собрать side-by-side comparison-промпт. ``modifier`` идёт в хвост
    user-сообщения — «Проще/Пример/Подробнее» поверх comparison-ответа
    меняет тон, не выкидывая table-формат."""
    system = build_system_prompt(subject, lang=lang)
    context = format_context(hits)
    if lang == "en":
        user = (
            f"Textbook fragments:\n{context}\n\n"
            f"Compare «{term_a}» and «{term_b}» strictly from the fragments.\n\n"
            "Output structure:\n"
            f"1) One sentence: what {term_a} is.\n"
            f"2) One sentence: what {term_b} is.\n"
            "3) 3–5 bullets with concrete side-by-side differences. Each bullet:\n"
            f"   «<b>{term_a}</b> — … | <b>{term_b}</b> — …».\n"
            "4) One closing sentence on the key distinction.\n\n"
            "No 'the fragment says', no invented attributions. Only from "
            "the textbook fragments above.\n\nAnswer:"
        )
    else:
        user = (
            f"Фрагменты учебника:\n{context}\n\n"
            f"Сравни «{term_a}» и «{term_b}» строго по фрагментам.\n\n"
            "Структура ответа:\n"
            f"1) Одна фраза: что такое «{term_a}».\n"
            f"2) Одна фраза: что такое «{term_b}».\n"
            "3) 3–5 пунктов с конкретикой — построчное сравнение. Формат каждого пункта:\n"
            f"   «<b>{term_a}</b> — … | <b>{term_b}</b> — …».\n"
            "4) Одно предложение в конце — главное отличие.\n\n"
            "Без «в тексте», без выдуманных атрибуций. Только из приведённых "
            "фрагментов.\n\nОтвет:"
        )
    if modifier:
        user = apply_followup_modifier(user, modifier, lang)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

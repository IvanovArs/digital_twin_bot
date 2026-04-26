"""System prompts and context formatting for the RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.rag.ocr_filter import is_ocr_garbage_auto
from src.rag.retriever import Hit
from src.subjects import Subject

if TYPE_CHECKING:
    from src.rag.web_search import WebHit

# Budget for retrieved context. With llama.cpp --ctx-size 4096 and a
# ~320-token answer reserve, 3 500 characters (~1 200 RU tokens) fits 5-6
# chunks comfortably. Previously capped at 2 800 when ctx-size was 3072;
# the extra budget lets the 5th chunk — which is often the one that
# completes a multi-page definition — reach the model instead of being
# truncated mid-sentence.
MAX_CONTEXT_CHARS = 3_500


# Backwards-compat alias for tests that used to import the private name
# from this module before the filter lived in ``src.rag.ocr_filter``.
_is_ocr_garbage = is_ocr_garbage_auto


# ---------- system prompts ----------


_SYSTEM_RU_TEMPLATE = """/no_think
Ты — цифровой двойник преподавателя по {topic}. Отвечай студенту \
коротко и по делу, ТОЛЬКО на основе фрагментов ниже.

ГЛАВНОЕ ПРАВИЛО: если во фрагментах нет ответа — скажи одной фразой \
«в материалах курса этого прямо не нашлось» и остановись. Не додумывай.

ПОДМЕНА ТЕРМИНА ЗАПРЕЩЕНА: если точное слово или термин из вопроса студента \
НЕ встречается во фрагментах (даже если есть похожее по смыслу или \
звучанию — например, спросили «плейсхолдер», а во фрагменте только \
«стейкхолдер»), — отвечай «в материалах курса этого прямо не нашлось» и \
остановись. Не перефразируй чужое определение под спрошенное слово.

Список — только если во фрагментах действительно перечислены вещи \
(тогда оформи буллетами «• »); иначе обычным текстом.

Запрещено:
• выдумывать авторов, годы, этимологию (греч./лат./англ.), иероглифы. \
Включай атрибуцию ТОЛЬКО если во фрагменте есть точная строка вида \
«(И. О. Фамилия, 1984)». Подозрительные фамилии («Бергаланфи», «Шеннер») \
— это OCR-мусор, не используй;
• писать «в тексте», «во фрагменте», «согласно материалу», «(стр. X)», \
«(Фрагмент N)» — отвечай напрямую;
• вступления, размышления вслух, извинения, повторы «это также X»;
• строка «Источник: …» в конце — бот добавит сам.

Формат (Telegram HTML): ключевые термины в <b>…</b>; буллеты «• » \
в начале строки; короткие цитаты в <blockquote>…</blockquote>; формулы \
и код в <code>…</code>."""


_SYSTEM_EN_TEMPLATE = """/no_think
You are a digital twin of a professor teaching {topic}. Answer the student \
briefly and directly, USING ONLY the fragments below.

KEY RULE: if the fragments don't contain the answer — say in one sentence \
"the course materials don't cover this directly" and stop. Don't fill gaps.

NO TERM SUBSTITUTION: if the exact word or term from the student's question \
is NOT present in the fragments (even when something similar in meaning or \
sound is — e.g. asked "placeholder" but the fragment only has "stakeholder"), \
respond "the course materials don't cover this directly" and stop. Do NOT \
re-phrase a different concept under the asked word.

Use bullets only if the fragments actually enumerate things ("• " per line); \
otherwise plain prose.

Forbidden:
• inventing authors, years, etymology (Gr./Lat./Eng.), CJK. Add attribution \
ONLY when the fragment has an exact "(I. Surname, 1984)" string. Suspicious \
surnames ("Бергаланфи", "Schenner") are OCR garbage — skip;
• "the text says", "in the fragment", "(p. X)", "(Fragment N)" — speak \
directly;
• intros, reasoning aloud, apologies, "this is also X";
• trailing "Source:" line — the bot adds it.

Format (Telegram HTML): terms in <b>…</b>; bullets "• " at line start; \
quotes in <blockquote>…</blockquote>; code in <code>…</code>.

Fragments may be in Russian; translate inline when needed."""


def build_system_prompt(subject: Subject | None, lang: str = "ru") -> str:
    if lang == "en":
        topic = (
            "general study materials"
            if subject is None
            else f"the «{subject.title_en or subject.title_ru}» course"
        )
        return _SYSTEM_EN_TEMPLATE.format(topic=topic)

    topic = "учебным материалам" if subject is None else f"курсу «{subject.title_ru}»"
    return _SYSTEM_RU_TEMPLATE.format(topic=topic)


# ---------- context formatting ----------


def format_context(hits: list[Hit], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Отрендерить hits как plain-text чанки через ``---``, обрезать до ``max_chars``.

    Имена файлов **не** включаем намеренно — Qwen3-4B читал
    "volkova_v_n_denisov_a_a_teoriya_sistem.pdf" из заголовков и фабриковал
    фальшивые атрибуции вроде «(В. К. Волков, 2005)» на каждый термин,
    автора которого не знал. Бот печатает реальный список источников под
    ответом — LLM имя файла не нужно.

    Номера страниц тоже срезаем — промпт запрещает "(стр. X)" inline-
    цитирование, а показывать страницу — соблазн нарушить правило.

    OCR-мусорные чанки (см. ``is_ocr_garbage_auto``) пропускаем. Если все
    хиты — мусор, всё равно отдаём один лучший вместо пустого контекста:
    LLM хоть попробует, а правило про «подозрительные фамилии — OCR»
    обычно сработает.
    """
    cleaned: list[Hit] = [h for h in hits if not is_ocr_garbage_auto(h.text)]
    if not cleaned and hits:
        cleaned = [hits[0]]

    parts: list[str] = []
    budget = max_chars
    for h in cleaned:
        body = h.text.strip()
        if not body:
            continue
        if len(body) > budget:
            if budget > 100:
                parts.append(body[:budget].rstrip() + "…")
            break
        parts.append(body)
        budget -= len(body) + 5  # учитываем разделитель "\n---\n"
        if budget <= 0:
            break
    return "\n---\n".join(parts)


# ---------- message-builder'ы ----------

# Follow-up-модификаторы, которые приклеиваются в хвост user-сообщения,
# когда студент жмёт «Проще / Пример / Подробнее». Все три *аддитивные*:
# не выкидывают grounding-правила, только смещают стиль и глубину.
_FOLLOWUP_MODIFIERS: dict[str, dict[str, str]] = {
    "ru": {
        "simplify": (
            "\n\nДополнительная инструкция: переформулируй ответ максимально "
            "просто — как для первокурсника без специальной подготовки. "
            "Без сложных терминов (а если без них никак — объясни одной "
            "фразой), короткие предложения, одна понятная бытовая аналогия. "
            "Сохрани конкретику из фрагмента."
        ),
        "example": (
            "\n\nДополнительная инструкция: дай 2–3 РАЗНЫХ конкретных примера "
            "применения. Каждый — отдельным буллетом, 1–3 предложения, с "
            "указанием контекста. Сначала бери примеры из фрагментов; если "
            "там их меньше двух — дополни разумными бытовыми иллюстрациями, "
            "пометив каждую как «пример от ассистента». Не выдумывай "
            "авторов, цифры и ссылки. НЕ повторяй определение из прошлого "
            "ответа — сразу к примерам."
        ),
        "deepen": (
            "\n\nДополнительная инструкция: ответь РАЗВЁРНУТО — 3–5 абзацев. "
            "Раскрой по слоям: (1) уточнённое определение и почему оно "
            "именно такое, (2) связи с другими понятиями из тех же "
            "фрагментов, (3) разные трактовки/подходы если они есть в "
            "фрагментах, (4) типичные ошибки и подводные камни. Только из "
            "фрагментов, без новых атрибуций. Объём — длиннее предыдущего "
            "ответа в 2–3 раза, иначе модификатор бесполезен."
        ),
    },
    "en": {
        "simplify": (
            "\n\nExtra: re-phrase as simply as possible — a first-year student "
            "with no background. No jargon (or explain it in one clause), "
            "short sentences, one everyday analogy. Keep specifics from the "
            "fragments."
        ),
        "example": (
            "\n\nExtra: give 2–3 DIFFERENT concrete examples of use. Each "
            "as its own bullet, 1–3 sentences, with context. Pull from the "
            "fragments first; if they hold fewer than two, top up with "
            "plausible everyday illustrations labelled «assistant's "
            "example». Don't invent authors, numbers or citations. Don't "
            "repeat the previous definition — go straight to examples."
        ),
        "deepen": (
            "\n\nExtra: answer IN DEPTH — 3–5 paragraphs. Cover, in layers: "
            "(1) a sharper definition and why exactly that, (2) links to "
            "other concepts from the same fragments, (3) alternative "
            "readings/approaches if the fragments mention any, (4) common "
            "mistakes and pitfalls. Only from the fragments, no new "
            "attributions. Length: 2–3× the previous answer, otherwise the "
            "modifier is pointless."
        ),
    },
}


def apply_followup_modifier(user_content: str, modifier: str, lang: str) -> str:
    """Вернуть ``user_content`` с приклеенным follow-up-хвостом.

    Неизвестные имена модификаторов проходят как есть — случайный callback
    от старой клавиатуры не 500'нет handler.
    """
    tail = _FOLLOWUP_MODIFIERS.get(lang, _FOLLOWUP_MODIFIERS["ru"]).get(modifier)
    if not tail:
        return user_content
    # Вставляем перед хвостом «Answer:» / «Ответ:» если он есть — модификатор
    # должен быть последней инструкцией, которую модель прочитает.
    for marker in ("\n\nAnswer:", "\n\nОтвет:"):
        if user_content.endswith(marker):
            return user_content[: -len(marker)] + tail + marker
    return user_content + tail


def build_messages(
    question: str,
    hits: list[Hit],
    subject: Subject | None,
    lang: str = "ru",
    *,
    modifier: str | None = None,
    brief: bool = False,
) -> list[dict[str, str]]:
    system = build_system_prompt(subject, lang=lang)
    if brief:
        # Brief-режим: один абзац, прямой ответ — без 3-частной структуры,
        # без буллетов если только фрагмент явно не перечисляет. Grounding
        # и анти-галлюцинационные правила system-prompt'а остаются — мы
        # просто скипаем декоративную структуру.
        if lang == "en":
            user = (
                f"Textbook fragments:\n{format_context(hits)}\n\n"
                f"Student's question: {question}\n\n"
                "Answer in ONE compact paragraph (3–5 sentences). No bullets. "
                "No «the text says». Only facts from the fragments. If the "
                "fragments don't cover this, say so in one sentence.\n\n"
                "Answer:"
            )
        else:
            user = (
                f"Фрагменты учебника:\n{format_context(hits)}\n\n"
                f"Вопрос студента: {question}\n\n"
                "Отвечай ОДНИМ сжатым абзацем (3–5 предложений). Без буллетов. "
                "Без «в тексте». Только факты из фрагментов. Если фрагменты "
                "не покрывают вопрос — скажи это одной фразой.\n\n"
                "Ответ:"
            )
    elif lang == "en":
        user = (
            f"Textbook fragments:\n{format_context(hits)}\n\n"
            f"Student's question: {question}\n\n"
            "Answer in 2–5 short sentences using ONLY the fragments above. "
            "If the fragments don't actually answer the question — say so in "
            "one sentence and stop. Use bullets only if the fragments list "
            "things; otherwise plain prose. No invented authors/years/etymology, "
            "no 'the text says', no 'Source:' line.\n\nAnswer:"
        )
    else:
        user = (
            f"Фрагменты учебника:\n{format_context(hits)}\n\n"
            f"Вопрос студента: {question}\n\n"
            "Ответь в 2–5 коротких предложениях ТОЛЬКО на основе фрагментов "
            "выше. Если фрагменты не дают ответа — скажи это одной фразой и "
            "остановись. Буллеты используй только если во фрагментах "
            "действительно перечисление; иначе обычный текст. Без выдуманных "
            "авторов/годов/этимологии, без «в тексте», без «Источник: …».\n\n"
            "Ответ:"
        )
    if modifier:
        user = apply_followup_modifier(user, modifier, lang)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Жёсткий лимит на каждый web-сниппет до попадания в промпт. DDG иногда
# возвращает 2 КБ сниппеты — забивают весь context-window рекламой и
# breadcrumb'ами; 600 chars хватает на definition-предложение и оставляет
# место под LLM-ответ.
_MAX_WEB_SNIPPET_CHARS = 600


def build_web_messages(
    question: str,
    web_hits: Sequence[WebHit],
    lang: str = "ru",
    *,
    modifier: str | None = None,
) -> list[dict[str, str]]:
    """Промпт для web-fallback-пути. ``web_hits`` — ``list[WebHit]``."""

    def _trim(s: str) -> str:
        s = (s or "").strip()
        if len(s) <= _MAX_WEB_SNIPPET_CHARS:
            return s
        return s[:_MAX_WEB_SNIPPET_CHARS].rstrip() + "…"

    numbered = "\n\n".join(
        f"[{i}] {h.title}\nURL: {h.url}\n{_trim(h.snippet)}"
        for i, h in enumerate(web_hits, start=1)
    )
    if lang == "en":
        system = (
            "/no_think\n"
            "You're answering from web search results because the course "
            "textbook didn't cover this question. Stay grounded in the "
            "snippets — don't invent anything.\n\n"
            "Format (Telegram HTML):\n"
            "• key terms in <b>…</b>;\n"
            "• bullets start with «• »;\n"
            "• short quotes from the snippets in <blockquote>…</blockquote>;\n"
            "• identifiers, formulas, short code in <code>…</code>;\n"
            "• multi-line code or ASCII diagrams in <pre>…</pre>;\n"
            "• cite snippets inline as [1], [2] by their number;\n"
            "• a short example when it helps;\n"
            "• if the snippets are thin, say so honestly."
        )
        user = f"Web search results:\n{numbered}\n\n" f"Student's question: {question}\n\nAnswer:"
    else:
        system = (
            "/no_think\n"
            "Ты отвечаешь по результатам поиска в интернете, потому что в "
            "учебнике этого нет. Строго из сниппетов ниже, ничего не "
            "выдумывай.\n\n"
            "Формат (Telegram HTML):\n"
            "• ключевые термины в <b>…</b>;\n"
            "• перечисления — каждая строка «• …»;\n"
            "• короткие цитаты из сниппетов — в <blockquote>…</blockquote>;\n"
            "• идентификаторы, формулы, короткий код — в <code>…</code>;\n"
            "• многострочный код или схемы — в <pre>…</pre>;\n"
            "• ссылайся на сниппеты [1], [2] по их номеру ниже;\n"
            "• короткий пример, если помогает;\n"
            "• если инфы мало — честно скажи."
        )
        user = (
            f"Результаты поиска в интернете:\n{numbered}\n\n"
            f"Вопрос студента: {question}\n\nОтвет:"
        )
    if modifier:
        user = apply_followup_modifier(user, modifier, lang)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

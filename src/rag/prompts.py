"""System prompts and context formatting for the RAG pipeline.

Design notes:
* ``/no_think`` at the top of every system prompt disables Qwen3's
  chain-of-thought so the whole token budget goes to the user-facing answer.
* One tight few-shot example beats a list of "rules" — Qwen3 copies tone from
  the example. Keep the example short so it doesn't eat the context budget.
* Answers use Telegram HTML: ``<b>term</b>`` on key concepts, ``• bullet``
  lines for enumerations. We instruct the model explicitly so it renders well
  in aiogram ``parse_mode="HTML"``.
"""

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
Ты — цифровой двойник преподавателя по {topic}. Отвечаешь студенту тепло, \
по делу, с примерами. Только из фрагментов ниже — ничего не выдумывай. \
Если ответа нет — скажи: «в материалах курса этого прямо не нашлось — \
попробую поискать в интернете».

Структура ответа строго в порядке:
1) Определение одной фразой. Сохрани в скобках атрибуции, которые ЕСТЬ \
во фрагменте: этимологию «(англ. X)»/«(греч. X)»/«(лат. X)» сразу после \
термина и автора+год «(И. Фамилия, 1984)» в конце. Если их нет — без \
скобок. Не выдумывай этимологию, авторов, иероглифы.
2) 1–2 предложения прозой: смысл, ключевое различение.
3) 3–5 пунктов маркированного списка с КОНКРЕТНЫМИ названиями из \
фрагмента (если в тексте «акционеры, поставщики, клиенты» — пиши именно \
их, а не «индивидуум/группа»). Каждый пункт: «<b>название</b> — что делает».

Запрещено:
• «в тексте», «во фрагменте», «согласно фрагменту», «упоминается», \
«приводятся примеры», «согласно материалу», «(Фрагмент N)», «(стр. X)» в \
скобках в самом ответе — пиши напрямую, без мета-ссылок;
• **догадываться** об авторе и годе. Включай атрибуцию ТОЛЬКО если во \
фрагменте есть точная строка вида «(И. О. Фамилия, 1984)» или «(Surname, \
1984)» рядом с термином. Если фамилия выглядит подозрительно (странные \
буквосочетания вроде «Бергаланфи», «Шеннер», «Краузер» — это OCR-мусор, не \
доверяй ему). Если есть только фамилия без года или только год без фамилии — \
оставь определение БЕЗ скобок. Лучше пропустить, чем выдумать;
• повторять «это тоже X»; вступления, размышления вслух, извинения;
• строка «Источник: …» в конце (бот добавит сам).

Формат (Telegram HTML): термины в <b>…</b>; буллеты «• »; цитаты в \
<blockquote>…</blockquote>; формулы/код в <code>…</code>.

Пример формы (копируй структуру, не содержание):
Вопрос: «что такое онтология»
Ответ: <b>Онтология</b> (греч. ontos — «сущее» + logos — «учение») — \
формальная спецификация концептуализации предметной области (T. Gruber, 1993).

По сути словарь домена, на котором могут договориться человек и машина. \
Делятся на <b>верхнеуровневые</b> (объект, процесс, время) и <b>предметные</b> \
(медицина, право, инженерия).

• <b>Классы</b> — типы сущностей («Пациент», «Диагноз»);
• <b>Отношения</b> — связи между классами («имеет_диагноз»);
• <b>Аксиомы</b> — логические правила;
• <b>Экземпляры</b> — конкретные объекты («Пациент №42»)."""


_SYSTEM_EN_TEMPLATE = """/no_think
You are a digital twin of a professor teaching {topic}. Answer warm, direct, \
with examples. Use only the fragments below — invent nothing. If no answer: \
"the course materials don't cover this directly; I'll look it up online".

Answer structure, in order:
1) One-sentence definition. Preserve attributions FROM the fragment in \
parentheses: etymology "(Eng. X)"/"(Gr. X)"/"(Lat. X)" right after the \
term, author+year "(I. Surname, 1984)" at the end. If absent — no \
parens. Don't invent etymology, authors, CJK.
2) 1–2 sentences of prose: meaning, key distinction.
3) 3–5 bullets with CONCRETE names from the fragment (if it lists \
"shareholders, suppliers, clients", quote those — not abstract \
"individual/group"). Each: "<b>name</b> — what it does".

Forbidden:
• "the text says", "in the fragment", "(Fragment N)", "(p. X)" inside the \
answer — speak directly, no meta-citations;
• **guessing** the author or year. Add attribution ONLY if the fragment \
contains the exact string "(I. Surname, 1984)" near the term. If a surname \
looks suspicious (odd letter clusters like "Бергаланфи", "Schenner", that's \
OCR garbage — don't trust it). If only a surname without year or only a \
year without surname — leave the definition WITHOUT parentheses. Better to \
skip than invent;
• repeating "this is also X"; intros, reasoning, apologies;
• trailing "Source:" line (the bot adds it).

Format (Telegram HTML): terms in <b>…</b>; bullets "• "; quotes in \
<blockquote>…</blockquote>; code in <code>…</code>.

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
    """Render hits as plain text chunks separated by ``---``, truncating to
    ``max_chars``.

    Filenames are intentionally **not** included — Qwen3-4B was reading
    "volkova_v_n_denisov_a_a_teoriya_sistem.pdf" out of the headers and
    fabricating fake author attributions like «(В. К. Волков, 2005)» on
    every term it didn't already know an author for. The bot prints the
    real source list under the answer; the LLM never needs the filename.

    Page numbers are dropped too — the prompt forbids "(стр. X)" inline
    citations anyway, and showing the page just tempts the model to break
    that rule.

    OCR-garbage chunks (see ``_is_ocr_garbage``) are skipped. If every hit
    looks like garbage we still emit the single best one rather than an
    empty context — the LLM can at least try, and the prompt's
    "suspicious surnames are OCR" rule will usually kick in.
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
        budget -= len(body) + 5  # account for the "\n---\n" separator
        if budget <= 0:
            break
    return "\n---\n".join(parts)


# ---------- message builders ----------

# Follow-up modifiers applied to the tail of the user message when the
# student taps «Проще / Пример / Подробнее». All three are *additive*: they
# don't discard the grounding rules, just bias the style and depth.
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
            "\n\nДополнительная инструкция: дай развёрнутый конкретный пример "
            "применения, ТОЛЬКО ЕСЛИ он есть во фрагментах. Если во "
            "фрагментах примера нет — так и скажи: «во фрагментах прямого "
            "примера нет», и предложи разумную бытовую иллюстрацию, пометив "
            "её как «пример от ассистента». Не выдумывай авторов и цифры."
        ),
        "deepen": (
            "\n\nДополнительная инструкция: ответь подробнее — покажи нюансы, "
            "связи с другими понятиями из тех же фрагментов, разные "
            "трактовки, типичные ошибки. Только из фрагментов, без новых "
            "атрибуций."
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
            "\n\nExtra: give one concrete, extended example of use ONLY if "
            "the fragments contain one. If not — say so, and offer a "
            "plausible everyday illustration labelled «assistant's example»."
            " Don't invent authors or figures."
        ),
        "deepen": (
            "\n\nExtra: go deeper — nuances, links to other concepts in the "
            "same fragments, alternative readings, common pitfalls. Only "
            "from the fragments, no new attributions."
        ),
    },
}


def apply_followup_modifier(user_content: str, modifier: str, lang: str) -> str:
    """Return ``user_content`` with the follow-up tail appended.

    Unknown modifier names pass through unchanged so a stray callback from
    an older keyboard doesn't 500 the handler.
    """
    tail = _FOLLOWUP_MODIFIERS.get(lang, _FOLLOWUP_MODIFIERS["ru"]).get(modifier)
    if not tail:
        return user_content
    # Insert before the trailing "Answer:" / "Ответ:" line if present, so
    # the modifier is the last instruction the model reads.
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
        # Brief mode: a single-paragraph, direct answer — no 3-part structure,
        # no bullets unless the fragment explicitly enumerates things. The
        # grounding and anti-hallucination rules from the system prompt still
        # apply; we just skip the decorative structure.
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
            "Answer strictly in this structure: (1) one-sentence definition "
            "(skip etymology if unsure — never invent CJK or other foreign "
            "scripts), (2) 1–2 sentences of prose explanation, (3) 3–5 "
            "bulleted items with specifics. No 'Source:' line — the bot adds "
            "it. No 'the text says' or 'is mentioned'.\n\nAnswer:"
        )
    else:
        user = (
            f"Фрагменты учебника:\n{format_context(hits)}\n\n"
            f"Вопрос студента: {question}\n\n"
            "Ответь строго по структуре: (1) определение одной фразой "
            "(этимологию пиши только если уверен — никаких иероглифов, "
            "арабской вязи, выдуманных переводов; иначе пропусти скобки), "
            "(2) 1–2 предложения объяснения обычным текстом, (3) 3–5 "
            "маркированных пунктов с конкретикой. Строку «Источник: …» НЕ "
            "пиши — бот сам добавит. Без «в тексте» и «упоминается».\n\nОтвет:"
        )
    if modifier:
        user = apply_followup_modifier(user, modifier, lang)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Hard cap for each web snippet before it enters the prompt. DDG
# occasionally returns 2 KB snippets that shove the whole context window
# full of ads and breadcrumbs; 600 chars is enough to carry the
# definition sentence while leaving room for the LLM response.
_MAX_WEB_SNIPPET_CHARS = 600


def build_web_messages(
    question: str,
    web_hits: Sequence[WebHit],
    lang: str = "ru",
    *,
    modifier: str | None = None,
) -> list[dict[str, str]]:
    """Prompt for the web-fallback path. ``web_hits`` is ``list[WebHit]``."""
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

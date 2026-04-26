"""UI-строки бота, билингва RU + EN.

Каждая константа — ``Tr``-dict с ключами по языку. Используй ``tr(lang, HELLO)``
для рендера; ``tr_all(HELLO)`` возвращает все варианты (нужно для фильтров
reply-keyboard, которые должны матчить любой текст, что показал клиент).
"""

from __future__ import annotations

from collections.abc import Iterable

Tr = dict[str, str]
DEFAULT_LANG = "ru"
SUPPORTED_LANGS = ("ru", "en")


def normalize_lang(code: str | None) -> str:
    """Свести Telegram ``language_code`` к одному из ``SUPPORTED_LANGS``.

    Неизвестные коды фолбечутся в :data:`DEFAULT_LANG`.
    """
    if not code:
        return DEFAULT_LANG
    code = code.lower().split("-")[0]
    if code in SUPPORTED_LANGS:
        return code
    # Экс-СССР-кириллические локали считаем русскими для лучшего дефолта
    if code in {"uk", "be", "kk", "ky", "tg", "uz", "az", "mo"}:
        return "ru"
    return "en"


def tr(lang: str, t: Tr) -> str:
    """Отрендерить ``t`` для ``lang``, fallback в RU потом EN."""
    return t.get(lang) or t.get(DEFAULT_LANG) or t.get("en") or ""


def tr_all(t: Tr) -> set[str]:
    """Все уникальные непустые варианты — для матча текста reply-кнопок."""
    return {v for v in t.values() if v}


def join(lines: Iterable[str]) -> str:
    return "\n".join(lines)


# ---------- greetings ----------

HELLO: Tr = {
    "ru": (
        "👋 Привет, {name}!\n\n"
        "Я — цифровой двойник преподавателя. Отвечаю по учебникам курса, "
        "со ссылками на страницы.\n\n"
        "💡 <b>Например, спроси:</b>\n"
        "• <i>Кто такие стейкхолдеры и как их классифицируют?</i>\n"
        "• <i>В чём разница между SWOT- и STEP-анализом?</i>\n"
        "• <i>Что такое ПИЦ и как построить дерево целей?</i>\n\n"
        "Нажми «💬 Задать вопрос» — или просто напиши сообщением."
    ),
    "en": (
        "👋 Hi, {name}!\n\n"
        "I'm a digital twin of a university professor. I answer from course "
        "textbooks with page references.\n\n"
        "💡 <b>For example, ask:</b>\n"
        "• <i>Who are stakeholders and how are they classified?</i>\n"
        "• <i>What's the difference between SWOT and STEP analysis?</i>\n"
        "• <i>What is a goal tree and how do I build one?</i>\n\n"
        "Tap “💬 Ask a question” — or just type a message."
    ),
}

PROMPT_QUESTION: Tr = {
    "ru": "✍️ Напиши свой вопрос следующим сообщением.",
    "en": "✍️ Type your question as the next message.",
}
PROMPT_QUESTION_HINT: Tr = {
    "ru": "✍️ Напиши свой вопрос — или выбери пример ниже:",
    "en": "✍️ Type your question — or pick a sample below:",
}

# Sample questions used by /ask buttons and inline empty-query suggestions.
# Pairs are (RU, EN) so we don't pay the Tr-dict lookup overhead for an array.
ASK_SAMPLES: tuple[tuple[str, str], ...] = (
    ("Кто такие стейкхолдеры и как их классифицируют?",
     "Who are stakeholders and how are they classified?"),
    ("В чём разница между SWOT- и STEP-анализом?",
     "What's the difference between SWOT and STEP analysis?"),
    ("Что такое ПИЦ и как построить дерево целей?",
     "What is a goal tree (ПИЦ) and how do I build one?"),
    ("Как проанализировать внешнюю среду организации?",
     "How do I analyse the external environment of an organisation?"),
)

# ---------- reply-keyboard buttons ----------

BTN_ASK: Tr = {"ru": "💬 Задать вопрос", "en": "💬 Ask a question"}
BTN_SUBJECTS: Tr = {"ru": "📚 Предметы", "en": "📚 Subjects"}
BTN_HELP: Tr = {"ru": "❓ Помощь", "en": "❓ Help"}
BTN_MORE: Tr = {"ru": "❓ Подробнее", "en": "❓ Learn more"}
BTN_SAMPLE_PREFIX: Tr = {"ru": "💡 ", "en": "💡 "}

# ---------- /help ----------

HELP_STUDENT: Tr = {
    "ru": (
        "<b>🎯 Что я умею</b>\n"
        "Отвечаю на вопросы по учебникам курса со ссылкой на страницу. "
        "Если в книге ответа нет — ищу в интернете и прямо говорю об этом.\n\n"
        "<b>📋 Вопросы</b>\n"
        "• /ask — задать вопрос\n"
        "• /term &lt;слово&gt; — только определение из глоссария (без LLM)\n"
        "• /glossary [слово] — весь глоссарий или поиск\n"
        "• /subjects — список предметов\n"
        "• /subject &lt;slug&gt; — закрепить предмет\n"
        "• /subject clear — снять фиксацию\n\n"
        "<b>🗂 Мои ответы</b>\n"
        "• /history [page] — история вопросов с ⭐ и пагинацией\n"
        "• /favourites — только помеченные ⭐\n"
        "• /find &lt;подстрока&gt; — поиск по моим вопросам\n"
        "• /ref &lt;id&gt; — источники прошлого ответа\n"
        "• /export — скачать всю историю в .txt\n\n"
        "<b>⚙️ Режимы</b>\n"
        "• /mode — <code>verbose</code> (по умолчанию) или <code>brief</code> "
        "(только ответ + источники, без кнопок «Проще/Пример/Подробнее»)\n"
        "• /whoami — моя роль и Telegram ID\n\n"
        "<b>💡 Подсказки</b>\n"
        "• Инлайн: <code>@{bot_username} твой вопрос</code> в любом чате\n"
        "• Кнопки под ответом: 👍/👎 оценка, «Проще / Пример / Подробнее» — "
        "переформулирует тот же ответ без повторного поиска\n"
        "• «Сравни X и Y» — выполняю два поиска параллельно\n\n"
        "<b>⚠️ Чего я не умею</b>\n"
        "• не знаю того, чего нет в учебниках курса (тогда иду в веб)\n"
        "• иногда могу ошибаться — галлюцинации ИИ никто не отменял\n"
        "• для экзамена и курсовой <b>проверь ответ у преподавателя</b>"
    ),
    "en": (
        "<b>🎯 What I can do</b>\n"
        "I answer questions from course textbooks with page references. "
        "If the book doesn't have it, I search the web and say so openly.\n\n"
        "<b>📋 Asking</b>\n"
        "• /ask — ask a question\n"
        "• /term &lt;word&gt; — glossary-only definition (no LLM)\n"
        "• /glossary [word] — full glossary or search\n"
        "• /subjects — list of subjects\n"
        "• /subject &lt;slug&gt; — pin a subject\n"
        "• /subject clear — unpin\n\n"
        "<b>🗂 My answers</b>\n"
        "• /history [page] — past questions with ⭐ and pagination\n"
        "• /favourites — ⭐ marked only\n"
        "• /find &lt;substring&gt; — search my own questions\n"
        "• /ref &lt;id&gt; — sources of a past answer\n"
        "• /export — download full history as .txt\n\n"
        "<b>⚙️ Modes</b>\n"
        "• /mode — <code>verbose</code> (default) or <code>brief</code> "
        "(answer + sources only, no follow-up buttons)\n"
        "• /whoami — my role and Telegram ID\n\n"
        "<b>💡 Tips</b>\n"
        "• Inline: <code>@{bot_username} your question</code> in any chat\n"
        "• Buttons under an answer: 👍/👎, «Simpler / Example / Go deeper» — "
        "reruns the same answer without a fresh retrieval\n"
        "• «Compare X and Y» — two parallel retrievals\n\n"
        "<b>⚠️ What I can't do</b>\n"
        "• I don't know what's not in the textbooks (I fall back to web)\n"
        "• I can hallucinate — it's an LLM after all\n"
        "• for exams and term papers, <b>verify with your professor</b>"
    ),
}

HELP_ADMIN_EXTRA: Tr = {
    "ru": (
        "\n\n<b>🧑‍🏫 Преподаватель / админ</b>\n"
        "• /admin_subjects — предметы и статус индекса\n"
        "• /admin_upload &lt;slug&gt; — загрузить <code>.pdf .docx .md .txt</code>\n"
        "• /admin_reindex [slug] — пересобрать индекс\n"
        "• /admin_stats — диалоги, оценки, latency (24 ч)\n"
        "• /teacher_stats &lt;slug&gt; [days] — метрики по предмету\n"
        "• /teacher_gaps [days] — вопросы без ответа в учебнике\n"
        "• /teacher_review — очередь 👎-отзывов\n"
        "• /teacher_fix &lt;id&gt; — FAQ-ответ на плохой диалог\n"
        "• /teacher_glossary_upload &lt;slug&gt; — загрузить CSV/YAML глоссарий"
    ),
    "en": (
        "\n\n<b>🧑‍🏫 Teacher / admin</b>\n"
        "• /admin_subjects — subjects and index status\n"
        "• /admin_upload &lt;slug&gt; — upload <code>.pdf .docx .md .txt</code>\n"
        "• /admin_reindex [slug] — rebuild the index\n"
        "• /admin_stats — dialogs, ratings, latency (24 h)\n"
        "• /teacher_stats &lt;slug&gt; [days] — per-subject metrics\n"
        "• /teacher_gaps [days] — questions not covered by the textbook\n"
        "• /teacher_review — 👎 feedback queue\n"
        "• /teacher_fix &lt;id&gt; — curate a FAQ answer for a bad dialog\n"
        "• /teacher_glossary_upload &lt;slug&gt; — upload CSV/YAML glossary"
    ),
}
HELP_SUPERADMIN_EXTRA: Tr = {
    "ru": (
        "\n\n<b>🛡 Супер-админ</b>\n"
        "• /admin_promote &lt;tg_id&gt; [teacher|admin|student] — назначить роль\n"
        "• /admin_demote &lt;tg_id&gt; — сбросить роль до student\n"
        "• /admin_users [role] — список пользователей"
    ),
    "en": (
        "\n\n<b>🛡 Super-admin</b>\n"
        "• /admin_promote &lt;tg_id&gt; [teacher|admin|student] — assign role\n"
        "• /admin_demote &lt;tg_id&gt; — reset to student\n"
        "• /admin_users [role] — list users"
    ),
}

# ---------- subjects ----------

BTN_GET_ANSWER: Tr = {"ru": "🔍 Получить ответ", "en": "🔍 Get answer"}
INLINE_EXPIRED: Tr = {
    "ru": "Запрос устарел. Наберите его в строке ещё раз.",
    "en": "Query expired. Retype it in the search bar.",
}

SUBJECTS_LIST_HEADER: Tr = {"ru": "📚 Доступные предметы:\n", "en": "📚 Available subjects:\n"}
SUBJECTS_LIST_ITEM: Tr = {
    "ru": "• <b>{title}</b> — <code>{slug}</code>",
    "en": "• <b>{title}</b> — <code>{slug}</code>",
}
SUBJECTS_EMPTY: Tr = {
    "ru": "Пока нет предметов. Администратор должен загрузить материалы и пересобрать индекс.",
    "en": "No subjects yet. An administrator needs to upload materials and rebuild the index.",
}

SUBJECT_LOCKED: Tr = {
    "ru": (
        "🔒 Зафиксирован предмет: <b>{title}</b>. "
        "Сменить — /subject &lt;slug&gt; или /subject clear."
    ),
    "en": "🔒 Subject pinned: <b>{title}</b>. Change via /subject &lt;slug&gt; or /subject clear.",
}
SUBJECT_CLEARED: Tr = {
    "ru": "🔓 Фиксация предмета снята. Теперь отвечаю авто-роутингом.",
    "en": "🔓 Subject unpinned. Auto-routing is active again.",
}
SUBJECT_UNKNOWN: Tr = {
    "ru": "Неизвестный предмет: <code>{slug}</code>. См. /subjects",
    "en": "Unknown subject: <code>{slug}</code>. See /subjects",
}

# ---------- answer flow ----------

NO_HITS: Tr = {
    "ru": "В материалах курсов нет ответа на этот вопрос. Уточните у преподавателя.",
    "en": "The course materials don't cover this question. Ask your professor.",
}

Q_WITH_STATUS: Tr = {
    "ru": "<blockquote>{q}</blockquote>\n\n⌛ <i>{status}</i>",
    "en": "<blockquote>{q}</blockquote>\n\n⌛ <i>{status}</i>",
}
STATUS_RETRIEVING: Tr = {
    "ru": "Ищу релевантные фрагменты в учебниках…",
    "en": "Searching relevant fragments in the textbooks…",
}
STATUS_WARMING: Tr = {
    "ru": "Прогреваю модели (первый запуск, может занять минуту)…",
    "en": "Warming up models (first boot, this can take a minute)…",
}
STATUS_BUSY: Tr = {
    "ru": "🛑 LLM-сервис временно недоступен — попробуй через минуту.",
    "en": "🛑 The LLM service is briefly unavailable — try again in a minute.",
}
STATUS_THINKING: Tr = {
    "ru": "Формулирую ответ по курсу «{subject}»…",
    "en": "Composing an answer from the “{subject}” course…",
}
STATUS_THINKING_FOUND: Tr = {
    "ru": "Нашёл {n} фрагментов в курсе «{subject}» — формулирую ответ…",
    "en": "Found {n} fragments in “{subject}” — composing the answer…",
}
STATUS_ANALYSING: Tr = {
    "ru": "🧠 Анализирую фрагменты…",
    "en": "🧠 Analysing fragments…",
}
STATUS_WEB_SEARCH: Tr = {
    "ru": "🌐 В учебниках ответа нет — ищу в интернете…",
    "en": "🌐 Not in the textbooks — searching the web…",
}
STATUS_WEB_THINKING: Tr = {
    "ru": "🧠 Формулирую ответ по источникам из сети…",
    "en": "🧠 Composing an answer from the web sources…",
}
STATUS_WEB_VISITING: Tr = {
    "ru": "🔗 Смотрю {host}…",
    "en": "🔗 Looking at {host}…",
}
STATUS_WEB_FOUND: Tr = {
    "ru": "✅ Нашёл {n} источников: {hosts} — читаю…",
    "en": "✅ Found {n} sources: {hosts} — reading…",
}
STATUS_STREAMING: Tr = {
    "ru": "✍️ Печатаю ответ — текст появляется прямо в сообщении.",
    "en": "✍️ Typing the answer — it's appearing in the message above.",
}

ANSWER_BODY: Tr = {
    "ru": (
        "<blockquote>{q}</blockquote>\n\n"
        "📚 Курс: <b>{subject}</b>\n\n"
        "{answer}"
        "\n\n— Источники —\n{sources}"
    ),
    "en": (
        "<blockquote>{q}</blockquote>\n\n"
        "📚 Course: <b>{subject}</b>\n\n"
        "{answer}"
        "\n\n— Sources —\n{sources}"
    ),
}
ANSWER_BODY_WEB: Tr = {
    "ru": (
        "<blockquote>{q}</blockquote>\n\n"
        "🌐 <i>В материалах курса ответа не нашёл — цитирую интернет:</i>\n\n"
        "{answer}"
        "\n\n— Источники —\n{sources}"
    ),
    "en": (
        "<blockquote>{q}</blockquote>\n\n"
        "🌐 <i>Not in the course materials — showing web results instead:</i>\n\n"
        "{answer}"
        "\n\n— Sources —\n{sources}"
    ),
}
ANSWER_BODY_BRIEF: Tr = {
    "ru": "{answer}\n\n<i>Источники:</i> {sources}",
    "en": "{answer}\n\n<i>Sources:</i> {sources}",
}
ANSWER_BODY_BRIEF_WEB: Tr = {
    "ru": "{answer}\n\n<i>🌐 Web-источники:</i> {sources}",
    "en": "{answer}\n\n<i>🌐 Web sources:</i> {sources}",
}
SOURCES_ITEM: Tr = {
    "ru": "• {book}, стр. {page}",
    "en": "• {book}, p. {page}",
}
SOURCES_ITEM_WEB: Tr = {
    "ru": '• <a href="{url}">{title}</a>',
    "en": '• <a href="{url}">{title}</a>',
}

# ---------- feedback ----------

BTN_FEEDBACK_UP: Tr = {"ru": "👍 Полезно", "en": "👍 Helpful"}
BTN_FEEDBACK_DOWN: Tr = {"ru": "👎 Не помогло", "en": "👎 Didn't help"}
BTN_ASK_ANOTHER: Tr = {"ru": "💬 Задать ещё вопрос", "en": "💬 Ask another question"}
BTN_WHATS_HAPPENING: Tr = {"ru": "👀 Что сейчас делается?", "en": "👀 What's happening?"}
MODE_SET: Tr = {
    "ru": "Режим ответа: <b>{mode}</b>.",
    "en": "Answer mode: <b>{mode}</b>.",
}
MODE_VERBOSE_LABEL: Tr = {"ru": "развёрнутый", "en": "verbose"}
MODE_BRIEF_LABEL: Tr = {"ru": "краткий", "en": "brief"}
MODE_HELP: Tr = {
    "ru": (
        "<b>Режимы ответа</b>:\n"
        "• <code>/mode verbose</code> — развёрнутый (по умолчанию): "
        "определение → объяснение → 3–5 пунктов, плюс кнопки «⬇️ Проще / "
        "💡 Пример / 📖 Подробнее» для переформулировки без нового поиска.\n"
        "• <code>/mode brief</code> — краткий: один абзац + источники одной "
        "строкой, без follow-up кнопок. Для «дай ответ и я пошёл».\n"
        "В любом режиме работают 👍/👎 и «Задать ещё вопрос»."
    ),
    "en": (
        "<b>Answer modes</b>:\n"
        "• <code>/mode verbose</code> — detailed (default): definition → "
        "explanation → 3-5 bullets, plus follow-up buttons (⬇️ Simpler / "
        "💡 Example / 📖 Go deeper) to re-shape the answer without a new "
        "retrieval.\n"
        "• <code>/mode brief</code> — one paragraph + inline sources, no "
        "follow-up chrome. For «give me the fact and I'm out».\n"
        "Both modes keep 👍/👎 and «Ask another»."
    ),
}
BTN_FU_SIMPLIFY: Tr = {"ru": "⬇️ Проще", "en": "⬇️ Simpler"}
BTN_FU_EXAMPLE: Tr = {"ru": "💡 Пример", "en": "💡 Example"}
BTN_FU_DEEPEN: Tr = {"ru": "📖 Подробнее", "en": "📖 Go deeper"}
BTN_EXPAND_SHORT: Tr = {
    "ru": "📖 Развёрнутый ответ",
    "en": "📖 Full answer",
}
FU_PLACEHOLDER: Tr = {
    "ru": "🔄 Формулирую по-другому…",
    "en": "🔄 Reformulating…",
}
FU_EXPIRED: Tr = {
    "ru": "Контекст ответа устарел — задай вопрос заново.",
    "en": "Answer context has expired — ask the question again.",
}
ALERT_STAGE_UNKNOWN: Tr = {
    "ru": "Уже почти готово — финализирую ответ. Секунду…",
    "en": "Almost there — finalising the answer. One moment…",
}
FEEDBACK_THANKS: Tr = {"ru": "Спасибо за оценку!", "en": "Thanks for the feedback!"}
FEEDBACK_ALREADY: Tr = {
    "ru": "Ты уже оценил этот ответ.",
    "en": "You already rated this answer.",
}

# ---------- history ----------

HISTORY_HEADER: Tr = {"ru": "📜 Последние вопросы:\n", "en": "📜 Recent questions:\n"}
HISTORY_ITEM: Tr = {
    "ru": "<b>{when}</b> · <i>{subject}</i>\nВ: {q}\nО: {a}",
    "en": "<b>{when}</b> · <i>{subject}</i>\nQ: {q}\nA: {a}",
}
HISTORY_EMPTY: Tr = {"ru": "Пока нет заданных вопросов.", "en": "No questions yet."}
FAVOURITES_EMPTY: Tr = {
    "ru": "Нет избранных вопросов. Нажми ⭐ рядом с ответом в /history.",
    "en": "No favourites yet. Tap ⭐ next to an answer in /history.",
}
FIND_EMPTY: Tr = {
    "ru": "По запросу «{q}» ничего не нашлось в твоих вопросах.",
    "en": "Nothing matches «{q}» in your questions.",
}

# ---------- errors ----------

INTERNAL_ERROR: Tr = {
    "ru": "⚠️ Что-то пошло не так. Попробуй ещё раз через минуту.",
    "en": "⚠️ Something went wrong. Please retry in a minute.",
}

# ---------- access gate ----------

ACCESS_DENIED: Tr = {
    "ru": "Бот сейчас в закрытом тестировании.",
    "en": "The bot is currently in closed testing.",
}

# ---------- glossary ----------

GLOSSARY_EMPTY: Tr = {
    "ru": "📖 Глоссарий пуст. Администратор ещё не загрузил термины.",
    "en": "📖 The glossary is empty. Admin hasn't uploaded terms yet.",
}
GLOSSARY_HEADER_ALL: Tr = {
    "ru": "📖 Глоссарий ({count} термин(ов)):\n",
    "en": "📖 Glossary ({count} term(s)):\n",
}
GLOSSARY_HEADER_SEARCH: Tr = {
    "ru": "📖 По запросу «<i>{query}</i>» — {count} термин(ов):\n",
    "en": "📖 For “<i>{query}</i>” — {count} term(s):\n",
}
GLOSSARY_HEADER_SUBJECT: Tr = {
    "ru": "📖 Глоссарий курса «<b>{subject}</b>» ({count}):\n",
    "en": "📖 Glossary of “<b>{subject}</b>” ({count}):\n",
}
GLOSSARY_ITEM: Tr = {
    "ru": "• <b>{term}</b> — {definition}",
    "en": "• <b>{term}</b> — {definition}",
}
GLOSSARY_NOT_FOUND: Tr = {
    "ru": "🔍 По запросу «<i>{query}</i>» ничего не нашёл.",
    "en": "🔍 Nothing found for “<i>{query}</i>”.",
}
GLOSSARY_TRUNCATED: Tr = {
    "ru": "\n\n<i>…показаны первые 20. Уточни запрос.</i>",
    "en": "\n\n<i>…showing the first 20. Refine the query.</i>",
}

# ---------- inline mode ----------

INLINE_EMPTY_TITLE: Tr = {
    "ru": "Напиши вопрос…",
    "en": "Type a question…",
}
INLINE_EMPTY_DESC: Tr = {
    "ru": "Например: что такое стейкхолдер",
    "en": "For example: what is a stakeholder",
}
INLINE_PREVIEW_TITLE: Tr = {
    "ru": "💬 {q}",
    "en": "💬 {q}",
}
INLINE_PREVIEW_DESC: Tr = {
    "ru": "Отправить — и спросить цифрового двойника преподавателя.",
    "en": "Send it to ask the professor's digital twin.",
}
INLINE_MESSAGE_TEXT: Tr = {
    "ru": "<blockquote>{q}</blockquote>",
    "en": "<blockquote>{q}</blockquote>",
}
BTN_INSTRUCTIONS: Tr = {
    "ru": "📖 Инструкция",
    "en": "📖 How to use",
}

# ---------- helpers ----------


def subject_title(subject: object, lang: str) -> str:
    """Выбрать EN/RU-title у Subject (ORM-ряд или pydantic) по языку."""
    if lang == "en" and getattr(subject, "title_en", None):
        return str(subject.title_en)  # type: ignore[attr-defined]
    return (
        str(getattr(subject, "title_ru", None) or getattr(subject, "title_en", "") or "")
    )

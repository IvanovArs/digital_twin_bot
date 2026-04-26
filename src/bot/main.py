"""Точка входа бота. Запуск: python -m src.bot.main

Последовательность загрузки:
  1. Подгружаем Settings + настраиваем логирование
  2. Создаём aiogram Bot + Dispatcher
  3. Цепляем middlewares (logging → access → lang → session)
  4. Регистрируем routers (common, admin, student, inline, errors)
  5. Запускаем startup-задачи (sync courses.yaml → subjects, глоссарий, bot-commands)
  6. Старт polling (dev) или webhook (prod)
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BotCommandScopeDefault

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from src.bot.handlers import admin as admin_handlers
from src.bot.handlers import common as common_handlers
from src.bot.handlers import errors as error_handlers
from src.bot.handlers import inline as inline_handlers
from src.bot.handlers import student as student_handlers
from src.bot.middlewares.access_mw import AccessMiddleware
from src.bot.middlewares.lang_mw import LangMiddleware
from src.bot.middlewares.logging_mw import LoggingMiddleware
from src.bot.middlewares.session_mw import SessionMiddleware
from src.bot.middlewares.throttle_mw import ThrottleMiddleware
from src.bot.services.glossary_service import sync_glossary_from_yaml
from src.bot.services.subjects_sync import sync_subjects
from src.bot.services.warmup import MODELS_READY
from src.config import ROOT, settings
from src.db.session import get_sessionmaker
from src.logging_conf import configure_logging
from src.rag.config import COURSES_YAML
from src.subjects import load_catalog

log = structlog.get_logger(__name__)

GLOSSARY_DIR = ROOT / "data" / "glossary"
ALEMBIC_INI = ROOT / "alembic.ini"


def _run_migrations() -> None:
    """Применить все pending alembic-миграции на старте бота.

    Исключение пробрасываем — ``_on_startup`` остановится до polling'а:
    наполовину применённая схема хуже, чем отказ загрузиться. Alembic
    оборачивает каждую ревизию в транзакцию на бэкендах с
    DDL-in-transaction (Postgres, SQLite), так что raise = БД либо
    полностью на новой ревизии, либо полностью на предыдущей.
    """
    cfg = AlembicConfig(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    try:
        alembic_command.upgrade(cfg, "head")
    except Exception:
        log.exception("migrations_failed")
        raise
    log.info("migrations_applied")


BOT_COMMANDS_RU = [
    BotCommand(command="ask", description="Задать вопрос"),
    BotCommand(command="term", description="Определение из глоссария"),
    BotCommand(command="glossary", description="Глоссарий курса"),
    BotCommand(command="subjects", description="Список предметов"),
    BotCommand(command="subject", description="Закрепить предмет"),
    BotCommand(command="history", description="История вопросов"),
    BotCommand(command="favourites", description="⭐ Избранное"),
    BotCommand(command="find", description="Поиск по моим вопросам"),
    BotCommand(command="ref", description="Источники прошлого ответа"),
    BotCommand(command="export", description="Скачать историю .txt"),
    BotCommand(command="mode", description="Краткий / развёрнутый ответ"),
    BotCommand(command="whoami", description="Моя роль и Telegram ID"),
    BotCommand(command="help", description="Справка"),
]

BOT_COMMANDS_EN = [
    BotCommand(command="ask", description="Ask a question"),
    BotCommand(command="term", description="Glossary definition"),
    BotCommand(command="glossary", description="Course glossary"),
    BotCommand(command="subjects", description="List of subjects"),
    BotCommand(command="subject", description="Pin a subject"),
    BotCommand(command="history", description="Past questions"),
    BotCommand(command="favourites", description="⭐ Favourites"),
    BotCommand(command="find", description="Search my questions"),
    BotCommand(command="ref", description="Sources of a past answer"),
    BotCommand(command="export", description="Download history as .txt"),
    BotCommand(command="mode", description="Brief / verbose answers"),
    BotCommand(command="whoami", description="My role and Telegram ID"),
    BotCommand(command="help", description="Help"),
]

# Telegram-лимиты: description ≤ 512 chars, short ≤ 120. Длинная версия
# показывается на странице профиля бота; короткая — в поиске чатов.
BOT_DESCRIPTION_RU = (
    "Цифровой двойник преподавателя. Отвечает на вопросы студентов "
    "по учебникам курса: семантический поиск bge-m3 + локальный Qwen3, "
    "без отправки данных во внешние API. Напишите /ask или просто задайте вопрос."
)
BOT_DESCRIPTION_EN = (
    "Digital twin of a university professor. Answers student questions from "
    "course textbooks via bge-m3 semantic search + local Qwen3 — nothing "
    "leaves the server. Type /ask or just send a question."
)
BOT_SHORT_RU = "Помогаю студентам находить ответы в учебниках по курсам."
BOT_SHORT_EN = "I help students find answers in course textbooks."


async def _warm_up_models() -> None:
    """Загрузить bge-m3 и прочитать unified-индекс с диска в worker-thread'е,
    чтобы первый студенческий вопрос не платил cold-start внутри своего
    handler'а (~500 МБ / 5–10 с для bge-m3, ~50–200 мс для index mmap).
    """
    import time

    # Лениво импортируем, чтобы старт бота не тянул sentence-transformers
    # до того, как настроено логирование.
    from src.rag.hybrid import _bm25
    from src.rag.reranker import _model as _reranker_model
    from src.rag.retriever import _load_index
    from src.rag.retriever import _model as _embedder_model

    async def _warm(name: str, loader) -> None:  # type: ignore[no-untyped-def]
        t0 = time.monotonic()
        log.info("model_warming", model=name)
        try:
            await asyncio.to_thread(loader)
        except Exception:
            log.exception("model_warm_failed", model=name)
            return
        log.info("model_ready", model=name, latency_ms=int((time.monotonic() - t0) * 1000))

    # Грузим всё, что нужно горячему retrieval-пути, параллельно: embedder,
    # dense-индекс, BM25-токенизированный корпус, cross-encoder реранкер.
    # Если что-то пропустить — первый вопрос платит cold-start в своём
    # handler'е (только реранкер — до ~5 с лишних).
    await asyncio.gather(
        _warm("bge-m3", _embedder_model),
        _warm("rag-index", _load_index),
        _warm("bm25", _bm25),
        _warm("bge-reranker", _reranker_model),
    )
    MODELS_READY.set()


async def _warm_up_models_safe() -> None:
    """Обёртка, гарантирующая, что ``MODELS_READY`` флипнется даже на crash'е —
    студенческие вопросы не зависнут на полные 120 с warm-up-таймаута.
    """
    try:
        await _warm_up_models()
    except Exception:
        log.exception("warmup_task_crashed")
        # Разблокируем ждущих; они быстро упадут на retrieval и покажут
        # реальную ошибку, а не молча будут блокировать плейсхолдер.
        MODELS_READY.set()


async def _on_startup(bot: Bot) -> None:
    # Миграции автоматом на каждом старте. Если упали (битая миграция,
    # FS-permissions), выходим чисто с громким логом — не cycle'имся
    # под docker --restart=always.
    try:
        await asyncio.to_thread(_run_migrations)
    except Exception as exc:
        log.exception("migrations_failed_aborting_startup")
        raise SystemExit(2) from exc

    catalog = load_catalog(COURSES_YAML)
    async with get_sessionmaker()() as session:
        await sync_subjects(session, catalog)
        await sync_glossary_from_yaml(session, GLOSSARY_DIR)
        await session.commit()

    # Меню команд по языкам. Клиенты с другими локалями падают в default.
    await bot.set_my_commands(
        commands=BOT_COMMANDS_RU,
        scope=BotCommandScopeDefault(),
        language_code="ru",
    )
    await bot.set_my_commands(
        commands=BOT_COMMANDS_EN,
        scope=BotCommandScopeDefault(),
        language_code="en",
    )
    await bot.set_my_commands(commands=BOT_COMMANDS_EN, scope=BotCommandScopeDefault())

    # Описание бота по языкам. Идемпотентно — Telegram возвращает
    # 400 «description is not modified» если не изменилось, мы это глотаем.
    for lang_code, long_txt, short_txt in (
        ("ru", BOT_DESCRIPTION_RU, BOT_SHORT_RU),
        ("en", BOT_DESCRIPTION_EN, BOT_SHORT_EN),
    ):
        try:
            await bot.set_my_description(description=long_txt, language_code=lang_code)
            await bot.set_my_short_description(
                short_description=short_txt, language_code=lang_code
            )
        except Exception:
            log.warning("set_bot_description_failed", lang=lang_code, exc_info=True)

    # Прогрев bge-m3 + индекса в фоне — не блокирует старт polling'а.
    # Обёртка _safe гарантирует, что MODELS_READY флипнется даже на crash'е.
    asyncio.create_task(_warm_up_models_safe(), name="warmup")  # noqa: RUF006

    log.info("startup_complete", subjects=catalog.slugs())


async def main() -> None:
    configure_logging()

    token = settings.BOT_TOKEN.get_secret_value()
    if not token:
        raise RuntimeError("BOT_TOKEN is empty. Put a real token into .env (see .env.example).")

    bot = Bot(
        token=token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
            # Web-fallback answers carry source links — without this Telegram
            # blows them up into a card under every reply.
            link_preview_is_disabled=True,
        ),
    )
    dp = Dispatcher(storage=MemoryStorage())

    # Middlewares (outer, порядок важен: logging → access → throttle → lang → session)
    # Throttle ПОСЛЕ access (allowlist-gate первым), но ДО session/lang —
    # burst-дропнутый update не должен даже открывать DB-сессию.
    dp.update.outer_middleware(LoggingMiddleware())
    dp.update.outer_middleware(AccessMiddleware(settings.allowed_ids))
    dp.update.outer_middleware(ThrottleMiddleware())
    dp.update.outer_middleware(LangMiddleware())
    dp.update.outer_middleware(SessionMiddleware(get_sessionmaker()))

    if settings.allowed_ids:
        log.info("access_allowlist_enabled", allowed_ids=list(settings.allowed_ids))

    # Routers
    dp.include_routers(
        common_handlers.router,
        admin_handlers.router,
        student_handlers.router,
        inline_handlers.router,
        error_handlers.router,
    )

    await _on_startup(bot)

    allowed = dp.resolve_used_update_types()
    if "chosen_inline_result" not in allowed:
        log.warning(
            "chosen_inline_result_not_in_allowed_updates",
            hint="inline deferred answers won't arrive",
        )
    log.info("bot_starting_polling", allowed_updates=allowed)

    # Polling параллельно с healthz/readyz HTTP — оба крутятся вечно.
    from src.bot import web as bot_web

    app = bot_web.build_app()
    try:
        await asyncio.gather(
            bot_web.serve_forever(app),
            dp.start_polling(bot, allowed_updates=allowed),
        )
    finally:
        # На SIGTERM aiogram сразу отменяет start_polling; in-flight
        # qa_pipeline-таски убились бы mid-stream, оставив flushed-но-
        # не-committed Dialog-строки. Сначала дренируем активные с дедлайном.
        from src.bot.services.task_registry import drain as drain_inflight
        from src.rag.llm import aclose_async_client

        with contextlib.suppress(Exception):
            await drain_inflight(timeout=25.0)
        with contextlib.suppress(Exception):
            await aclose_async_client()
        with contextlib.suppress(Exception):
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

# digital_twin_bot

Telegram-бот, который отвечает студентам на вопросы по учебникам разных курсов. Ищет ответ по смыслу в материалах (устойчив к опечаткам и искажённым терминам), сам определяет нужный курс, цитирует страницы. LLM и эмбеддинги — локально, без внешних API.

## Как это работает

```
вопрос → эмбеддинг (bge-m3) → top-k чанков → роутер выбирает курс
                                             → llama.cpp + Qwen3-4B
                                             → ответ + источники
```

Один глобальный numpy-индекс по всем курсам, `subject_slug` в метаданных чанка. Никакой внешней векторной БД. OCR через Tesseract для сканированных PDF.

## Локальный запуск

```bash
python -m venv .venv && .venv\Scripts\activate         # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                                    # вписать BOT_TOKEN
python -m src.rag.ingest                                # собрать индекс по data/books/
python -m src.rag.cli_search "что такое стейкхолдер"    # проверить поиск
```

Для разговора нужен поднятый llama-server:
```bash
python -m src.rag.serve_llm      # Windows, dev: llama-server.exe + Qwen3-8B
python -m src.bot.main           # в другом терминале
```

## Прод на VPS (CPU-only, 8 ГБ RAM)

```bash
cd deploy
curl -L -o ../data/models/Qwen3-4B-Q4_K_M.gguf \
  'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=1'
cp .env.prod.example .env
nano .env                         # BOT_TOKEN, POSTGRES_PASSWORD, ADMIN_TELEGRAM_IDS, ALLOWED_TELEGRAM_IDS
docker compose --env-file .env up -d
```

Подробнее — `deploy/README.md`.

## Добавить курс

1. В `courses.yaml`:
   ```yaml
   - slug: discrete_math
     title_en: "Discrete Mathematics"
     title_ru: "Дискретная математика"
     description_ru: "..."
   ```
2. Положить PDF/TXT в `data/books/discrete_math/`.
3. `python -m src.rag.ingest` или из бота: `/admin_upload discrete_math` → `/admin_reindex`.

## Команды бота

Студент:
- `/start` — приветствие с коротким видео
- `/ask` или кнопка «💬 Задать вопрос» — вопрос → ответ + цитаты + 👍👎
- `/glossary [слово]` — глоссарий курса; с аргументом — поиск по подстроке
- `/subject <slug>` — закрепить предмет на сессию (опционально)
- `/subjects`, `/history`, `/help` (админам показываются их команды тоже)

Админ (`ADMIN_TELEGRAM_IDS` в env):
- `/admin_subjects` — что индексировано
- `/admin_upload <slug>` — загрузить PDF/TXT
- `/admin_reindex [slug]` — пересобрать индекс
- `/admin_stats` — диалогов, средний рейтинг, p95 latency

## Тестовый allowlist

Если в `.env` задан `ALLOWED_TELEGRAM_IDS=12345,67890`, бот отвечает только этим пользователям. Пусто — открыт всем.

## Разработка

```bash
make lint     # ruff
make type     # mypy strict
make test     # pytest
make check    # всё сразу + bandit
```

## Структура

```
src/
├── rag/          ingest, retriever, router, prompts, llm, pipeline
├── subjects/     courses.yaml loader
├── db/           SQLAlchemy модели + session
├── bot/          aiogram handlers, middlewares, services
├── config.py     pydantic-settings (всё env)
└── logging_conf.py
alembic/          миграции
deploy/           Dockerfile + docker-compose
tests/            pytest
```

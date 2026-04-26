# Развёртывание digital_twin_bot на CPU-only VPS

## Требования к железу

| Ресурс | Минимум | Комфортно |
|---|---|---|
| CPU-ядра | 4 | 8 |
| RAM | 8 ГБ | 16 ГБ |
| Диск | 20 ГБ | 40 ГБ |

Проверено на Ubuntu 22.04 / Debian 12.

## Первоначальная настройка

```bash
# 1. Установить Docker Engine + compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # после этого выйти/войти

# 2. Клонировать репо
git clone https://github.com/<your-fork>/digital_twin_bot.git
cd digital_twin_bot

# 3. Скачать LLM (~2,5 ГБ для Q4_K_M)
mkdir -p data/models
curl -L -o data/models/Qwen3-4B-Q4_K_M.gguf \
  'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=1'

# 4. Конфиг окружения
cp deploy/.env.prod.example deploy/.env
nano deploy/.env   # BOT_TOKEN, ADMIN_TELEGRAM_IDS, POSTGRES_PASSWORD
```

## Запуск

```bash
cd deploy
docker compose --env-file .env up -d
docker compose logs -f bot
```

При первом старте скачивается `bge-m3` (~2,3 ГБ) **и** `bge-reranker-v2-m3`
(~570 МБ) в volume `hf_cache`. Дальнейшие рестарты — мгновенно.

## Тонкая настройка производительности

`deploy/.env` содержит ключевые рычаги для маленького VPS:

- `LLAMA_CTX_SIZE=4096` — хватает для top-5 чанков × 650 chars + system + 220
  токенов ответа. Поднятие стоит RAM (KV-cache растёт линейно) и замедляет
  prefill; не трогать без подгона `MAX_CONTEXT_CHARS`.
- `LLAMA_PARALLEL=1` — обрабатываем по одному запросу за раз, чтобы все CPU
  шли на генерацию активного юзера. `parallel=2` имеет смысл от 8 vCPU.
- `LLAMA_BATCH_SIZE=512` + `LLAMA_UBATCH_SIZE=512` — оптимально для prefill
  на CPU. Меньше — медленнее prefill; больше — без выигрыша.
- `LLM_TEMPERATURE=0.7` (env) — это для второстепенных вызовов. Основной
  RAG-стрим в `qa_pipeline._stream_answer_to_ui` форсит `0.1` для
  фактических ответов.
- `LLM_MAX_TOKENS=220` — вместе с новым свободным промптом (без жёсткой
  «3-5 буллетов» структуры) этого хватает на 2-5 предложений с источниками.

KV-cache квантуется (`--cache-type-k q8_0 --cache-type-v q8_0`) и пинится
в RAM (`--mlock`) — латентность стабильна, без свопа. `--cache-reuse 256`
переиспользует префикс system-prompt'а между запросами (~600 токенов
prefill сэкономлено в burst).

## Загрузка учебников

Через Telegram (рекомендуется — работает откуда угодно):

1. Напиши боту. Если твой Telegram ID в `ADMIN_TELEGRAM_IDS`, доступны admin-команды.
2. `/admin_subjects` — список курсов и сколько материалов проиндексировано.
3. `/admin_upload <slug>` — бот попросит файл; отправь PDF/TXT/MD/DOCX.
4. `/admin_reindex <slug>` — пересобрать индекс предмета (или без аргумента
   — всё разом).
5. `/admin_stats` — счётчики, средний рейтинг, p95 latency.

Через SSH (массовая офлайн-загрузка):

```bash
cp /path/to/*.pdf data/books/theory_of_systems/
docker compose exec bot python -m src.rag.ingest
```

## Добавить новый курс

1. В `courses.yaml` — новая запись (уникальный slug + title_en/title_ru).
2. Перезапустить бота: `docker compose restart bot`.
3. Загрузить материалы и пересобрать (см. выше).

## Обновление

```bash
git pull
docker compose --env-file .env build bot
docker compose --env-file .env up -d
# миграции alembic запускаются автоматически на старте бота
```

## Бэкап / восстановление

Данные Postgres лежат в volume `deploy_pgdata`:

```bash
docker compose exec postgres pg_dump -U bot digital_twin > backup.sql
```

Учебники и RAG-индекс — на хосте под `data/books` и `data/index`. Бэкап
обычным `tar`/`rsync`.

## Решение проблем

- **`llama-cpp` healthcheck падает.** Проверь, что модель лежит по пути
  `data/models/$LLM_MODEL_FILE`. Первая загрузка весов занимает ~1 минуту.
- **Бот в петле `connection failed`.** Postgres ещё не готов —
  `depends_on.condition: service_healthy` это лечит, но можно следить
  в `docker compose logs postgres`.
- **Ответ занимает 60+ с.** Норма для 4-ядерного CPU с Qwen3-4B. Увеличить
  `LLAMA_THREADS` до фактического числа ядер; апгрейд до 8 ядер.
- **OCR не работает.** В логах ingest «[warn] OCR недоступен». Tesseract
  ставится в bot-образ; если ты подменил entrypoint, проверь, что
  `tesseract`, `tesseract-ocr-rus`, `tesseract-ocr-eng` присутствуют.
- **Ответы «выдумывают».** Скорее всего среди фрагментов мало
  релевантного. Подстрой `RERANK_MIN_SCORE` (по умолчанию 1.5) и проверь
  через `python -m src.rag.cli_search "твой вопрос"` — нет ли в топе
  явного мусора. Если есть — пересобрать индекс с лучшим OCR.
- **«Не нашлось» хотя в учебнике есть.** Бот идёт в web-fallback. Если
  это нежелательно, поднимите `RERANK_MIN_SCORE` (фильтр строже) или
  пересоберите индекс с большим chunk overlap.

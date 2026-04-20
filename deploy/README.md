# Deploying digital_twin_bot on a CPU-only VPS

## Hardware requirements

| Resource | Minimum | Comfortable |
|---|---|---|
| CPU cores | 4 | 8 |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB | 40 GB |

Tested on Ubuntu 22.04 / Debian 12.

## One-time setup

```bash
# 1. Install Docker Engine + compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # log out/in afterwards

# 2. Clone and bootstrap
git clone https://github.com/<your-fork>/digital_twin_bot.git
cd digital_twin_bot

# 3. Fetch the LLM (5 GB GPU, 2.5 GB Q4)
mkdir -p data/models
curl -L -o data/models/Qwen3-4B-Q4_K_M.gguf \
  'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=1'

# 4. Configure env
cp deploy/.env.prod.example deploy/.env
nano deploy/.env   # fill in BOT_TOKEN, ADMIN_TELEGRAM_IDS, POSTGRES_PASSWORD
```

## Run

```bash
cd deploy
docker compose --env-file .env up -d
docker compose logs -f bot
```

First start downloads the `bge-m3` embedding model (~2.3 GB) **and** the
`bge-reranker-v2-m3` cross-encoder (~570 MB) into the `hf_cache` volume;
subsequent restarts are instant.

## Performance knobs

`deploy/.env` exposes the levers that matter on a small VPS:

- `LLAMA_CTX_SIZE=4096` — enough for our top-8 × 650-char chunks plus the
  600-token answer budget. Raising it costs RAM (KV-cache scales linearly) and
  slows prefill; don't bump it unless you also tune `MAX_CONTEXT_CHARS`.
- `LLAMA_PARALLEL=1` — process one request at a time so all CPU goes to the
  active user's generation. `parallel=2` only helps when you have ≥8 vCPU.
- `LLAMA_BATCH_SIZE=128` — match `parallel=1`; larger batches don't help when
  nothing else is in-flight.
- `LLM_TEMPERATURE=0.7`, `LLM_TOP_P=0.8`, `LLM_TOP_K=20`, `LLM_MIN_P=0.0`,
  `LLM_REPEAT_PENALTY=1.05` — Qwen3 non-think recipe (official). Change only
  if answers feel off; raising `temperature` to 1.0 makes them more creative
  but less grounded in the textbook.

The KV-cache is quantised (`--cache-type-k q4_0 --cache-type-v q4_0`) and
pinned in RAM (`--mlock`) so latency is stable across requests instead of
fluctuating with swap pressure.

## Feeding textbooks

Via Telegram (recommended — works from anywhere):

1. Talk to your bot. If your Telegram ID is in `ADMIN_TELEGRAM_IDS`, admin commands unlock.
2. `/admin_subjects` — see what courses and how many materials are indexed.
3. `/admin_upload <slug>` — the bot asks for a file; send a PDF or TXT.
4. `/admin_reindex <slug>` — re-build the RAG index for that subject (or run
   `/admin_reindex` without args to rebuild all subjects).
5. `/admin_stats` — request counts, average rating, p95 latency.

Via SSH (offline bulk load):

```bash
# Drop files directly, then reindex
cp /path/to/*.pdf data/books/theory_of_systems/
docker compose exec bot python -m src.rag.ingest
```

## Adding a new subject

1. Edit `courses.yaml`, add a new entry (unique slug, titles in EN + RU).
2. Restart bot: `docker compose restart bot`.
3. Upload materials and reindex (see above).

## Upgrading

```bash
git pull
docker compose --env-file .env build bot
docker compose --env-file .env up -d
# alembic migrations run automatically on bot startup
```

## Backup / restore

Postgres data lives in the `deploy_pgdata` volume:

```bash
docker compose exec postgres pg_dump -U bot digital_twin > backup.sql
```

Books and the RAG index live under `data/books` and `data/index` on the host —
back them up with `tar` or `rsync`.

## Troubleshooting

- **`llama-cpp` healthcheck failing.** Check the model file exists at
  `data/models/$LLM_MODEL_FILE`. First start takes ~1 minute to load weights.
- **Bot loops on `connection failed`.** Postgres not yet ready —
  `depends_on.condition: service_healthy` handles this, but you can watch
  `docker compose logs postgres`.
- **Answer takes 60+ s.** Expected on 4-core CPU with Qwen3-4B. Increase
  `LLAMA_THREADS` to match actual cores (up to nproc). Upgrade to 8 cores.
- **OCR not working.** Ingest logs `[warn] OCR недоступен`. Tesseract is
  installed inside the bot image; if you mounted a different entrypoint,
  make sure `tesseract`, `tesseract-ocr-rus`, `tesseract-ocr-eng` are present.

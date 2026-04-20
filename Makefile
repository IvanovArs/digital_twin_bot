.PHONY: help install install-dev run bot ingest serve-llm migrate makemigration \
        lint format type test cov security check \
        up down logs clean backup

PY ?= .venv/Scripts/python.exe

help:
	@echo "Targets:"
	@echo "  install       — install runtime deps into .venv"
	@echo "  install-dev   — install runtime + dev/test deps"
	@echo "  run           — one-shot: start llama-server + bot locally"
	@echo "  bot           — run only the Telegram bot (expects llama-server up)"
	@echo "  ingest        — (re)build the RAG index from data/books/"
	@echo "  serve-llm     — start local llama-server with Qwen3-8B for dev"
	@echo "  migrate       — apply alembic migrations"
	@echo "  makemigration MSG=..   — generate a new alembic revision"
	@echo "  lint          — ruff check"
	@echo "  format        — ruff format"
	@echo "  type          — mypy src/"
	@echo "  test          — pytest"
	@echo "  cov           — pytest with coverage"
	@echo "  security      — bandit -r src/"
	@echo "  check         — lint + type + test + security (CI-equivalent)"
	@echo "  up / down     — docker compose up/down (prod stack)"
	@echo "  logs          — docker compose logs -f bot"
	@echo "  backup        — snapshot DB + data/index + data/books to backups/"

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

bot:
	$(PY) -m src.bot.main

run:
	$(PY) -m src.bot.run

ingest:
	$(PY) -m src.rag.ingest

serve-llm:
	$(PY) -m src.rag.serve_llm

migrate:
	$(PY) -m alembic upgrade head

makemigration:
	$(PY) -m alembic revision --autogenerate -m "$(MSG)"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

type:
	$(PY) -m mypy src/

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest --cov=src --cov-report=term-missing

security:
	$(PY) -m bandit -c pyproject.toml -r src/ -ll

check: lint type test security

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down

logs:
	docker compose -f deploy/docker-compose.yml logs -f bot

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# Snapshot a full working copy of everything that can't be rebuilt from git:
# the SQLite DB (dialogs/feedback/FAQ), the RAG index, and the uploaded
# textbook/glossary files. Embeddings + chunks get re-generated from books
# via `make ingest`, but doing a bundle backup is idempotent and safer.
# For Postgres prod, swap the DB line for pg_dump.
backup:
	@mkdir -p backups
	@ts=$$(date -u +%Y%m%d-%H%M%S); \
	out=backups/snap-$$ts.tar.gz; \
	echo "→ $$out"; \
	tar --exclude='data/index/*.bak' -czf $$out \
	    $$(ls bot.db 2>/dev/null || true) \
	    data/index data/books data/glossary 2>/dev/null || \
	    echo "backup: some sources are missing — that's okay if you haven't populated them yet"
	@echo "Latest backups:"; ls -lht backups/ 2>/dev/null | head -5

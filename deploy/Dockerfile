# syntax=docker/dockerfile:1.6

# -----------------------------------------------------------------------------
# Stage 1 — builder: install Python deps into a venv
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — runtime: slim image with Tesseract + source code
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

COPY --from=builder /venv /venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app courses.yaml ./
COPY --chown=app:app src ./src

# Writable volumes mounted at runtime
RUN mkdir -p /app/data/books /app/data/index /app/data/models /app/.cache/huggingface \
    && chown -R app:app /app/data /app/.cache

USER app

# Entrypoint: alembic migrations run inside main.py::_on_startup, no need to double them here.
CMD ["python", "-m", "src.bot.main"]

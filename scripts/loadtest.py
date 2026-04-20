"""Concurrency / latency harness for the RAG pipeline.

Bypasses Telegram entirely — fires N workers that loop over a fixed query set
straight against ``run_qa_pipeline`` (full mode) or ``resolve_subject``
(retrieval-only mode). Measures per-query wall-time, prints p50/p95/p99 and
throughput, and surfaces where the bottleneck is so you don't have to guess.

Usage:
    # 1. Make sure the model + index are warm (or they will be after first run)
    # 2. (full mode only) start llama-server: python -m src.bot.run --no-bot
    #    OR docker compose up llama-cpp
    # 3. Run the harness:

    python scripts/loadtest.py --mode retrieval --concurrency 8 --queries 40
    python scripts/loadtest.py --mode full      --concurrency 4 --queries 12

The full mode talks to the live LLM, so total wall-time can be minutes.
Retrieval-only is fast — measures bge-m3 + cosine + glossary expansion only.

Designed to run on the operator's box (dev or VPS) without touching prod
data: no DB writes, no Telegram, no dialog persistence.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

# Force UTF-8 stdio on Windows so ✅ / Cyrillic don't crash on cp1251.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Realistic queries from the indexed material. Mix of definitional ("что
# такое X") and applied ("как сделать Y") so we don't over-optimise for a
# single query shape.
QUERIES = [
    "что такое стейкхолдер",
    "что такое эмерджентность",
    "как построить дерево целей",
    "в чём разница между swot и step анализом",
    "что такое обратная связь в системе",
    "опиши методику Перегудова Сагатовского",
    "что такое жизненный цикл системы",
    "кто такие лпр и что они делают",
    "что такое чёрный ящик",
    "как анализировать внешнюю среду организации",
    "опиши пиц пространство инициирования целей",
    "что такое декомпозиция системы",
]


@dataclass
class Sample:
    query: str
    elapsed_s: float
    ok: bool
    info: str = ""


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    k = (len(sorted_xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def _summarise(samples: list[Sample], total_wall_s: float, concurrency: int) -> None:
    oks = [s for s in samples if s.ok]
    fails = [s for s in samples if not s.ok]
    times = [s.elapsed_s for s in oks]
    print()
    print("=" * 72)
    print(f"Concurrency: {concurrency}    Total queries: {len(samples)}")
    print(f"Wall time:   {total_wall_s:.1f} s")
    print(f"Throughput:  {len(oks) / total_wall_s:.2f} q/s   "
          f"(per worker: {len(oks) / total_wall_s / concurrency:.2f} q/s)")
    if times:
        print(f"Latency p50: {_percentile(times, 0.5):.2f} s")
        print(f"Latency p95: {_percentile(times, 0.95):.2f} s")
        print(f"Latency p99: {_percentile(times, 0.99):.2f} s")
        print(f"Latency max: {max(times):.2f} s")
        print(f"Latency mean:{statistics.mean(times):.2f} s")
    if fails:
        print(f"Failed: {len(fails)} → first error: {fails[0].info}")
    print("=" * 72)


# ---------- retrieval-only mode (no LLM needed) ----------


async def _retrieve_once(query: str) -> Sample:
    from src.rag.pipeline import resolve_subject

    t0 = time.monotonic()
    try:
        subject, hits, _ = await asyncio.to_thread(resolve_subject, query, None)
        info = (
            f"subject={subject.slug if subject else '∅'} "
            f"hits={len(hits)} top={hits[0].score:.3f}" if hits else "no hits"
        )
        return Sample(query=query, elapsed_s=time.monotonic() - t0, ok=True, info=info)
    except Exception as exc:
        return Sample(
            query=query,
            elapsed_s=time.monotonic() - t0,
            ok=False,
            info=f"{type(exc).__name__}: {exc}",
        )


# ---------- full pipeline mode (needs LLM) ----------


async def _full_once(query: str) -> Sample:

    from src.bot.services.qa_pipeline import run_qa_pipeline
    from src.bot.services.warmup import MODELS_READY
    from src.db.models import User

    if not MODELS_READY.is_set():
        # Loadtest harness loads models eagerly in main(); we just need to
        # flag them ready so the pipeline doesn't show STATUS_WARMING.
        MODELS_READY.set()

    # Stand-in for DB session/user — the harness skips persistence by passing
    # a stub session whose .add/.flush are no-ops. We never commit.
    class _StubSession:
        def add(self, *_: object, **__: object) -> None: ...
        async def flush(self) -> None: ...
        async def execute(self, *_: object, **__: object) -> _Empty:
            return _Empty()
        async def commit(self) -> None: ...

    class _Empty:
        def scalar_one_or_none(self) -> None:
            return None

    user = User(
        id=0, telegram_id=0, full_name="loadtest", username="loadtest",
    )

    sink: list[str] = []

    async def set_status(_text: str) -> None:
        return None

    async def set_final(body: str, _dialog_id: int) -> None:
        sink.append(body)

    t0 = time.monotonic()
    try:
        # save_dialog needs a real session; patch it out for the loadtest.
        from src.bot.services import dialog_service as _ds

        async def _fake_save(*_: object, **__: object):  # type: ignore[no-untyped-def]
            class _D:
                id = 0
            return _D()

        original = _ds.save_dialog
        _ds.save_dialog = _fake_save  # type: ignore[assignment]
        try:
            await run_qa_pipeline(
                question=query,
                session=_StubSession(),  # type: ignore[arg-type]
                user=user,
                lang="ru",
                set_status=set_status,
                set_final=set_final,
            )
        finally:
            _ds.save_dialog = original  # type: ignore[assignment]
        ok = bool(sink)
        info = f"answer_len={len(sink[0]) if sink else 0}"
        return Sample(query=query, elapsed_s=time.monotonic() - t0, ok=ok, info=info)
    except Exception as exc:
        return Sample(
            query=query,
            elapsed_s=time.monotonic() - t0,
            ok=False,
            info=f"{type(exc).__name__}: {exc}",
        )


# ---------- worker pool ----------


async def _worker(
    name: int,
    queue: asyncio.Queue[str | None],
    runner: Callable[[str], Awaitable[Sample]],
    samples: list[Sample],
    log_buf: io.StringIO,
) -> None:
    while True:
        q = await queue.get()
        if q is None:
            queue.task_done()
            return
        s = await runner(q)
        samples.append(s)
        line = (
            f"  [w{name:02d}] {s.elapsed_s:5.2f}s "
            f"{'OK' if s.ok else 'ER'} {q[:48]:<48} {s.info}"
        )
        print(line)
        log_buf.write(line + "\n")
        queue.task_done()


async def _warm_up() -> None:
    """Force one cold load of bge-m3 + the index so timings start fair."""
    from src.rag.retriever import _load_index, _model

    print("warm: loading bge-m3 + index ...")
    t0 = time.monotonic()
    await asyncio.to_thread(_model)
    await asyncio.to_thread(_load_index)
    print(f"warm: ready in {time.monotonic() - t0:.1f} s")


async def _run(
    *,
    concurrency: int,
    repetitions: int,
    runner: Callable[[str], Awaitable[Sample]],
) -> None:
    queries = (QUERIES * ((repetitions // len(QUERIES)) + 1))[:repetitions]
    print(f"running {len(queries)} queries with concurrency {concurrency} ...")

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for q in queries:
        queue.put_nowait(q)
    for _ in range(concurrency):
        queue.put_nowait(None)

    samples: list[Sample] = []
    log_buf = io.StringIO()

    t0 = time.monotonic()
    workers = [
        asyncio.create_task(_worker(i, queue, runner, samples, log_buf))
        for i in range(concurrency)
    ]
    await asyncio.gather(*workers)
    total_wall_s = time.monotonic() - t0

    _summarise(samples, total_wall_s, concurrency)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("retrieval", "full"),
        default="retrieval",
        help="retrieval = bge-m3 + cosine only (no LLM); full = entire pipeline",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--queries", type=int, default=24, help="total queries to fire")
    args = p.parse_args()

    asyncio.run(_warm_up())
    runner = _retrieve_once if args.mode == "retrieval" else _full_once
    asyncio.run(_run(concurrency=args.concurrency, repetitions=args.queries, runner=runner))


if __name__ == "__main__":
    main()

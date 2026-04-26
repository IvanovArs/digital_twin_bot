"""End-to-end CLI: question → answer + sources.

Требует запущенный llama-server (см. `python -m src.rag.serve_llm`).
"""

from __future__ import annotations

import argparse
import sys

from src.rag.llm import ping
from src.rag.pipeline import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Спроси цифрового двойника преподавателя")
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument(
        "--subject",
        dest="subject",
        help="ограничить ответ одним предметом (slug из courses.yaml)",
    )
    args = parser.parse_args()

    if not ping():
        print(
            "llama-server не отвечает. Запусти: python -m src.rag.serve_llm",
            file=sys.stderr,
        )
        sys.exit(2)

    q = " ".join(args.question)
    print(f"Вопрос: {q}\n")

    result = ask(q, subject_slug=args.subject, k=args.k)

    if result.subject is not None:
        print(f"📚 Курс: {result.subject.title_ru}")
    if result.route is not None and result.route.ambiguous:
        print(
            "⚠️  Не уверен в курсе. Варианты: "
            + ", ".join(f"{s} ({v:.2f})" for s, v in result.route.ranked[:3])
        )
    print()
    print(result.answer)
    if result.hits:
        print("\n— Источники —")
        for h in result.hits:
            print(f"  {h.book}, стр. {h.page}  (score={h.score:.2f})")


if __name__ == "__main__":
    main()

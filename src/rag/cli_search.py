"""Ручная проверка поиска: python -m src.rag.cli_search "что такое логарифм?" """

from __future__ import annotations

import argparse

from src.rag.retriever import search
from src.rag.router import detect_subject


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест поиска по учебникам")
    parser.add_argument("query", nargs="+", help="вопрос (можно с опечатками)")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument(
        "--subject",
        dest="subject",
        help="ограничить поиск одним предметом (slug из courses.yaml)",
    )
    args = parser.parse_args()

    q = " ".join(args.query)
    hits = search(q, k=args.k, subject_slug=args.subject)

    print(f"\nЗапрос: {q}\n" + "=" * 70)
    if not hits:
        print("Ничего не найдено.")
        return

    if args.subject is None:
        route = detect_subject(hits)
        print(
            f"Роутер: subject={route.subject_slug}  margin={route.margin:.2f}  "
            f"ambiguous={route.ambiguous}"
        )
        print("Голоса:", ", ".join(f"{s}={v:.2f}" for s, v in route.ranked))
        print("-" * 70)

    for i, h in enumerate(hits, 1):
        preview = h.text[:300].replace("\n", " ")
        print(f"\n[{i}] score={h.score:.3f}  subject={h.subject_slug}  " f"{h.book}  стр. {h.page}")
        print(preview + ("…" if len(h.text) > 300 else ""))


if __name__ == "__main__":
    main()

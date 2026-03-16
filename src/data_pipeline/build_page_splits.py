from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data_pipeline.dataset_utils import deterministic_hash_int, infer_section


def load_pages(docs_file: Path) -> list[str]:
    pages: list[str] = []
    with docs_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_id = row.get("id")
            if page_id:
                pages.append(str(page_id))
    return sorted(set(pages))


def split_pages(pages: list[str], gold_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    by_section: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        by_section[infer_section(page)].append(page)

    gold_pages: list[str] = []
    silver_pages: list[str] = []
    for section, section_pages in sorted(by_section.items()):
        ordered = sorted(section_pages, key=lambda p: deterministic_hash_int(seed, p))
        n_gold = max(1, round(len(ordered) * gold_ratio))
        n_gold = min(n_gold, len(ordered))
        section_gold = ordered[:n_gold]
        section_silver = ordered[n_gold:]
        if not section_silver and section_gold:
            # Keep at least one silver page per section.
            section_silver.append(section_gold.pop())
        gold_pages.extend(section_gold)
        silver_pages.extend(section_silver)
    return sorted(silver_pages), sorted(gold_pages)


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-file", default="data/processed/fastapi_pages.jsonl")
    ap.add_argument("--silver-pages-out", default="data/processed/silver_pages.txt")
    ap.add_argument("--gold-pages-out", default="data/processed/gold_pages.txt")
    ap.add_argument("--summary-out", default="data/processed/page_split_summary.json")
    ap.add_argument("--gold-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pages = load_pages(Path(args.docs_file))
    silver_pages, gold_pages = split_pages(pages, gold_ratio=args.gold_ratio, seed=args.seed)

    write_lines(Path(args.silver_pages_out), silver_pages)
    write_lines(Path(args.gold_pages_out), gold_pages)

    summary = {
        "total_pages": len(pages),
        "silver_pages": len(silver_pages),
        "gold_pages": len(gold_pages),
        "gold_ratio": args.gold_ratio,
        "seed": args.seed,
        "silver_sections": {},
        "gold_sections": {},
    }
    for p in silver_pages:
        s = infer_section(p)
        summary["silver_sections"][s] = summary["silver_sections"].get(s, 0) + 1
    for p in gold_pages:
        s = infer_section(p)
        summary["gold_sections"][s] = summary["gold_sections"].get(s, 0) + 1

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "silver_pages": len(silver_pages),
                "gold_pages": len(gold_pages),
                "summary_out": args.summary_out,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

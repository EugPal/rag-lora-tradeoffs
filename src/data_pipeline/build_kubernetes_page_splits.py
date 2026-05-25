from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data_pipeline.dataset_utils import deterministic_hash_int
from src.utils.io_utils import read_jsonl


def page_section(page: dict) -> str:
    section = str(page.get("section") or "").strip().lower()
    if section:
        return section
    url = str(page.get("url") or "").lower()
    if "/docs/concepts/" in url:
        return "concepts"
    if "/docs/tasks/" in url:
        return "tasks"
    if "/docs/reference/" in url:
        return "reference"
    if "/docs/tutorials/" in url:
        return "tutorials"
    if "/docs/setup/" in url:
        return "setup"
    if "/docs/contribute/" in url:
        return "contribute"
    return "general"


def split_pages(pages: list[dict], gold_ratio: float, seed: int) -> tuple[list[str], list[str], dict]:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_section[page_section(page)].append(page)

    gold_pages: list[str] = []
    silver_pages: list[str] = []
    summary = {"silver_sections": {}, "gold_sections": {}}

    for section, section_pages in sorted(by_section.items()):
        ordered = sorted(section_pages, key=lambda row: deterministic_hash_int(seed, row["id"]))
        n_gold = max(1, round(len(ordered) * gold_ratio))
        n_gold = min(n_gold, len(ordered))
        section_gold = ordered[:n_gold]
        section_silver = ordered[n_gold:]
        if not section_silver and section_gold:
            section_silver.append(section_gold.pop())

        gold_ids = [row["id"] for row in section_gold]
        silver_ids = [row["id"] for row in section_silver]
        gold_pages.extend(gold_ids)
        silver_pages.extend(silver_ids)
        summary["gold_sections"][section] = len(gold_ids)
        summary["silver_sections"][section] = len(silver_ids)

    return sorted(silver_pages), sorted(gold_pages), summary


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build section-aware page splits for Kubernetes docs.")
    parser.add_argument(
        "--pages-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/kubernetes_pages.jsonl"),
    )
    parser.add_argument(
        "--silver-pages-out",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/silver_pages.txt"),
    )
    parser.add_argument(
        "--gold-pages-out",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/gold_pages.txt"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/page_split_summary.json"),
    )
    parser.add_argument("--gold-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pages = read_jsonl(args.pages_file)
    if not pages:
        raise SystemExit(f"No pages found in {args.pages_file}")

    silver_pages, gold_pages, split_summary = split_pages(pages, gold_ratio=args.gold_ratio, seed=args.seed)
    write_lines(args.silver_pages_out, silver_pages)
    write_lines(args.gold_pages_out, gold_pages)

    summary = {
        "total_pages": len(pages),
        "silver_pages": len(silver_pages),
        "gold_pages": len(gold_pages),
        "gold_ratio": args.gold_ratio,
        "seed": args.seed,
        **split_summary,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"silver_pages": len(silver_pages), "gold_pages": len(gold_pages)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

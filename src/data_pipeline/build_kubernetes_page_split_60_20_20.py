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


def allocate_counts(n: int) -> tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = max(1, round(n * 0.6))
    n_eval = max(1, round(n * 0.2))
    n_test = n - n_train - n_eval
    if n_test <= 0:
        n_test = 1
        if n_train >= n_eval and n_train > 1:
            n_train -= 1
        elif n_eval > 1:
            n_eval -= 1
    while n_train + n_eval + n_test > n:
        if n_train >= n_eval and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_eval >= n_test and n_eval > 1:
            n_eval -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_train, n_eval, n_test


def split_pages(pages: list[dict], seed: int) -> tuple[list[str], list[str], list[str], dict]:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_section[page_section(page)].append(page)

    train_pages: list[str] = []
    eval_pages: list[str] = []
    test_pages: list[str] = []
    summary = {"train_sections": {}, "eval_sections": {}, "test_sections": {}}

    for section, section_pages in sorted(by_section.items()):
        ordered = sorted(section_pages, key=lambda row: deterministic_hash_int(seed, row["id"]))
        n_train, n_eval, n_test = allocate_counts(len(ordered))
        train_ids = [row["id"] for row in ordered[:n_train]]
        eval_ids = [row["id"] for row in ordered[n_train : n_train + n_eval]]
        test_ids = [row["id"] for row in ordered[n_train + n_eval : n_train + n_eval + n_test]]

        train_pages.extend(train_ids)
        eval_pages.extend(eval_ids)
        test_pages.extend(test_ids)
        summary["train_sections"][section] = len(train_ids)
        summary["eval_sections"][section] = len(eval_ids)
        summary["test_sections"][section] = len(test_ids)

    return sorted(train_pages), sorted(eval_pages), sorted(test_pages), summary


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a section-aware 60/20/20 Kubernetes page split.")
    parser.add_argument(
        "--pages-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/kubernetes_pages.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/page_split_60_20_20"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pages = read_jsonl(args.pages_file)
    if not pages:
        raise SystemExit(f"No pages found in {args.pages_file}")

    train_pages, eval_pages, test_pages, summary_parts = split_pages(pages, seed=args.seed)
    write_lines(args.out_dir / "page_ids_train.txt", train_pages)
    write_lines(args.out_dir / "page_ids_eval.txt", eval_pages)
    write_lines(args.out_dir / "page_ids_test.txt", test_pages)

    summary = {
        "total_pages": len(pages),
        "train_pages": len(train_pages),
        "eval_pages": len(eval_pages),
        "test_pages": len(test_pages),
        "seed": args.seed,
        **summary_parts,
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "train_pages": len(train_pages),
                "eval_pages": len(eval_pages),
                "test_pages": len(test_pages),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

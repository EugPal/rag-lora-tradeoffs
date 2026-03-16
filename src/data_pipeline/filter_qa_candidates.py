from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from src.data_pipeline.dataset_utils import (
    allocate_category_targets,
    infer_category,
    infer_section,
    normalize,
    source_page_id,
)
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


BAD_SUBSTRINGS = [
    "<span",
    "<font",
    "javascript:",
    "mailto:",
    "back to top",
    "waiting list",
    "subscribe",
    "<html",
    "<script",
    "function(",
    "json.parse",
    "copied from",
    "oauth2-redirect.html",
]


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text))


def _looks_code_like(text: str) -> bool:
    lower = text.lower()
    code_markers = [
        "def ",
        "class ",
        "return ",
        "import ",
        "from ",
        "=>",
        "::",
        "().",
        "{",
        "}",
        "[",
        "]",
        "| None",
    ]
    marker_hits = sum(1 for marker in code_markers if marker in text or marker in lower)
    punct = sum(ch in "{}[]()|:;" for ch in text)
    return marker_hits >= 3 or punct >= 12


def sent_count(text: str) -> int:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if p.strip()])


def is_valid_row(row: dict, valid_chunk_ids: set[str]) -> bool:
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()
    source_chunk = row.get("source_chunk")

    if not question or not answer or not source_chunk:
        return False
    if source_chunk not in valid_chunk_ids:
        return False

    text_all = f"{question}\n{answer}".lower()
    if any(b in text_all for b in BAD_SUBSTRINGS):
        return False
    # Reject line-number dumps / high-digit noise.
    if re.search(r"\d{4,}", question) or re.search(r"\d{4,}", answer):
        return False
    digits = sum(ch.isdigit() for ch in text_all)
    if len(text_all) > 0 and (digits / max(1, len(text_all))) > 0.15:
        return False
    # Reject fragments that look like signatures/snippets rather than prose.
    if _looks_code_like(answer):
        return False
    # Very low alphabetic ratio usually means token soup / markup residue.
    if _alpha_ratio(answer) < 0.55:
        return False

    answer_words = answer.split()
    if not (8 <= len(answer_words) <= 80):
        return False
    if sent_count(answer) > 4:
        return False

    qn = normalize(question)
    an = normalize(answer)
    # Keep synthetic prompts, but avoid outright duplicates / trivial echoes.
    if qn == an or (an and an in qn):
        return False

    return True


def filter_candidates(
    rows: list[dict],
    valid_chunk_ids: set[str],
    target_size: int,
    seed: int,
    max_per_page: int,
    allowed_pages: set[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    category_targets = allocate_category_targets(target_size)
    by_category: dict[str, list[dict]] = defaultdict(list)
    overflow: list[dict] = []
    seen_q: set[str] = set()
    seen_a: set[str] = set()
    per_page: dict[str, int] = {}

    for row in rows:
        if not is_valid_row(row, valid_chunk_ids):
            continue

        qn = normalize(row["question"])
        an = normalize(row["answer"])
        if qn in seen_q or an in seen_a:
            continue

        p = source_page_id(row["source_chunk"])
        if not p:
            continue
        if allowed_pages is not None and p not in allowed_pages:
            continue
        if per_page.get(p, 0) >= max_per_page:
            continue

        category = infer_category(row["question"], row["answer"], p)
        section = infer_section(p)
        row = dict(row)
        row["source_page"] = p
        row["section"] = section
        row["category"] = category

        seen_q.add(qn)
        seen_a.add(an)
        per_page[p] = per_page.get(p, 0) + 1
        by_category[category].append(row)
        overflow.append(row)

    kept: list[dict] = []
    used_ids: set[int] = set()
    for category, target in category_targets.items():
        bucket = by_category.get(category, [])
        take = min(target, len(bucket))
        for row in bucket[:take]:
            kept.append(row)
            used_ids.add(id(row))

    if len(kept) < target_size:
        for row in overflow:
            if id(row) in used_ids:
                continue
            kept.append(row)
            if len(kept) >= target_size:
                break

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter synthetic QA candidates into silver set.")
    parser.add_argument("--in-file", type=Path, default=Path("data/processed/qa_small.jsonl"))
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("data/processed/qa_silver_filtered.jsonl"))
    parser.add_argument("--target-size", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-page", type=int, default=4)
    parser.add_argument("--silver-pages-file", type=Path, default=Path("data/processed/silver_pages.txt"))
    args = parser.parse_args()

    logger = setup_logging("filter_qa_candidates")
    rows = read_jsonl(args.in_file)
    docs = read_jsonl(args.docs_file)
    if not rows:
        logger.warning("No candidates found in %s", args.in_file)
        return
    if not docs:
        logger.warning("No docs found in %s", args.docs_file)
        return

    valid_chunk_ids = {r.get("id") for r in docs if r.get("id")}
    allowed_pages: set[str] | None = None
    if args.silver_pages_file.exists():
        allowed_pages = {
            line.strip()
            for line in args.silver_pages_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        logger.info("Loaded silver pages allow-list: %d pages", len(allowed_pages))
    else:
        logger.warning("No silver pages file found: %s (page filtering disabled)", args.silver_pages_file)

    kept = filter_candidates(
        rows,
        valid_chunk_ids=valid_chunk_ids,
        target_size=args.target_size,
        seed=args.seed,
        max_per_page=args.max_per_page,
        allowed_pages=allowed_pages,
    )
    write_jsonl(args.out_file, kept)
    by_cat: dict[str, int] = {}
    for row in kept:
        c = row.get("category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    logger.info("Filtered silver set: %d -> %d rows (%s)", len(rows), len(kept), args.out_file)
    logger.info("Category distribution: %s", by_cat)


if __name__ == "__main__":
    main()

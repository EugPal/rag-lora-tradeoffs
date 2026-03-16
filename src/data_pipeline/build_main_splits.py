from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.data_pipeline.dataset_utils import allocate_category_targets, source_page_id
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def stratified_silver(rows: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("category", "endpoints_routing")].append(row)
    for key in buckets:
        rng.shuffle(buckets[key])

    targets = allocate_category_targets(size)
    picked: list[dict] = []
    used_ids: set[int] = set()
    for cat, target in targets.items():
        bucket = buckets.get(cat, [])
        take = min(target, len(bucket))
        for row in bucket[:take]:
            picked.append(row)
            used_ids.add(id(row))

    if len(picked) < size:
        rest = rows[:]
        rng.shuffle(rest)
        for row in rest:
            if id(row) in used_ids:
                continue
            picked.append(row)
            if len(picked) >= size:
                break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed gold/silver splits for main experiments.")
    parser.add_argument("--gold-file", type=Path, default=Path("data/processed/qa_gold_full.jsonl"))
    parser.add_argument("--silver-file", type=Path, default=Path("data/processed/qa_silver_filtered.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--gold-test-size", type=int, default=100)
    parser.add_argument("--gold-val-size", type=int, default=20)
    parser.add_argument("--silver-train-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-page-disjoint", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("build_main_splits")
    gold_rows = read_jsonl(args.gold_file)
    silver_rows = read_jsonl(args.silver_file)
    if not gold_rows:
        logger.warning("No gold rows found at %s", args.gold_file)
        return
    if not silver_rows:
        logger.warning("No silver rows found at %s", args.silver_file)
        return

    rng = random.Random(args.seed)
    gold = gold_rows[:]
    silver = silver_rows[:]
    rng.shuffle(gold)
    rng.shuffle(silver)

    need_gold = args.gold_test_size + args.gold_val_size
    if len(gold) < need_gold:
        logger.warning("Gold rows are less than requested split size: %d < %d", len(gold), need_gold)
    if len(silver) < args.silver_train_size:
        logger.warning(
            "Silver rows are less than requested train size: %d < %d",
            len(silver),
            args.silver_train_size,
        )

    gold_val = gold[: args.gold_val_size]
    gold_test = gold[args.gold_val_size : args.gold_val_size + args.gold_test_size]
    silver_train = stratified_silver(silver, args.silver_train_size, args.seed)
    silver_train_ids = {id(r) for r in silver_train}
    silver_extra = [r for r in silver if id(r) not in silver_train_ids]

    gold_pages = {
        row.get("source_page") or source_page_id(row.get("source_chunk"))
        for row in (gold_val + gold_test)
    }
    silver_pages = {
        row.get("source_page") or source_page_id(row.get("source_chunk"))
        for row in silver_train
    }
    overlap = {p for p in gold_pages if p and p in silver_pages}
    if overlap:
        msg = f"Page overlap detected between gold and silver splits: {len(overlap)} pages"
        if args.strict_page_disjoint:
            raise ValueError(msg)
        logger.warning(msg)

    out = args.out_dir
    write_jsonl(out / "qa_gold_val.jsonl", gold_val)
    write_jsonl(out / "qa_gold_test.jsonl", gold_test)
    write_jsonl(out / "qa_silver_train.jsonl", silver_train)
    write_jsonl(out / "qa_silver_extra.jsonl", silver_extra)

    logger.info(
        "Wrote splits: gold_val=%d gold_test=%d silver_train=%d silver_extra=%d",
        len(gold_val),
        len(gold_test),
        len(silver_train),
        len(silver_extra),
    )
    if silver_train:
        by_cat: dict[str, int] = {}
        for row in silver_train:
            c = row.get("category", "unknown")
            by_cat[c] = by_cat.get(c, 0) + 1
        logger.info("Silver train category distribution: %s", by_cat)


if __name__ == "__main__":
    main()

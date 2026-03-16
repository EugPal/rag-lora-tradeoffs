from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Split real-user QA into train/val/test.")
    ap.add_argument("--in-file", type=Path, default=Path("data/processed/qa_real_user_full.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--train-size", type=int, default=120)
    ap.add_argument("--val-size", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger = setup_logging("build_real_user_splits")
    rows = read_jsonl(args.in_file)
    if not rows:
        logger.warning("No rows found in %s", args.in_file)
        return

    rng = random.Random(args.seed)
    items = rows[:]
    rng.shuffle(items)

    total_req = args.train_size + args.val_size + args.test_size
    n_total = len(items)
    if n_total >= total_req:
        n_train = args.train_size
        n_val = args.val_size
        n_test = args.test_size
    else:
        # Scale requested sizes proportionally when dataset is smaller than requested.
        if total_req <= 0:
            n_train = n_val = n_test = 0
        else:
            train_ratio = args.train_size / total_req
            val_ratio = args.val_size / total_req
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val
            if n_total >= 3:
                n_train = max(n_train, 1)
                n_val = max(n_val, 1)
                n_test = max(n_total - n_train - n_val, 1)
        logger.warning(
            "Real-user rows less than requested total: %d < %d; scaled splits to train=%d val=%d test=%d",
            n_total,
            total_req,
            n_train,
            n_val,
            n_test,
        )

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]

    out = args.out_dir
    write_jsonl(out / "qa_real_user_train.jsonl", train)
    write_jsonl(out / "qa_real_user_val.jsonl", val)
    write_jsonl(out / "qa_real_user_test.jsonl", test)
    logger.info("Wrote real-user splits: train=%d val=%d test=%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()

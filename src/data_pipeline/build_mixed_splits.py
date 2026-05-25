from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data_pipeline.dataset_utils import normalize
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def merge_unique(base: list[dict], extra: list[dict], seed: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in base + extra:
        q = normalize(row.get("question", ""))
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(row)
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build mixed train/eval/test splits with real-user QA.")
    ap.add_argument("--silver-train", type=Path, default=Path("data/processed/qa_silver_train.jsonl"))
    ap.add_argument("--gold-val", type=Path, default=Path("data/processed/qa_gold_val.jsonl"))
    ap.add_argument("--gold-test", type=Path, default=Path("data/processed/qa_gold_test.jsonl"))
    ap.add_argument("--real-train", type=Path, default=Path("data/processed/qa_real_user_train.jsonl"))
    ap.add_argument("--real-val", type=Path, default=Path("data/processed/qa_real_user_val.jsonl"))
    ap.add_argument("--real-test", type=Path, default=Path("data/processed/qa_real_user_test.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger = setup_logging("build_mixed_splits")
    train = merge_unique(read_jsonl(args.silver_train), read_jsonl(args.real_train), args.seed)
    eval_rows = merge_unique(read_jsonl(args.gold_val), read_jsonl(args.real_val), args.seed)
    test = merge_unique(read_jsonl(args.gold_test), read_jsonl(args.real_test), args.seed)

    out = args.out_dir
    write_jsonl(out / "qa_train_mixed.jsonl", train)
    write_jsonl(out / "qa_eval_mixed.jsonl", eval_rows)
    write_jsonl(out / "qa_test_mixed.jsonl", test)
    logger.info("Wrote mixed splits: train=%d eval=%d test=%d", len(train), len(eval_rows), len(test))


if __name__ == "__main__":
    main()

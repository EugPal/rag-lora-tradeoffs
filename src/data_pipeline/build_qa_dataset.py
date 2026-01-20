from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def split_dataset(rows: list[dict], seed: int, ratios: tuple[float, float, float]):
    random.seed(seed)
    rows = rows[:]
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


BAD_SUBSTRINGS = [
    "back to top",
    "next",
    "previous",
    "waiting list",
    "subscribe",
    "initializing search",
    "http://",
    "https://",
]


def pick_answer(sentences: list[str]) -> str | None:
    """
    Pick a short, factual, answer-like sentence. This keeps the QA dataset
    from being dominated by navigation/boilerplate artifacts.
    """
    for s in sentences:
        low = s.lower()
        if any(b in low for b in BAD_SUBSTRINGS):
            continue
        words = s.split()
        if 8 <= len(words) <= 25:
            return " ".join(words[:25])
    return None


def build_qa_from_docs(rows: list[dict], max_qa: int) -> list[dict]:
    qa_rows = []
    pbar = tqdm(rows, desc="build_qa_dataset", unit="chunk", disable=not sys.stderr.isatty())
    for row in pbar:
        sentences = sentence_split(row.get("text", ""))
        if not sentences:
            continue
        answer = pick_answer(sentences)
        if not answer:
            continue
        anchor = " ".join(answer.split()[:6])
        question = f"According to the documentation, what does it say about {anchor}?"
        qa_rows.append(
            {
                "id": f"qa-{row.get('id', len(qa_rows))}",
                "question": question,
                "answer": answer,
                "source_chunk": row.get("id"),
            }
        )
        if len(qa_rows) >= max_qa:
            break
    return qa_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA splits from docs.jsonl.")
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-qa", type=int, default=20)
    args = parser.parse_args()

    logger = setup_logging("build_qa_dataset")
    rows = read_jsonl(args.docs_file)
    if not rows:
        logger.warning("No docs found at %s", args.docs_file)
        return

    qa_rows = build_qa_from_docs(rows, args.max_qa)
    if not qa_rows:
        logger.warning("No QA pairs generated from %s", args.docs_file)
        return

    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    train, val, test = split_dataset(qa_rows, args.seed, ratios)

    write_jsonl(args.out_dir / "qa_small.jsonl", qa_rows)
    write_jsonl(args.out_dir / "qa_train.jsonl", train)
    write_jsonl(args.out_dir / "qa_val.jsonl", val)
    write_jsonl(args.out_dir / "qa_test.jsonl", test)
    logger.info(
        "Wrote qa_small=%d train=%d val=%d test=%d",
        len(qa_rows),
        len(train),
        len(val),
        len(test),
    )


if __name__ == "__main__":
    main()

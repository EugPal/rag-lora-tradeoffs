from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.rag.generator import BaseGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "it",
    "you",
    "your",
    "we",
    "our",
    "from",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def token_set(text: str) -> set[str]:
    tokens = [t for t in normalize(text).split() if len(t) > 2 and t not in STOPWORDS]
    return set(tokens)


def is_hit(answer: str, contexts: list[str], min_overlap: int) -> bool:
    norm_answer = normalize(answer)
    if not norm_answer:
        return False
    answer_tokens = token_set(answer)
    for ctx in contexts:
        norm_ctx = normalize(ctx)
        if norm_answer in norm_ctx:
            return True
        if answer_tokens:
            overlap = answer_tokens.intersection(token_set(ctx))
            if len(overlap) >= min_overlap:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute retrieval hit rate.")
    parser.add_argument("--qa-file", type=Path, default=Path("data/processed/qa_small.jsonl"))
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-overlap", type=int, default=3)
    args = parser.parse_args()

    logger = setup_logging("retrieval_hit_rate")
    qa_rows = read_jsonl(args.qa_file)
    if not qa_rows:
        logger.warning("No QA rows found at %s", args.qa_file)
        return

    config = RagConfig(docs_path=args.docs_file, top_k=args.top_k)
    pipeline = RagPipeline(config, generator=BaseGenerator())

    hits = 0
    for row in qa_rows:
        contexts, _results = pipeline.retrieve(row["question"])
        if is_hit(row["answer"], contexts, args.min_overlap):
            hits += 1

    hit_rate = hits / len(qa_rows)
    logger.info("Retrieval hit rate: %.2f (%d/%d)", hit_rate, hits, len(qa_rows))


if __name__ == "__main__":
    main()

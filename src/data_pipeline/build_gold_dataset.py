from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from src.data_pipeline.dataset_utils import infer_section, normalize, source_page_id
from src.rag.generator import BaseGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


TEMPLATE_PREFIX = "according to the documentation, what does it say about "
BAD_SUBSTRINGS = ["<span", "<font", "javascript:", "mailto:", "back to top"]

def sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if p.strip()])


def clean_question(question: str) -> str:
    q = question.strip()
    low = q.lower()
    if low.startswith(TEMPLATE_PREFIX) and q.endswith("?"):
        tail = q[len("According to the documentation, what does it say about ") : -1].strip()
        return f"What does the FastAPI documentation say about {tail}?"
    return q

def is_strict_gold_candidate(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    qa = f"{question}\n{answer}".lower()
    if any(b in qa for b in BAD_SUBSTRINGS):
        return False
    words = answer.split()
    if not (10 <= len(words) <= 50):
        return False
    if sentence_count(answer) > 3:
        return False
    qn = normalize(question)
    an = normalize(answer)
    if qn == an or (an and an in qn):
        return False
    return True


def source_chunk_from_retrieval(pipeline: RagPipeline, question: str) -> str | None:
    _contexts, scored = pipeline.retrieve(question)
    if not scored:
        return None
    return scored[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict gold dataset from seed + candidate pool.")
    parser.add_argument("--seed-file", type=Path, default=Path("data/raw/qa_seed/seed.jsonl"))
    parser.add_argument("--candidates-file", type=Path, default=Path("data/processed/qa_small.jsonl"))
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--index-file", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings-file", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    parser.add_argument("--out-file", type=Path, default=Path("data/processed/qa_gold_full.jsonl"))
    parser.add_argument("--gold-pages-file", type=Path, default=Path("data/processed/gold_pages.txt"))
    parser.add_argument("--target-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-page", type=int, default=2)
    args = parser.parse_args()

    logger = setup_logging("build_gold_dataset")
    seed_rows = read_jsonl(args.seed_file)
    candidate_rows = read_jsonl(args.candidates_file)
    docs = read_jsonl(args.docs_file)
    if not docs:
        logger.warning("No docs found at %s", args.docs_file)
        return
    valid_ids = {row.get("id") for row in docs if row.get("id")}
    gold_pages: set[str] | None = None
    if args.gold_pages_file.exists():
        gold_pages = {
            line.strip()
            for line in args.gold_pages_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        logger.info("Loaded gold pages allow-list: %d pages", len(gold_pages))
    else:
        logger.warning("No gold pages file found: %s (page filtering disabled)", args.gold_pages_file)

    pipeline = RagPipeline(
        RagConfig(
            docs_path=args.docs_file,
            index_path=args.index_file,
            embeddings_path=args.embeddings_file,
            top_k=8,
            use_hf_generator=False,
        ),
        generator=BaseGenerator(),
    )

    rows_out: list[dict] = []
    seen_q: set[str] = set()
    seen_a: set[str] = set()
    per_page: dict[str, int] = {}

    # 1) keep manually seeded QA first, auto-map source chunk via retrieval.
    for i, row in enumerate(seed_rows):
        q = clean_question(row.get("question", ""))
        a = row.get("answer", "").strip()
        if not is_strict_gold_candidate(q, a):
            continue
        source_chunk = row.get("source_chunk") or source_chunk_from_retrieval(pipeline, q)
        if not source_chunk or source_chunk not in valid_ids:
            continue
        p = source_page_id(source_chunk)
        if not p:
            continue
        if gold_pages is not None and p not in gold_pages:
            continue
        if per_page.get(p, 0) >= args.max_per_page:
            continue
        qn = normalize(q)
        an = normalize(a)
        if qn in seen_q or an in seen_a:
            continue
        seen_q.add(qn)
        seen_a.add(an)
        per_page[p] = per_page.get(p, 0) + 1
        rows_out.append(
            {
                "id": f"gold-seed-{i}",
                "question": q,
                "answer": a,
                "source_chunk": source_chunk,
                "source_page": p,
                "section": infer_section(p),
                "provenance": "manual_seed",
            }
        )

    # 2) fill the rest from strict candidate pool (gold-reserved pages only).
    rng = random.Random(args.seed)
    candidates = candidate_rows[:]
    rng.shuffle(candidates)

    for row in candidates:
        if len(rows_out) >= args.target_size:
            break
        q = clean_question(row.get("question", ""))
        a = row.get("answer", "").strip()
        source_chunk = row.get("source_chunk")
        if not source_chunk or source_chunk not in valid_ids:
            continue
        if not is_strict_gold_candidate(q, a):
            continue
        p = source_page_id(source_chunk)
        if not p:
            continue
        if gold_pages is not None and p not in gold_pages:
            continue
        if per_page.get(p, 0) >= args.max_per_page:
            continue
        qn = normalize(q)
        an = normalize(a)
        if qn in seen_q or an in seen_a:
            continue
        seen_q.add(qn)
        seen_a.add(an)
        per_page[p] = per_page.get(p, 0) + 1
        rows_out.append(
            {
                "id": f"gold-{len(rows_out)}",
                "question": q,
                "answer": a,
                "source_chunk": source_chunk,
                "source_page": p,
                "section": infer_section(p),
                "provenance": "candidate_curated",
            }
        )

    if len(rows_out) < args.target_size:
        logger.warning(
            "Gold target not reached: requested %d, built %d", args.target_size, len(rows_out)
        )

    write_jsonl(args.out_file, rows_out[: args.target_size])
    logger.info("Wrote %d rows to %s", min(len(rows_out), args.target_size), args.out_file)


if __name__ == "__main__":
    main()

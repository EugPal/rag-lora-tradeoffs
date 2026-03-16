from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from src.rag.index import load_index
from src.utils.io_utils import read_jsonl, write_jsonl


def _base_chunk_id(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    return re.sub(r"-\d+$", "", chunk_id)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _contains_answer(answer: str, chunk_text: str) -> bool:
    a = _normalize(answer)
    c = _normalize(chunk_text)
    if not a or not c:
        return False
    return a in c


def _token_overlap_count(a_text: str, b_text: str) -> int:
    a = set(_normalize(a_text).split())
    b = set(_normalize(b_text).split())
    if not a or not b:
        return 0
    return len(a.intersection(b))


def _chunk_suffix(chunk_id: str | None) -> tuple[str | None, int | None]:
    if not chunk_id:
        return None, None
    m = re.match(r"^(.*)-(\d+)$", chunk_id)
    if not m:
        return chunk_id, None
    return m.group(1), int(m.group(2))


def _maybe_neighbor_positives(
    answer: str,
    source_chunk: str | None,
    doc_lookup: dict[str, str],
    *,
    max_neighbors: int = 1,
) -> list[str]:
    page, idx = _chunk_suffix(source_chunk)
    if page is None or idx is None or source_chunk is None:
        return []
    neighbors: list[str] = []
    for step in (-1, 1):
        cand_id = f"{page}-{idx + step}"
        if cand_id not in doc_lookup:
            continue
        cand_text = doc_lookup[cand_id]
        if _contains_answer(answer, cand_text) or _token_overlap_count(answer, cand_text) >= 4:
            neighbors.append(cand_id)
        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def _sample_without_replacement(
    candidates: list[str],
    k: int,
    rng: random.Random,
    blocked: set[str] | None = None,
) -> list[str]:
    if k <= 0:
        return []
    blocked = blocked or set()
    pool = [x for x in candidates if x not in blocked]
    if not pool:
        return []
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def _clip_negative_lists(
    hard_negs: list[str],
    in_page_negs: list[str],
    random_negs: list[str],
    *,
    budget: int,
) -> tuple[list[str], list[str], list[str]]:
    if budget <= 0:
        return [], [], []
    out_hard = hard_negs[:budget]
    budget -= len(out_hard)
    out_in_page = in_page_negs[:budget]
    budget -= len(out_in_page)
    out_random = random_negs[:budget]
    return out_hard, out_in_page, out_random


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build retrieval supervision with positive/negative chunk lists."
    )
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    ap.add_argument(
        "--index-file",
        type=Path,
        default=Path("data/embeddings/docs_embeddings.faiss"),
    )
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--stats-file", type=Path, default=None)
    ap.add_argument("--top-n", type=int, default=50, help="Retriever depth for hard-negative mining.")
    ap.add_argument("--hard-negatives", type=int, default=3)
    ap.add_argument("--in-page-negatives", type=int, default=1)
    ap.add_argument("--random-negatives", type=int, default=1)
    ap.add_argument("--add-neighbor-positive", action="store_true")
    ap.add_argument("--neighbor-max", type=int, default=1)
    ap.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=5,
        help="Cap total negatives to this multiplier of positive chunk count.",
    )
    ap.add_argument("--max-items", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa_rows = read_jsonl(args.qa_file)
    if args.max_items > 0:
        qa_rows = qa_rows[: args.max_items]
    docs = read_jsonl(args.docs_file)
    if not qa_rows:
        raise ValueError(f"No QA rows found in {args.qa_file}")
    if not docs:
        raise ValueError(f"No docs found in {args.docs_file}")

    doc_lookup = {row["id"]: row.get("text", "") for row in docs if row.get("id")}
    all_doc_ids = list(doc_lookup.keys())
    page_to_chunk_ids: dict[str, list[str]] = {}
    for chunk_id in all_doc_ids:
        page = _base_chunk_id(chunk_id)
        if page is None:
            continue
        page_to_chunk_ids.setdefault(page, []).append(chunk_id)

    index = load_index(args.index_file)
    missing_source_chunk = 0
    no_positive = 0
    total_hard = 0
    total_in_page = 0
    total_random = 0

    out_rows = []
    for i, row in enumerate(qa_rows):
        qid = row.get("id") or f"row-{i}"
        question = row.get("question", "")
        answer = row.get("answer", "")
        source_chunk = row.get("source_chunk")
        source_page = row.get("source_page") or _base_chunk_id(source_chunk)

        rng = random.Random(f"{args.seed}:{qid}")

        positives: list[str] = []
        if source_chunk and source_chunk in doc_lookup:
            positives.append(source_chunk)
        elif source_chunk:
            missing_source_chunk += 1

        # Recover positives by exact-answer containment, if possible.
        if not positives and source_page and source_page in page_to_chunk_ids:
            page_candidates = page_to_chunk_ids[source_page]
            positives = [cid for cid in page_candidates if _contains_answer(answer, doc_lookup[cid])]
        if args.add_neighbor_positive and source_chunk:
            neighbor_pos = _maybe_neighbor_positives(
                answer,
                source_chunk,
                doc_lookup,
                max_neighbors=max(0, args.neighbor_max),
            )
            for cid in neighbor_pos:
                if cid not in positives:
                    positives.append(cid)

        if not positives:
            no_positive += 1

        retrieved = [doc_id for doc_id, _ in index.search(question, top_k=args.top_n)]
        retrieved = [doc_id for doc_id in retrieved if doc_id in doc_lookup]

        blocked = set(positives)
        blocked.update(
            cid for cid in retrieved if _contains_answer(answer, doc_lookup.get(cid, ""))
        )

        hard_pool = [cid for cid in retrieved if cid not in blocked]
        hard_negs = _sample_without_replacement(hard_pool, args.hard_negatives, rng)
        blocked.update(hard_negs)

        in_page_pool: list[str] = []
        if source_page and source_page in page_to_chunk_ids:
            in_page_pool = [cid for cid in page_to_chunk_ids[source_page] if cid not in blocked]
        in_page_negs = _sample_without_replacement(in_page_pool, args.in_page_negatives, rng)
        blocked.update(in_page_negs)

        random_negs = _sample_without_replacement(all_doc_ids, args.random_negatives, rng, blocked=blocked)
        blocked.update(random_negs)

        neg_budget = max(0, len(positives) * max(1, int(args.max_negatives_per_positive)))
        hard_negs, in_page_negs, random_negs = _clip_negative_lists(
            hard_negs,
            in_page_negs,
            random_negs,
            budget=neg_budget,
        )

        negatives = hard_negs + in_page_negs + random_negs
        total_hard += len(hard_negs)
        total_in_page += len(in_page_negs)
        total_random += len(random_negs)

        out_rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "source_chunk": source_chunk,
                "source_page": source_page,
                "positive_chunks": positives,
                "negative_chunks": negatives,
                "negative_types": {
                    "hard": hard_negs,
                    "in_page": in_page_negs,
                    "random": random_negs,
                },
                "retrieved_top_n": retrieved,
            }
        )

    write_jsonl(args.out_file, out_rows)

    stats = {
        "input_rows": len(qa_rows),
        "output_rows": len(out_rows),
        "missing_source_chunk": missing_source_chunk,
        "rows_without_positive": no_positive,
        "avg_positive_per_row": float(np.mean([len(r["positive_chunks"]) for r in out_rows])),
        "avg_negative_per_row": float(np.mean([len(r["negative_chunks"]) for r in out_rows])),
        "avg_hard_negative_per_row": total_hard / max(1, len(out_rows)),
        "avg_in_page_negative_per_row": total_in_page / max(1, len(out_rows)),
        "avg_random_negative_per_row": total_random / max(1, len(out_rows)),
        "top_n": args.top_n,
        "hard_negatives": args.hard_negatives,
        "in_page_negatives": args.in_page_negatives,
        "random_negatives": args.random_negatives,
    }
    if args.stats_file:
        args.stats_file.parent.mkdir(parents=True, exist_ok=True)
        args.stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    else:
        print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

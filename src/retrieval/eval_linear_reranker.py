from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.rag.embeddings import text_to_embedding
from src.retrieval.linear_reranker import cosine_similarity, load_model
from src.utils.io_utils import read_jsonl


def _ids_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".ids.json")


def _load_doc_embedding_lookup(index_path: Path, embeddings_path: Path) -> dict[str, np.ndarray]:
    ids = json.loads(_ids_path(index_path).read_text(encoding="utf-8"))
    vectors = np.load(embeddings_path)
    if len(ids) != len(vectors):
        raise ValueError("ids/vectors mismatch for embeddings lookup")
    return {doc_id: vectors[i].astype(np.float32) for i, doc_id in enumerate(ids)}


def _recall_at_k(ordered: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    return 1.0 if any(cid in positives for cid in ordered[:k]) else 0.0


def _mrr(ordered: list[str], positives: set[str]) -> float:
    if not positives:
        return 0.0
    for i, cid in enumerate(ordered, start=1):
        if cid in positives:
            return 1.0 / i
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate linear reranker on supervision rows.")
    ap.add_argument("--supervision-file", type=Path, required=True)
    ap.add_argument("--model-file", type=Path, required=True)
    ap.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    ap.add_argument("--index-file", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    ap.add_argument("--embeddings-file", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--max-items", type=int, default=-1)
    args = ap.parse_args()

    sup_rows = read_jsonl(args.supervision_file)
    if args.max_items > 0:
        sup_rows = sup_rows[: args.max_items]
    docs = read_jsonl(args.docs_file)
    doc_lookup = {row["id"]: row.get("text", "") for row in docs if row.get("id")}
    emb_lookup = _load_doc_embedding_lookup(args.index_file, args.embeddings_file)
    model = load_model(args.model_file)

    q_cache: dict[str, np.ndarray] = {}
    rows_eval = []
    ks = [1, 3, 5, 10]

    for row in sup_rows:
        qid = row.get("id")
        query = row.get("question", "")
        positives = set(row.get("positive_chunks") or [])
        baseline_order = [cid for cid in (row.get("retrieved_top_n") or []) if cid in doc_lookup]
        if not query or not positives or not baseline_order:
            continue
        q_emb = q_cache.get(query)
        if q_emb is None:
            q_emb = text_to_embedding(query).astype(np.float32)
            q_cache[query] = q_emb

        scored = []
        for rank0, cid in enumerate(baseline_order):
            c_emb = emb_lookup.get(cid)
            if c_emb is None:
                continue
            score = model.score(
                query=query,
                chunk_text=doc_lookup[cid],
                base_rank_1_based=rank0 + 1,
                cosine_sim=cosine_similarity(q_emb, c_emb),
            )
            scored.append((cid, float(score)))
        if not scored:
            continue
        reranked_order = [cid for cid, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

        entry = {
            "id": qid,
            "baseline_mrr": _mrr(baseline_order, positives),
            "rerank_mrr": _mrr(reranked_order, positives),
        }
        for k in ks:
            entry[f"baseline_recall@{k}"] = _recall_at_k(baseline_order, positives, k)
            entry[f"rerank_recall@{k}"] = _recall_at_k(reranked_order, positives, k)
        rows_eval.append(entry)

    if not rows_eval:
        raise ValueError("No evaluable rows found for reranker evaluation.")

    agg = {
        "rows": len(rows_eval),
        "baseline_mrr": float(np.mean([r["baseline_mrr"] for r in rows_eval])),
        "rerank_mrr": float(np.mean([r["rerank_mrr"] for r in rows_eval])),
    }
    for k in ks:
        agg[f"baseline_recall@{k}"] = float(np.mean([r[f"baseline_recall@{k}"] for r in rows_eval]))
        agg[f"rerank_recall@{k}"] = float(np.mean([r[f"rerank_recall@{k}"] for r in rows_eval]))

    payload = {"summary": agg, "per_row": rows_eval[:200]}
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()


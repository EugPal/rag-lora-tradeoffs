from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.rag.embeddings import text_to_embedding
from src.retrieval.linear_reranker import (
    LinearRerankerModel,
    cosine_similarity,
    lexical_overlap_ratio,
    rank_feature,
    save_model,
)
from src.utils.io_utils import read_jsonl


def _ids_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".ids.json")


def _load_doc_embedding_lookup(index_path: Path, embeddings_path: Path) -> dict[str, np.ndarray]:
    ids = json.loads(_ids_path(index_path).read_text(encoding="utf-8"))
    vectors = np.load(embeddings_path)
    if len(ids) != len(vectors):
        raise ValueError("ids/vectors mismatch for embeddings lookup")
    return {doc_id: vectors[i].astype(np.float32) for i, doc_id in enumerate(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train lightweight linear reranker from supervision data.")
    ap.add_argument("--supervision-file", type=Path, required=True)
    ap.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    ap.add_argument("--index-file", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    ap.add_argument("--embeddings-file", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--max-items", type=int, default=-1)
    args = ap.parse_args()

    sup_rows = read_jsonl(args.supervision_file)
    docs = read_jsonl(args.docs_file)
    if args.max_items > 0:
        sup_rows = sup_rows[: args.max_items]
    if not sup_rows:
        raise ValueError("No supervision rows to train reranker.")
    doc_lookup = {row["id"]: row.get("text", "") for row in docs if row.get("id")}
    emb_lookup = _load_doc_embedding_lookup(args.index_file, args.embeddings_file)

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    q_cache: dict[str, np.ndarray] = {}

    for row in sup_rows:
        query = row.get("question", "")
        positives = set(row.get("positive_chunks") or [])
        candidates = row.get("retrieved_top_n") or []
        if not query or not positives or not candidates:
            continue
        q_emb = q_cache.get(query)
        if q_emb is None:
            q_emb = text_to_embedding(query).astype(np.float32)
            q_cache[query] = q_emb
        for rank0, chunk_id in enumerate(candidates):
            if chunk_id not in doc_lookup:
                continue
            c_emb = emb_lookup.get(chunk_id)
            if c_emb is None:
                continue
            chunk_text = doc_lookup[chunk_id]
            feats = [
                rank_feature(rank0 + 1),
                lexical_overlap_ratio(query, chunk_text),
                cosine_similarity(q_emb, c_emb),
            ]
            label = 1 if chunk_id in positives else 0
            x_rows.append(feats)
            y_rows.append(label)

    if not x_rows:
        raise ValueError("No trainable pairs were built for reranker.")
    if len(set(y_rows)) < 2:
        raise ValueError("Need both positive and negative labels for reranker training.")

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x, y)

    model = LinearRerankerModel(
        feature_names=["base_rank", "lex_overlap", "cosine_sim"],
        weights=[float(v) for v in clf.coef_[0].tolist()],
        bias=float(clf.intercept_[0]),
        metadata={
            "pairs": int(len(y_rows)),
            "positives": int(np.sum(y)),
            "negatives": int(len(y) - np.sum(y)),
            "source_supervision_file": str(args.supervision_file),
        },
    )
    save_model(args.out_file, model)
    print(
        json.dumps(
            {
                "saved_model": str(args.out_file),
                "pairs": int(len(y_rows)),
                "positives": int(np.sum(y)),
                "negatives": int(len(y) - np.sum(y)),
                "weights": model.weights,
                "bias": model.bias,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


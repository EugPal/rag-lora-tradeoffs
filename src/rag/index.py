from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.embeddings import (
    embed_docs,
    get_retriever_embedding_dim,
    get_retriever_model_name,
    text_to_embedding,
)
from src.utils.io_utils import ensure_dir


class FaissIndex:
    def __init__(self, index: faiss.Index, ids: list[str]) -> None:
        self.index = index
        self.ids = ids

    def search(self, query: str, top_k: int = 4) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q = text_to_embedding(query).reshape(1, -1)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx >= len(self.ids):
                continue
            results.append((self.ids[idx], float(score)))
        return results


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".ids.json")


def _meta_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def build_index(
    docs_path: Path,
    out_index_path: Path,
    out_embeddings_path: Path,
    dim: int | None = None,
) -> FaissIndex:
    embeddings, ids = embed_docs(docs_path)
    ensure_dir(out_index_path.parent)
    ensure_dir(out_embeddings_path.parent)
    np.save(out_embeddings_path, embeddings)
    if embeddings.size:
        index_dim = embeddings.shape[1]
    else:
        index_dim = int(dim or get_retriever_embedding_dim())
    index = faiss.IndexFlatIP(index_dim)
    if embeddings.size:
        index.add(embeddings)
    faiss.write_index(index, str(out_index_path))

    # Persist ids alongside the FAISS index to prevent silent idв†”vector order mismatches.
    _ids_path(out_index_path).write_text(
        json.dumps(ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _meta_path(out_index_path).write_text(
        json.dumps(
            {
                "docs_path": str(docs_path),
                "docs_sha256": _sha256_file(docs_path),
                "embedding_model": get_retriever_model_name(),
                "embedding_dim": int(index_dim),
                "ntotal": int(index.ntotal),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FaissIndex(index=index, ids=ids)


def load_index(index_path: Path) -> FaissIndex:
    index = faiss.read_index(str(index_path))
    ids_file = _ids_path(index_path)
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing ids file for index: {ids_file}")
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    if index.ntotal != len(ids):
        raise ValueError("Index/id mismatch; rebuild index.")

    # Guard against stale index built with a different embedding model.
    meta_file = _meta_path(index_path)
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        built_with = str(meta.get("embedding_model") or "")
        current = get_retriever_model_name()
        if built_with and built_with != current:
            raise ValueError(
                f"Index embedding model mismatch: built_with={built_with}, current={current}. Rebuild required."
            )

    expected_dim = get_retriever_embedding_dim()
    if int(index.d) != int(expected_dim):
        raise ValueError(
            f"Index dimension mismatch: index.d={index.d}, expected={expected_dim}. Rebuild required."
        )

    return FaissIndex(index=index, ids=ids)

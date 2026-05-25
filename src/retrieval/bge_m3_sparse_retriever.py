from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.utils.io_utils import read_jsonl


def _normalize_weights(raw: dict) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in (raw or {}).items():
        try:
            score = float(value)
        except Exception:
            continue
        if score <= 0.0:
            continue
        normalized[str(key)] = score
    return normalized


class BGEM3SparseRetriever:
    def __init__(
        self,
        docs_path: Path,
        sparse_index_path: Path,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
    ) -> None:
        self.docs_path = docs_path
        self.sparse_index_path = sparse_index_path
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None
        self._doc_ids: list[str] = []
        self._postings: dict[str, list[tuple[int, float]]] = {}
        self._load_or_build()

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            raise RuntimeError(
                "Native BGE-M3 sparse retrieval requires FlagEmbedding backend. "
                "Install compatible FlagEmbedding (e.g. 1.3.5)."
            ) from exc
        self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def _encode_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        model = self._get_model()
        output = model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = output.get("lexical_weights", [])
        return [_normalize_weights(item) for item in lexical_weights]

    def _load_or_build(self) -> None:
        if self.sparse_index_path.exists():
            payload = json.loads(self.sparse_index_path.read_text(encoding="utf-8"))
            self._doc_ids = [str(x) for x in payload.get("doc_ids", [])]
            raw_postings = payload.get("postings", {})
            self._postings = {
                str(term): [(int(i), float(v)) for i, v in pairs]
                for term, pairs in raw_postings.items()
            }
            if self._doc_ids and self._postings:
                return
        self._build_and_save()

    def _build_and_save(self) -> None:
        rows = read_jsonl(self.docs_path)
        self._doc_ids = [str(r["id"]) for r in rows]
        texts = [str(r.get("text", "")) for r in rows]
        sparse_vectors = self._encode_sparse(texts)
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx, weights in enumerate(sparse_vectors):
            for term, score in weights.items():
                postings[term].append((idx, score))
        self._postings = dict(postings)
        self.sparse_index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "doc_ids": self._doc_ids,
            "postings": self._postings,
        }
        self.sparse_index_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not query.strip():
            return []
        query_weights = self._encode_sparse([query])[0]
        scores: dict[int, float] = defaultdict(float)
        for term, q_weight in query_weights.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            for doc_idx, d_weight in postings:
                scores[doc_idx] += float(q_weight) * float(d_weight)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_ids[idx], float(score)) for idx, score in ranked]

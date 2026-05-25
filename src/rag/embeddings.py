from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.io_utils import read_jsonl


_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3"))
    return _MODEL


def text_to_embedding(text: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=sys.stderr.isatty(),
    )
    return vectors.astype(np.float32)


def embed_docs(docs_path: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_jsonl(docs_path)
    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    embeddings = embed_texts(texts)
    return embeddings, ids



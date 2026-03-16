from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.io_utils import read_jsonl


RETRIEVER_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
QUERY_INSTRUCTION = os.getenv(
    "RAG_QUERY_INSTRUCTION",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
)

_MODEL: SentenceTransformer | None = None


def get_retriever_model_name() -> str:
    return RETRIEVER_MODEL_NAME


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(RETRIEVER_MODEL_NAME)
    return _MODEL


def get_retriever_embedding_dim() -> int:
    model = get_model()
    dim = model.get_sentence_embedding_dimension()
    return int(dim) if dim is not None else 1024


def _encode_texts(texts: list[str], *, is_query: bool) -> np.ndarray:
    model = get_model()

    if is_query:
        # Qwen3 embeddings are instruction-aware; apply query prompt when possible.
        try:
            if getattr(model, "prompts", None) and "query" in model.prompts:
                vectors = model.encode(
                    texts,
                    prompt_name="query",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            else:
                vectors = model.encode(
                    texts,
                    prompt=QUERY_INSTRUCTION,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except TypeError:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
    else:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=sys.stderr.isatty(),
        )

    return vectors.astype(np.float32)


def text_to_embedding(text: str) -> np.ndarray:
    return _encode_texts([text], is_query=True)[0]


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, get_retriever_embedding_dim()), dtype=np.float32)
    return _encode_texts(texts, is_query=False)


def embed_docs(docs_path: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_jsonl(docs_path)
    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    embeddings = embed_texts(texts)
    return embeddings, ids

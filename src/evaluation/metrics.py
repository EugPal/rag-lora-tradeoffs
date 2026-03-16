from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score


def normalize(text: str) -> str:
    text = text.lower()
    # Keep word characters (including underscore) to avoid destroying identifiers like
    # `fastapi[standard]`, `path_params`, etc. Collapse punctuation to spaces.
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def f1_components(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = pred_counts & ref_counts
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def f1_score(prediction: str, reference: str) -> float:
    _precision, _recall, f1 = f1_components(prediction, reference)
    return f1


_EMBED_MODEL: Optional[SentenceTransformer] = None


def get_embed_model(model_name: str) -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None or _EMBED_MODEL.model_card_data.model_id != model_name:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def embedding_cosine(
    prediction: str,
    reference: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    model = get_embed_model(model_name)
    vectors = model.encode([prediction, reference], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.dot(vectors[0], vectors[1]))


def bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> list[float]:
    precision, recall, f1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    return [float(score) for score in f1]

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 2]


def lexical_overlap_ratio(query: str, chunk: str) -> float:
    q = set(normalize_tokens(query))
    c = set(normalize_tokens(chunk))
    if not q or not c:
        return 0.0
    return float(len(q.intersection(c)) / len(q))


def rank_feature(base_rank_1_based: int) -> float:
    return 1.0 / (1.0 + float(base_rank_1_based))


@dataclass
class LinearRerankerModel:
    feature_names: list[str]
    weights: list[float]
    bias: float
    metadata: dict

    def score_features(self, features: dict[str, float]) -> float:
        score = float(self.bias)
        for name, w in zip(self.feature_names, self.weights):
            score += float(w) * float(features.get(name, 0.0))
        return score

    def score(
        self,
        query: str,
        chunk_text: str,
        base_rank_1_based: int,
        cosine_sim: float,
    ) -> float:
        feats = {
            "base_rank": rank_feature(base_rank_1_based),
            "lex_overlap": lexical_overlap_ratio(query, chunk_text),
            "cosine_sim": float(cosine_sim),
        }
        return self.score_features(feats)

    def to_json_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "LinearRerankerModel":
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(x) for x in payload["weights"]],
            bias=float(payload["bias"]),
            metadata=dict(payload.get("metadata", {})),
        )


def save_model(path: Path, model: LinearRerankerModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json_dict(), indent=2), encoding="utf-8")


def load_model(path: Path) -> LinearRerankerModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LinearRerankerModel.from_json_dict(payload)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


@dataclass
class _DocStats:
    doc_id: str
    length: int
    tf: Counter[str]


class BM25Retriever:
    def __init__(
        self,
        docs: Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._docs: list[_DocStats] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl = 0.0

        total_len = 0
        for doc_id, text in docs:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self._docs.append(_DocStats(doc_id=doc_id, length=len(tokens), tf=tf))
            total_len += len(tokens)
            for term in tf.keys():
                self._df[term] += 1
        self._n_docs = len(self._docs)
        if self._n_docs > 0:
            self._avgdl = total_len / self._n_docs

    def _idf(self, term: str) -> float:
        # Standard BM25 idf with +1 smoothing to keep values positive.
        n_qi = self._df.get(term, 0)
        return math.log(1.0 + ((self._n_docs - n_qi + 0.5) / (n_qi + 0.5)))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._n_docs == 0 or top_k <= 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        q_unique = list(dict.fromkeys(q_terms))
        scores: list[tuple[str, float]] = []
        avgdl = self._avgdl if self._avgdl > 0 else 1.0
        for doc in self._docs:
            if doc.length == 0:
                continue
            score = 0.0
            for term in q_unique:
                f = doc.tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc.length / avgdl))
                score += idf * ((f * (self.k1 + 1.0)) / max(denom, 1e-9))
            if score > 0:
                scores.append((doc.doc_id, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


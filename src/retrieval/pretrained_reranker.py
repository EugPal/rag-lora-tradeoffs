from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PretrainedReranker:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, chunks: Sequence[str]) -> list[float]:
        if not chunks:
            return []
        scores: list[float] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = list(chunks[start : start + self.batch_size])
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            logits = self.model(**encoded).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                vals = logits[:, 0]
            elif logits.ndim == 2 and logits.shape[1] > 1:
                # Fallback for models with 2-class logits.
                vals = logits[:, -1]
            else:
                vals = logits.reshape(-1)
            scores.extend([float(x) for x in vals.detach().cpu().tolist()])
        return scores


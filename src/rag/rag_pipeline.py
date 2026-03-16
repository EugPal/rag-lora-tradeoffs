from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.rag.generator import BaseGenerator, GenerationConfig, HFGenerator
from src.rag.embeddings import text_to_embedding
from src.rag.index import build_index, load_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.linear_reranker import cosine_similarity, load_model
from src.retrieval.pretrained_reranker import PretrainedReranker
from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


@dataclass
class RagConfig:
    docs_path: Path = Path("data/processed/docs.jsonl")
    index_path: Path = Path("data/embeddings/docs_embeddings.faiss")
    embeddings_path: Path = Path("data/embeddings/docs_embeddings.npy")
    top_k: int = 8
    prompt_system: str = "You are a technical documentation assistant."
    use_hf_generator: bool = True
    use_4bit_generator: bool = True
    lora_adapter_dir: Path | None = None
    reranker_model_path: Path | None = None
    pretrained_reranker_model: str | None = None
    reranker_batch_size: int = 16
    use_hybrid_retrieval: bool = False
    vector_top_n: int = 20
    bm25_top_n: int = 20
    hybrid_rrf_k: int = 60
    retrieve_top_n: int = 0


def build_prompt(system: str, query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    return (
        f"{system}\n\n"
        "You are answering questions about FastAPI documentation.\n\n"
        "Use only the provided context as the source of facts.\n"
        "Do not use prior knowledge.\n"
        "Do not guess.\n\n"
        "When possible, copy the answer exactly as it appears in the Context.\n"
        "If multiple contexts are provided, select the one that directly contains the answer.\n"
        "Answer only if the answer is explicitly supported by the context.\n"
        "If the answer is missing from the context, return exactly:\n"
        "FINAL_ANSWER: NOT_FOUND\n\n"
        "Do not explain.\n"
        "Do not show reasoning.\n\n"
        "Return only the final answer in the format:\n"
        "FINAL_ANSWER: <answer>\n\n"
        "Keep the answer short, 1 to 3 words if possible.\n\n"
        f"{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "FINAL_ANSWER:"
    )

def load_contexts(docs_path: Path, ids: list[str]) -> list[str]:
    docs = read_jsonl(docs_path)
    lookup = {row["id"]: row["text"] for row in docs}
    return [lookup.get(doc_id, "") for doc_id in ids]


class RagPipeline:
    def __init__(self, config: RagConfig, generator: BaseGenerator | None = None) -> None:
        self.config = config
        if generator is not None:
            self.generator = generator
        elif self.config.use_hf_generator:
            self.generator = HFGenerator(
                GenerationConfig(
                    max_tokens=512,
                    temperature=0.0,
                    use_4bit=self.config.use_4bit_generator,
                    lora_adapter_dir=str(self.config.lora_adapter_dir)
                    if self.config.lora_adapter_dir
                    else None,
                )
            )
        else:
            self.generator = BaseGenerator(
                GenerationConfig(max_tokens=256, temperature=0.0)
            )
        docs = read_jsonl(self.config.docs_path)
        self._doc_lookup = {row["id"]: row["text"] for row in docs}
        self.bm25 = BM25Retriever(
            ((doc_id, text) for doc_id, text in self._doc_lookup.items())
        )
        self.reranker = None
        self.pretrained_reranker = None
        self._doc_emb_lookup: dict[str, np.ndarray] = {}
        try:
            if self.config.index_path.exists():
                self.index = load_index(self.config.index_path)
                # Validate that the loaded index ids match the current docs.jsonl.
                # If docs were regenerated, rebuild the index to avoid silent mismatches.
                if set(self.index.ids) != set(self._doc_lookup.keys()):
                    raise ValueError("Index/docs id mismatch; rebuilding index.")
            else:
                raise FileNotFoundError
        except Exception:
            self.index = build_index(
                self.config.docs_path,
                self.config.index_path,
                self.config.embeddings_path,
            )
        if self.config.reranker_model_path is not None:
            reranker_path = Path(self.config.reranker_model_path)
            if reranker_path.exists():
                self.reranker = load_model(reranker_path)
                if self.config.embeddings_path.exists():
                    emb = np.load(self.config.embeddings_path)
                    if len(self.index.ids) == len(emb):
                        self._doc_emb_lookup = {
                            doc_id: emb[i].astype(np.float32)
                            for i, doc_id in enumerate(self.index.ids)
                        }
            else:
                raise FileNotFoundError(f"Reranker model not found: {reranker_path}")
        if self.config.pretrained_reranker_model:
            self.pretrained_reranker = PretrainedReranker(
                model_name=self.config.pretrained_reranker_model,
                batch_size=self.config.reranker_batch_size,
            )

    def _hybrid_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vec_k = max(top_k, self.config.vector_top_n)
        bm25_k = max(top_k, self.config.bm25_top_n)
        vec_results = self.index.search(query, top_k=vec_k)
        bm25_results = self.bm25.search(query, top_k=bm25_k)
        # Reciprocal Rank Fusion for robust merge across heterogeneous scorers.
        fused: dict[str, float] = {}
        rank_bias = max(1, int(self.config.hybrid_rrf_k))
        for rank0, (doc_id, _score) in enumerate(vec_results):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank_bias + rank0 + 1)
        for rank0, (doc_id, _score) in enumerate(bm25_results):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank_bias + rank0 + 1)
        merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def _initial_retrieve(self, query: str, retrieve_n: int) -> list[tuple[str, float]]:
        if self.config.use_hybrid_retrieval:
            return self._hybrid_search(query, top_k=retrieve_n)
        return self.index.search(query, top_k=retrieve_n)

    def retrieve_with_stages(
        self, query: str
    ) -> tuple[list[str], list[tuple[str, float]], list[tuple[str, float]]]:
        retrieve_n = max(self.config.top_k, self.config.retrieve_top_n or self.config.top_k)
        initial_results = self._initial_retrieve(query, retrieve_n=retrieve_n)
        results = list(initial_results)
        if self.pretrained_reranker is not None and results:
            candidate_ids = [doc_id for doc_id, _ in results]
            candidate_texts = [self._doc_lookup.get(doc_id, "") for doc_id in candidate_ids]
            scores = self.pretrained_reranker.score(query, candidate_texts)
            reranked = list(zip(candidate_ids, scores))
            results = sorted(reranked, key=lambda x: x[1], reverse=True)[: self.config.top_k]
        elif self.reranker is not None and results:
            q_emb = text_to_embedding(query).astype(np.float32)
            reranked = []
            for rank0, (doc_id, base_score) in enumerate(results):
                doc_text = self._doc_lookup.get(doc_id, "")
                doc_emb = self._doc_emb_lookup.get(doc_id)
                cos = cosine_similarity(q_emb, doc_emb) if doc_emb is not None else 0.0
                score = self.reranker.score(
                    query=query,
                    chunk_text=doc_text,
                    base_rank_1_based=rank0 + 1,
                    cosine_sim=cos if doc_emb is not None else base_score,
                )
                reranked.append((doc_id, float(score)))
            results = sorted(reranked, key=lambda x: x[1], reverse=True)[: self.config.top_k]
        else:
            results = results[: self.config.top_k]
        ids = [doc_id for doc_id, _score in results]
        contexts = [self._doc_lookup.get(doc_id, "") for doc_id in ids]
        return contexts, results, initial_results

    def retrieve(self, query: str) -> tuple[list[str], list[tuple[str, float]]]:
        contexts, results, _initial_results = self.retrieve_with_stages(query)
        return contexts, results

    def answer(self, query: str) -> str:
        contexts, _results = self.retrieve(query)
        prompt = build_prompt(self.config.prompt_system, query, contexts)
        return self.generator.generate(prompt)

    def answer_with_context(self, query: str) -> tuple[str, list[str], list[tuple[str, float]]]:
        contexts, results = self.retrieve(query)
        prompt = build_prompt(self.config.prompt_system, query, contexts)
        return self.generator.generate(prompt), contexts, results

    def answer_with_context_stages(
        self, query: str
    ) -> tuple[str, list[str], list[tuple[str, float]], list[tuple[str, float]]]:
        contexts, results, initial_results = self.retrieve_with_stages(query)
        prompt = build_prompt(self.config.prompt_system, query, contexts)
        return self.generator.generate(prompt), contexts, results, initial_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple RAG query.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hybrid-retrieval", action="store_true")
    parser.add_argument("--vector-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument("--retrieve-top-n", type=int, default=0)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--pretrained-reranker-model", type=str, default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    args = parser.parse_args()

    logger = setup_logging("rag_pipeline")
    config = RagConfig(
        docs_path=args.docs,
        index_path=args.index,
        embeddings_path=args.embeddings,
        top_k=args.top_k,
        use_hybrid_retrieval=args.hybrid_retrieval,
        vector_top_n=args.vector_top_n,
        bm25_top_n=args.bm25_top_n,
        retrieve_top_n=args.retrieve_top_n,
        reranker_model_path=args.reranker_model,
        pretrained_reranker_model=args.pretrained_reranker_model,
        reranker_batch_size=args.reranker_batch_size,
    )
    pipeline = RagPipeline(config)
    answer = pipeline.answer(args.query)
    logger.info("Answer: %s", answer)


if __name__ == "__main__":
    main()

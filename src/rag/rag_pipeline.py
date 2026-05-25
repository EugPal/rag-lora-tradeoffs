from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.rag.generator import BaseGenerator, GenerationConfig, HFGenerator
from src.rag.embeddings import text_to_embedding
from src.rag.index import build_index, load_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.bge_m3_sparse_retriever import BGEM3SparseRetriever
from src.retrieval.linear_reranker import cosine_similarity, load_model
from src.retrieval.pretrained_reranker import PretrainedReranker
from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


@dataclass
class RagConfig:
    docs_path: Path = Path("data/processed/fresh_start/kubernetes/docs_kubernetes_semantic_v1.jsonl")
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
    use_bge_m3_native_sparse: bool = False
    sparse_top_n: int = 20
    sparse_index_path: Path = Path("data/embeddings/docs_sparse_bge_m3.json")


NORMAL_MODE_PATTERNS = (
    r"\bhow\b",
    r"\bwhy\b",
    r"\bwhat happens\b",
    r"\bwhat occurs\b",
    r"\bwhat is the purpose of\b",
    r"\bwhat does .+\bdo\b",
    r"\bwhen should\b",
    r"\bwhen would\b",
    r"\bwhen do\b",
    r"\bwhen does\b",
    r"\bdefault behavior\b",
)

EXACT_MODE_PREFIXES = (
    "which field",
    "what field",
    "which flag",
    "what flag",
    "which path",
    "what path",
    "at what path",
    "which api version",
    "what api version",
    "which feature gate",
    "what feature gate",
    "which command",
    "what command",
    "which port",
    "what port",
)

EXACT_MODE_PATTERNS = (
    r"\bdefault value\b",
    r"\bfield name\b",
    r"\bapi version\b",
    r"\bfeature gate\b",
    r"\benum value\b",
    r"\bannotation key\b",
    r"\blabel key\b",
    r"\benvironment variable\b",
    r"\benv var\b",
    r"\bwhat is the value of\b",
    r"\bwhat is the name of\b",
    r"\bwhat is the hostname\b",
    r"\bwhat is the token\b",
    r"\bwhat is the default value of\b",
)


def predict_answer_mode(question: str) -> str:
    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return "normal"
    if normalized.startswith(EXACT_MODE_PREFIXES):
        return "exact"
    for pattern in NORMAL_MODE_PATTERNS:
        if re.search(pattern, normalized):
            return "normal"
    for pattern in EXACT_MODE_PATTERNS:
        if re.search(pattern, normalized):
            return "exact"
    return "normal"


def resolve_answer_mode(question: str, answer_mode: str | None) -> str:
    mode = str(answer_mode or "none").strip().lower()
    if mode in {"none", "neutral", "no_mode", "nomode"}:
        return "none"
    if mode in {"auto", "router", "predicted"}:
        return predict_answer_mode(question)
    if mode in {"exact", "normal"}:
        return mode
    if mode in {"explicit_grounded", "explicit-grounded", "grounded_explicit"}:
        return "explicit_grounded"
    raise ValueError(f"Unsupported answer_mode: {answer_mode}")


def _build_system_prompt(system: str, answer_mode: str | None = "none") -> str:
    resolved_mode = resolve_answer_mode("", answer_mode)
    if resolved_mode == "none":
        return (
            f"{system}\n\n"
            "Answer the question using only the provided context.\n"
            "If the answer is not explicitly stated in the context, output exactly:\n"
            "FINAL_ANSWER: NOT_FOUND\n\n"
            "Rules:\n"
            "- Output exactly one final answer line. The line may contain multiple items or clauses if needed for a complete answer.\n"
            "- Do not add reasoning or commentary beyond the answer itself.\n"
            "- Do not use outside knowledge.\n"
            "- Do not add any text before or after the final answer.\n"
            "- Include key qualifiers if they appear in the context.\n"
            "- If the answer is a list, set of conditions, or multi-part fact, include all items explicitly supported by the context.\n"
            "- Prefer a complete short answer over a single keyword fragment.\n\n"
            "Examples:\n"
            "Context:\n"
            "The server supports Server-Sent Events (SSE) for streaming.\n"
            "Question:\n"
            "What transport does the server support for streaming?\n"
            "FINAL_ANSWER: Server-Sent Events (SSE)\n\n"
            "Context:\n"
            "The document describes authentication and task updates.\n"
            "Question:\n"
            "What database engine is required?\n"
            "FINAL_ANSWER: NOT_FOUND\n\n"
            "Output format:\n"
            "FINAL_ANSWER: <answer>"
        )
    if resolved_mode == "explicit_grounded":
        return (
            f"{system}\n\n"
            "Answer the question using only the provided context.\n"
            "If the answer is not explicitly supported by the context, output exactly:\n"
            "FINAL_ANSWER: NOT_FOUND\n\n"
            "Rules:\n"
            "- Output exactly one final answer line. The line may contain multiple items or clauses if needed for a complete answer.\n"
            "- Do not add reasoning or commentary beyond the answer itself.\n"
            "- Do not use outside knowledge.\n"
            "- Do not add any text before or after the final answer.\n"
            "- Include key qualifiers if they appear in the context.\n"
            "- If the answer is a list, set of conditions, or multi-part fact, include all items explicitly supported by the context.\n"
            "- Prefer a complete short answer over a single keyword fragment.\n"
            "- Stay as close as possible to the wording of the context while keeping the answer readable.\n"
            "- Do not add unsupported inferences, background detail, or speculative explanation.\n"
            "- If only part of the answer is explicitly supported, output only the supported part.\n"
            "- If support is missing, indirect, or ambiguous, output exactly:\n"
            "FINAL_ANSWER: NOT_FOUND\n\n"
            "Examples:\n"
            "Context:\n"
            "The server supports Server-Sent Events (SSE) for streaming.\n"
            "Question:\n"
            "What transport does the server support for streaming?\n"
            "FINAL_ANSWER: Server-Sent Events (SSE)\n\n"
            "Context:\n"
            "The document describes authentication and task updates.\n"
            "Question:\n"
            "What database engine is required?\n"
            "FINAL_ANSWER: NOT_FOUND\n\n"
            "Output format:\n"
            "FINAL_ANSWER: <answer>"
        )
    common = (
        f"{system}\n\n"
        "You answer Kubernetes questions using only the provided context.\n\n"
        "Global rules:\n"
        "- Use only the provided context.\n"
        "- If the answer is not supported by the context, say: not found in context\n"
        "- Be factual and directly answer the question.\n"
        "- Do not use outside knowledge.\n"
        "- Do not add irrelevant background or commentary.\n\n"
    )
    if resolved_mode == "exact":
        return common + (
            "Answering style:\n"
            "- Output only the literal answer span supported by the context.\n"
            "- Copy exact values exactly as written.\n"
            "- Do not paraphrase.\n"
            "- Do not normalize formatting.\n"
            "- Do not add introductory or trailing words.\n"
            "- Output the answer text only, not a full sentence.\n"
            "- Do not explain, rename, or restate the answer.\n"
            "- For flags, field names, paths, API versions, enum values, commands, labels, annotation keys, hostnames, URLs, quantities, and short item lists, preserve the source wording exactly.\n"
            "- If multiple items are required, output only those items and nothing else."
        )
    return common + (
        "Answering style:\n"
        "- Output a docs-grounded answer supported by the context.\n"
        "- Paraphrasing is allowed if it stays faithful to the context.\n"
        "- Include the specific detail needed to fully answer the question, including important qualifiers.\n"
        "- Prefer the most directly supported answer from the context, but include any additional detail needed to make the answer complete.\n"
        "- Use 1-3 sentences, or a short list if the question asks for multiple items.\n"
        "- Be concise, but include all essential qualifiers, conditions, and distinctions supported by the context.\n"
        "- If the question asks what, why, how, when, or what happens, prefer a complete factual answer over a short phrase."
    )


def _build_user_prompt(query: str, contexts: list[str], answer_mode: str | None = "none") -> str:
    resolved_mode = resolve_answer_mode(query, answer_mode)
    context_block = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    if resolved_mode in {"none", "explicit_grounded"}:
        return (
            f"{context_block}\n\n"
            f"Question:\n{query}\n\n"
            "FINAL_ANSWER:"
        )
    return (
        f"answer_mode: {resolved_mode}\n\n"
        f"{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:\n"
    )


def build_messages(
    system: str,
    query: str,
    contexts: list[str],
    answer_mode: str | None = "none",
) -> list[dict[str, str]]:
    resolved_mode = resolve_answer_mode(query, answer_mode)
    return [
        {"role": "system", "content": _build_system_prompt(system, resolved_mode)},
        {
            "role": "user",
            "content": _build_user_prompt(query, contexts, answer_mode=resolved_mode),
        },
    ]


def build_prompt(
    system: str,
    query: str,
    contexts: list[str],
    answer_mode: str | None = "none",
) -> str:
    resolved_mode = resolve_answer_mode(query, answer_mode)
    return _build_system_prompt(system, resolved_mode) + "\n\n" + _build_user_prompt(
        query,
        contexts,
        answer_mode=resolved_mode,
    )


def load_contexts(docs_path: Path, ids: list[str]) -> list[str]:
    docs = read_jsonl(docs_path)
    lookup = {row["id"]: row["text"] for row in docs}
    return [lookup.get(doc_id, "") for doc_id in ids]


def build_generation_input(
    system: str,
    query: str,
    contexts: list[str],
    answer_mode: str | None = "none",
) -> str | list[dict[str, str]]:
    resolved_mode = resolve_answer_mode(query, answer_mode)
    if resolved_mode in {"none", "explicit_grounded"}:
        return build_prompt(system, query, contexts, answer_mode=resolved_mode)
    return build_messages(system, query, contexts, answer_mode=resolved_mode)


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
                GenerationConfig(max_tokens=512, temperature=0.0)
            )
        docs = read_jsonl(self.config.docs_path)
        self._doc_lookup = {row["id"]: row["text"] for row in docs}
        self.bm25 = BM25Retriever(
            ((doc_id, text) for doc_id, text in self._doc_lookup.items())
        )
        self.native_sparse = None
        if self.config.use_bge_m3_native_sparse:
            self.native_sparse = BGEM3SparseRetriever(
                docs_path=self.config.docs_path,
                sparse_index_path=self.config.sparse_index_path,
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
        vec_results = self.index.search(query, top_k=vec_k)

        if self.config.use_bge_m3_native_sparse and self.native_sparse is not None:
            sparse_k = max(top_k, self.config.sparse_top_n)
            sparse_results = self.native_sparse.search(query, top_k=sparse_k)
            second_stage = sparse_results
        else:
            bm25_k = max(top_k, self.config.bm25_top_n)
            second_stage = self.bm25.search(query, top_k=bm25_k)

        # Reciprocal Rank Fusion for robust merge across heterogeneous scorers.
        fused: dict[str, float] = {}
        rank_bias = max(1, int(self.config.hybrid_rrf_k))
        for rank0, (doc_id, _score) in enumerate(vec_results):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank_bias + rank0 + 1)
        for rank0, (doc_id, _score) in enumerate(second_stage):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank_bias + rank0 + 1)
        merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def _initial_retrieve(self, query: str, retrieve_n: int) -> list[tuple[str, float]]:
        if self.config.use_hybrid_retrieval:
            return self._hybrid_search(query, top_k=retrieve_n)
        if self.config.use_bge_m3_native_sparse and self.native_sparse is not None:
            sparse_k = max(retrieve_n, self.config.sparse_top_n)
            return self.native_sparse.search(query, top_k=sparse_k)
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

    def answer(self, query: str, answer_mode: str = "none") -> str:
        contexts, _results = self.retrieve(query)
        generation_input = build_generation_input(
            self.config.prompt_system,
            query,
            contexts,
            answer_mode=answer_mode,
        )
        return self.generator.generate(generation_input)

    def answer_with_context(
        self,
        query: str,
        answer_mode: str = "none",
    ) -> tuple[str, list[str], list[tuple[str, float]]]:
        contexts, results = self.retrieve(query)
        generation_input = build_generation_input(
            self.config.prompt_system,
            query,
            contexts,
            answer_mode=answer_mode,
        )
        return self.generator.generate(generation_input), contexts, results

    def answer_with_context_stages(
        self,
        query: str,
        answer_mode: str = "none",
    ) -> tuple[str, list[str], list[tuple[str, float]], list[tuple[str, float]]]:
        contexts, results, initial_results = self.retrieve_with_stages(query)
        generation_input = build_generation_input(
            self.config.prompt_system,
            query,
            contexts,
            answer_mode=answer_mode,
        )
        return self.generator.generate(generation_input), contexts, results, initial_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple RAG query.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/processed/fresh_start/kubernetes/docs_kubernetes_semantic_v1.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hybrid-retrieval", action="store_true")
    parser.add_argument("--vector-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument("--retrieve-top-n", type=int, default=0)
    parser.add_argument("--native-sparse-retrieval", action="store_true")
    parser.add_argument("--sparse-top-n", type=int, default=20)
    parser.add_argument(
        "--answer-mode",
        type=str,
        choices=["none", "auto", "exact", "normal", "explicit_grounded"],
        default="none",
        help="Answer mode for generation; none uses the neutral mainline prompt, auto uses the built-in rule-based router.",
    )
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
        use_bge_m3_native_sparse=args.native_sparse_retrieval,
        sparse_top_n=args.sparse_top_n,
        reranker_model_path=args.reranker_model,
        pretrained_reranker_model=args.pretrained_reranker_model,
        reranker_batch_size=args.reranker_batch_size,
    )
    pipeline = RagPipeline(config)
    answer = pipeline.answer(args.query, answer_mode=args.answer_mode)
    logger.info("Answer: %s", answer)


if __name__ == "__main__":
    main()



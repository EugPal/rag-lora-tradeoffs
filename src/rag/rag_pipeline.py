from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.rag.generator import BaseGenerator, GenerationConfig, HFGenerator
from src.rag.index import build_index, load_index
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


def build_prompt(system: str, query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    return (
        f"{system}\n\n"
        "Rules:\n"
        "- Use ONLY the Context.\n"
        "- If the answer is not in the Context, say exactly: \"I don't know based on the provided context.\".\n"
        "- Be concise (1-3 sentences).\n"
        "- You MUST include 1-2 short verbatim quotes from the Context to support your answer.\n"
        "- Put quotes under a 'Quotes:' section, then the final response under an 'Answer:' section.\n"
        "- Each quote must be copied exactly from the Context.\n\n"
        "Output format:\n"
        "Quotes:\n"
        "- \"...\"\n"
        "- \"...\"\n"
        "Answer:\n"
        "<your answer>\n\n"
        f"{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Response:"
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
                GenerationConfig(max_tokens=128, temperature=0.0)
            )
        else:
            self.generator = BaseGenerator(
                GenerationConfig(max_tokens=256, temperature=0.0)
            )
        docs = read_jsonl(self.config.docs_path)
        self._doc_lookup = {row["id"]: row["text"] for row in docs}
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

    def retrieve(self, query: str) -> tuple[list[str], list[tuple[str, float]]]:
        results = self.index.search(query, top_k=self.config.top_k)
        ids = [doc_id for doc_id, _score in results]
        contexts = [self._doc_lookup.get(doc_id, "") for doc_id in ids]
        return contexts, results

    def answer(self, query: str) -> str:
        contexts, _results = self.retrieve(query)
        prompt = build_prompt(self.config.prompt_system, query, contexts)
        return self.generator.generate(prompt)

    def answer_with_context(self, query: str) -> tuple[str, list[str], list[tuple[str, float]]]:
        contexts, results = self.retrieve(query)
        prompt = build_prompt(self.config.prompt_system, query, contexts)
        return self.generator.generate(prompt), contexts, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple RAG query.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    logger = setup_logging("rag_pipeline")
    config = RagConfig(
        docs_path=args.docs,
        index_path=args.index,
        embeddings_path=args.embeddings,
        top_k=args.top_k,
    )
    pipeline = RagPipeline(config)
    answer = pipeline.answer(args.query)
    logger.info("Answer: %s", answer)


if __name__ == "__main__":
    main()

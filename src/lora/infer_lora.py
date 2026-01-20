from __future__ import annotations

import argparse
from pathlib import Path

from src.lora.lora_config import get_preset
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG inference with LoRA config.")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    args = parser.parse_args()

    logger = setup_logging("infer_lora")
    _config = get_preset(args.preset)
    rag_config = RagConfig(
        docs_path=args.docs,
        index_path=args.index,
        embeddings_path=args.embeddings,
    )
    pipeline = RagPipeline(rag_config)
    answer = pipeline.answer(args.query)
    logger.info("LoRA preset %s answer: %s", args.preset, answer)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from src.lora.lora_config import get_preset
from src.rag.generator import GenerationConfig, HFGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG inference with LoRA config.")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Override adapter directory (defaults to experiments/lora/<preset>/adapter)",
    )
    args = parser.parse_args()

    logger = setup_logging("infer_lora")
    _config = get_preset(args.preset)
    adapter_dir = args.adapter_dir or (Path("experiments/lora") / args.preset / "adapter")
    rag_config = RagConfig(
        docs_path=args.docs,
        index_path=args.index,
        embeddings_path=args.embeddings,
    )
    generator = HFGenerator(GenerationConfig(lora_adapter_dir=str(adapter_dir)))
    pipeline = RagPipeline(rag_config, generator=generator)
    answer = pipeline.answer(args.query)
    logger.info("LoRA preset %s answer: %s", args.preset, answer)


if __name__ == "__main__":
    main()

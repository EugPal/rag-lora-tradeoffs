from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import exact_match, f1_score
from src.lora.lora_config import get_preset
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA RAG.")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/qa_test.jsonl"))
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    logger = setup_logging("eval_lora")
    _config = get_preset(args.preset)
    rows = read_jsonl(args.test_file)
    if not rows:
        logger.warning("No QA test data found at %s", args.test_file)
        return

    pipeline = RagPipeline(RagConfig())
    em_scores = []
    f1_scores = []
    for row in rows:
        prediction = pipeline.answer(row["question"])
        em_scores.append(exact_match(prediction, row["answer"]))
        f1_scores.append(f1_score(prediction, row["answer"]))

    results = {
        "preset": args.preset,
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "samples": len(rows),
    }
    out_file = args.out_file or Path("experiments/lora") / args.preset / "metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", out_file)


if __name__ == "__main__":
    main()

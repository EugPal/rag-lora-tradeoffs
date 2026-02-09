from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lora.lora_config import get_preset
from src.utils.io_utils import write_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA RAG.")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/qa_test.jsonl"))
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-max-samples", type=int, default=200)
    args = parser.parse_args()

    logger = setup_logging("eval_lora")
    _config = get_preset(args.preset)

    adapter_dir = Path("experiments/lora") / args.preset / "adapter"
    out_file = args.out_file or Path("experiments/lora") / args.preset / "metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Delegate to eval_baseline with LoRA adapter so metrics are comparable.
    from src.evaluation.eval_baseline import main as eval_baseline_main  # noqa: PLC0415
    import sys  # noqa: PLC0415

    argv = [
        "eval_baseline",
        "--test-file",
        str(args.test_file),
        "--out-file",
        str(out_file),
        "--predictions-file",
        str(out_file.parent / "predictions.jsonl"),
        "--top-k",
        str(args.top_k),
        "--lora-adapter",
        str(adapter_dir),
    ]
    if args.judge:
        argv += ["--judge", "--judge-max-samples", str(args.judge_max_samples)]
    sys.argv = argv
    eval_baseline_main()
    logger.info("Saved LoRA eval results to %s", out_file)


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    py = sys.executable
    cmd = [
        py,
        "-m",
        "src.evaluation.eval_baseline",
        "--test-file",
        "data/processed/qa_eval_main.jsonl",
        "--top-k",
        "4",
        "--judge",
        "--judge-max-samples",
        "50",
        "--no-quant-judge",
        "--no-quant-generator",
        "--lora-adapter",
        "experiments/lora_fp16/L4-F/adapter",
        "--out-file",
        "experiments/pilot/results_lora_L4-F_eval_main_fp16_adapter.json",
        "--predictions-file",
        "experiments/pilot/predictions_lora_L4-F_eval_main_fp16_adapter.jsonl",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

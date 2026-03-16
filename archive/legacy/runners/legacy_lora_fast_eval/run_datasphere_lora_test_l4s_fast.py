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
        "data/processed/qa_test_mixed.jsonl",
        "--judge",
        "--judge-max-samples",
        "50",
        "--lora-adapter",
        "experiments/lora/L4-S/adapter",
        "--out-file",
        "experiments/pilot/results_lora_L4-S_test_mixed_fast.json",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys


def run_step(args: list[str]) -> None:
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    py = sys.executable
    steps: list[list[str]] = [
        [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_eval_mixed.jsonl",
            "--judge",
            "--judge-max-samples",
            "50",
            "--out-file",
            "experiments/pilot/results_baseline_eval_mixed_fast.json",
        ],
        [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_eval_mixed.jsonl",
            "--judge",
            "--judge-max-samples",
            "50",
            "--lora-adapter",
            "experiments/lora/L4-S/adapter",
            "--out-file",
            "experiments/pilot/results_lora_L4-S_eval_mixed_fast.json",
        ],
        [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_test_mixed.jsonl",
            "--judge",
            "--judge-max-samples",
            "50",
            "--out-file",
            "experiments/pilot/results_baseline_test_mixed_fast.json",
        ],
        [
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
        ],
    ]
    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()

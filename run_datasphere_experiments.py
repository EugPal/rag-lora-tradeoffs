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
            "src.lora.train_lora",
            "--preset",
            "L4-S",
            "--train-file",
            "data/processed/qa_train_mixed.jsonl",
            "--val-file",
            "data/processed/qa_eval_main.jsonl",
            "--embed-top-k",
            "4",
            "--gradient-checkpointing",
            "--out-dir",
            "experiments/lora",
        ],
        [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_eval_main.jsonl",
            "--bertscore",
            "--judge",
            "--judge-max-samples",
            "50",
            "--out-file",
            "experiments/pilot/results_baseline_eval_main_fast.json",
        ],
        [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_test_main.jsonl",
            "--bertscore",
            "--judge",
            "--judge-max-samples",
            "50",
            "--out-file",
            "experiments/pilot/results_baseline_test_main_fast.json",
        ],
    ]
    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()

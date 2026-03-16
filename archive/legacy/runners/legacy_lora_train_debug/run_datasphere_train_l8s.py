from __future__ import annotations

import subprocess
import sys


def main() -> None:
    py = sys.executable
    cmd = [
        py,
        "-m",
        "src.lora.train_lora",
        "--preset",
        "L4-S",
        "--train-file",
        "data/processed/qa_train_mixed.jsonl",
        "--val-file",
        "data/processed/qa_eval_mixed.jsonl",
        "--embed-top-k",
        "4",
        "--gradient-checkpointing",
        "--out-dir",
        "experiments/lora",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

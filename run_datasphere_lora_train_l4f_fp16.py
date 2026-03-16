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
        "L4-F",
        "--train-file",
        "data/processed/qa_train_mixed.jsonl",
        "--embed-top-k",
        "4",
        "--gradient-checkpointing",
        "--no-quantization",
        "--out-dir",
        "experiments/lora_fp16",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

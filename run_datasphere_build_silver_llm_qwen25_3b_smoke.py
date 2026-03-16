from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(args: list[str]) -> None:
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _log_disk(label: str, path: str) -> None:
    try:
        total, used, free = shutil.disk_usage(path)
        gib = 1024**3
        print(
            f"[disk] {label} path={path} total_gb={total / gib:.2f} used_gb={used / gib:.2f} free_gb={free / gib:.2f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[disk] {label} path={path} error={exc}", flush=True)


def main() -> None:
    py = sys.executable
    hf_home = os.environ.get("HF_HOME", "/tmp/huggingface")
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path("/tmp/huggingface").mkdir(parents=True, exist_ok=True)
    _log_disk("root", "/")
    _log_disk("tmp", "/tmp")
    _log_disk("hf_home", hf_home)
    if os.environ.get("DS_PROJECT_HOME"):
        _log_disk("project", os.environ["DS_PROJECT_HOME"])

    steps: list[list[str]] = [
        [
            py,
            "-m",
            "src.data_pipeline.build_qa_dataset",
            "--docs-file",
            "data/processed/docs.jsonl",
            "--out-dir",
            "data/processed",
            "--max-qa",
            "40",
            "--max-source-chunks",
            "120",
            "--seed",
            "42",
            "--use-llm",
            "--llm-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--llm-max-tokens",
            "384",
            "--llm-max-attempts",
            "1",
            "--show-progress",
        ],
        [
            py,
            "-m",
            "src.data_pipeline.filter_qa_candidates",
            "--in-file",
            "data/processed/qa_small.jsonl",
            "--docs-file",
            "data/processed/docs.jsonl",
            "--silver-pages-file",
            "data/processed/silver_pages.txt",
            "--out-file",
            "data/processed/qa_silver_filtered.jsonl",
            "--target-size",
            "40",
            "--seed",
            "42",
            "--max-per-page",
            "4",
        ],
    ]
    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()


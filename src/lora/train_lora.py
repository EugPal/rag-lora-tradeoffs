from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.lora.lora_config import get_preset
from src.utils.io_utils import read_jsonl, write_yaml
from src.utils.logging_utils import setup_logging


def estimate_params(rank: int, target_layers: str, data_fraction: float) -> int:
    base = rank * (8 if target_layers == "all" else 4)
    return int(base * (1 + data_fraction))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub LoRA training.")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/qa_train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/lora"))
    args = parser.parse_args()

    logger = setup_logging("train_lora")
    config = get_preset(args.preset)
    train_rows = read_jsonl(args.train_file)
    used_rows = int(len(train_rows) * config.data_fraction)
    params = estimate_params(config.rank, config.target_layers, config.data_fraction)

    run_dir = args.out_dir / args.preset
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config_used.yaml", asdict(config))
    metrics = {
        "train_rows": used_rows,
        "estimated_lora_params": params,
        "status": "stub_training_completed",
    }
    (run_dir / "logs.txt").write_text(
        "Stub training complete. Replace with actual LoRA training.\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        f"{metrics}\n",
        encoding="utf-8",
    )
    logger.info("Saved stub LoRA artifacts to %s", run_dir)


if __name__ == "__main__":
    main()

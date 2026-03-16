from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    py = sys.executable
    frozen_cfg_path = Path(
        os.environ.get("BASELINE_FROZEN_CONFIG", "experiments/pilot/baseline_frozen_config.json")
    )
    if not frozen_cfg_path.exists():
        raise FileNotFoundError(
            f"Frozen baseline config not found: {frozen_cfg_path}. "
            "Run baseline val selection job first."
        )
    frozen_cfg = json.loads(frozen_cfg_path.read_text(encoding="utf-8"))
    top_k = int(frozen_cfg["selected_top_k"])
    cmd = [
        py,
        "-m",
        "src.evaluation.eval_baseline",
        "--test-file",
        "data/processed/qa_test_main.jsonl",
        "--top-k",
        str(top_k),
        "--judge",
        "--judge-max-samples",
        "50",
        "--no-quant-judge",
        "--no-quant-generator",
        "--out-file",
        "experiments/pilot/results_baseline_test_main_fast_fp16_all.json",
        "--predictions-file",
        "experiments/pilot/predictions_baseline_test_main_fast_fp16_all.jsonl",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

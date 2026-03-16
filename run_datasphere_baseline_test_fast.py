from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    py = sys.executable
    frozen_cfg_path = Path(os.environ.get("BASELINE_FROZEN_CONFIG", "experiments/pilot/baseline_frozen_config.json"))
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
        "--out-file",
        "experiments/pilot/results_baseline_test_main_fast.json",
    ]
    if os.environ.get("BASELINE_NO_QUANT_JUDGE", "").strip() == "1":
        cmd.append("--no-quant-judge")
    if os.environ.get("BASELINE_NO_QUANT_GENERATOR", "").strip() == "1":
        cmd.append("--no-quant-generator")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

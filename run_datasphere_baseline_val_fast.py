from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _parse_top_k_candidates(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("BASELINE_TOP_K_CANDIDATES is empty; expected comma-separated ints like '4,8,12'.")
    return values


def _choose_top_k_retrieval_centric(results: list[dict], hit_threshold: float) -> int:
    # Prefer the smallest k that achieves the retrieval page-hit threshold.
    passing = [
        r for r in results if float(r.get("retrieval_hit_rate_page", 0.0)) >= hit_threshold
    ]
    if passing:
        return int(sorted(passing, key=lambda r: int(r["top_k"]))[0]["top_k"])

    # Fallback: highest page-hit, then highest page-MRR, then smallest k.
    best = sorted(
        results,
        key=lambda r: (
            float(r.get("retrieval_hit_rate_page", 0.0)),
            float(r.get("retrieval_mrr_page", 0.0)),
            -int(r["top_k"]),
        ),
        reverse=True,
    )[0]
    return int(best["top_k"])


def main() -> None:
    py = sys.executable
    candidates = _parse_top_k_candidates(os.environ.get("BASELINE_TOP_K_CANDIDATES", "4,8,12"))
    hit_threshold = float(os.environ.get("BASELINE_HIT_THRESHOLD", "0.75"))
    out_dir = Path("experiments/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_k_results = []
    for top_k in candidates:
        out_file = out_dir / f"results_baseline_eval_main_topk{top_k}_fast.json"
        pred_file = out_dir / f"predictions_baseline_eval_main_topk{top_k}_fast.jsonl"
        cmd = [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_eval_main.jsonl",
            "--top-k",
            str(top_k),
            "--judge",
            "--judge-max-samples",
            "50",
            "--out-file",
            str(out_file),
            "--predictions-file",
            str(pred_file),
        ]
        if os.environ.get("BASELINE_NO_QUANT_JUDGE", "").strip() == "1":
            cmd.append("--no-quant-judge")
        if os.environ.get("BASELINE_NO_QUANT_GENERATOR", "").strip() == "1":
            cmd.append("--no-quant-generator")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        result = json.loads(out_file.read_text(encoding="utf-8"))
        result["top_k"] = top_k
        per_k_results.append(result)

    selected_top_k = _choose_top_k_retrieval_centric(per_k_results, hit_threshold)
    best = [r for r in per_k_results if int(r["top_k"]) == selected_top_k][0]

    frozen_cfg = {
        "dataset": "qa_eval_main.jsonl",
        "selection_metric": "retrieval-centric: minimal top_k with retrieval_hit_rate_page >= threshold; fallback max retrieval_hit_rate_page, then retrieval_mrr_page, then smaller top_k",
        "candidate_top_k": candidates,
        "hit_threshold": hit_threshold,
        "selected_top_k": selected_top_k,
    }
    (out_dir / "baseline_frozen_config.json").write_text(
        json.dumps(frozen_cfg, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results_baseline_eval_grid_fast.json").write_text(
        json.dumps({"runs": per_k_results, "selected_top_k": selected_top_k}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results_baseline_eval_main_fast.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen baseline selected: top_k={selected_top_k}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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
    passing = [r for r in results if float(r.get("retrieval_hit_rate_page", 0.0)) >= hit_threshold]
    if passing:
        best = sorted(
            passing,
            key=lambda r: (
                float(r.get("retrieval_recall_positive_chunks_at_k") or 0.0),
                -(float(r.get("hard_negative_top1_error_rate") or 0.0)),
                -int(r["top_k"]),
            ),
            reverse=True,
        )[0]
        return int(best["top_k"])
    best = sorted(
        results,
        key=lambda r: (
            float(r.get("retrieval_hit_rate_page", 0.0)),
            float(r.get("retrieval_recall_positive_chunks_at_k") or 0.0),
            -(float(r.get("hard_negative_top1_error_rate") or 0.0)),
            float(r.get("retrieval_mrr_page", 0.0)),
            -int(r["top_k"]),
        ),
        reverse=True,
    )[0]
    return int(best["top_k"])


def _print_storage_diagnostics() -> None:
    hf_home = os.environ.get("HF_HOME", "")
    print(f"HF_HOME={hf_home}", flush=True)
    if hf_home:
        try:
            Path(hf_home).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to create HF_HOME directory: {exc}", flush=True)
        try:
            usage = shutil.disk_usage(hf_home)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            print(f"HF_HOME disk free: {free_gb:.2f} GB / {total_gb:.2f} GB", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read HF_HOME disk usage: {exc}", flush=True)
    try:
        subprocess.run(["df", "-h"], check=False)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to run df -h: {exc}", flush=True)


def main() -> None:
    py = sys.executable
    out_dir = Path("experiments/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    _print_storage_diagnostics()

    candidates = _parse_top_k_candidates(os.environ.get("BASELINE_TOP_K_CANDIDATES", "5"))
    hit_threshold = float(os.environ.get("BASELINE_HIT_THRESHOLD", "0.75"))

    retrieve_top_n = int(os.environ.get("RERANKER_RETRIEVE_TOP_N", "40"))
    vector_top_n = int(os.environ.get("HYBRID_VECTOR_TOP_N", "20"))
    bm25_top_n = int(os.environ.get("HYBRID_BM25_TOP_N", "20"))
    reranker_model_name = os.environ.get(
        "PRETRAINED_RERANKER_MODEL",
        "BAAI/bge-reranker-base",
    )
    supervision_file = Path(
        os.environ.get(
            "RETRIEVAL_SUPERVISION_EVAL_FILE",
            "data/processed/retrieval_supervision_eval_reranker.jsonl",
        )
    )
    supervision_stats_file = Path(
        os.environ.get(
            "RETRIEVAL_SUPERVISION_EVAL_STATS_FILE",
            "data/processed/retrieval_supervision_eval_reranker_stats.json",
        )
    )
    build_supervision_cmd = [
        py,
        "-m",
        "src.data_pipeline.build_retrieval_supervision",
        "--qa-file",
        "data/processed/qa_eval_main.jsonl",
        "--docs-file",
        "data/processed/docs.jsonl",
        "--index-file",
        "data/embeddings/docs_embeddings.faiss",
        "--out-file",
        str(supervision_file),
        "--stats-file",
        str(supervision_stats_file),
        "--top-n",
        str(max(50, retrieve_top_n)),
        "--hard-negatives",
        "3",
        "--in-page-negatives",
        "1",
        "--random-negatives",
        "1",
        "--add-neighbor-positive",
        "--max-negatives-per-positive",
        "5",
        "--seed",
        "42",
    ]
    print("Running:", " ".join(build_supervision_cmd), flush=True)
    subprocess.run(build_supervision_cmd, check=True)

    # Evaluate baseline+pretrained-reranker on val grid.
    per_k_results = []
    for top_k in candidates:
        out_file = out_dir / f"results_baseline_eval_main_topk{top_k}_reranker_fast.json"
        pred_file = out_dir / f"predictions_baseline_eval_main_topk{top_k}_reranker_fast.jsonl"
        cmd = [
            py,
            "-m",
            "src.evaluation.eval_baseline",
            "--test-file",
            "data/processed/qa_eval_main.jsonl",
            "--hybrid-retrieval",
            "--vector-top-n",
            str(vector_top_n),
            "--bm25-top-n",
            str(bm25_top_n),
            "--top-k",
            str(top_k),
            "--retrieve-top-n",
            str(retrieve_top_n),
            "--pretrained-reranker-model",
            reranker_model_name,
            "--retrieval-supervision-file",
            str(supervision_file),
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
        "selection_metric": "retrieval-centric: pass retrieval_hit_rate_page threshold, then maximize positive recall@k, minimize hard-negative top1 error, prefer smaller top_k; fallback by page hit/recall/hard-neg/page mrr",
        "candidate_top_k": candidates,
        "hit_threshold": hit_threshold,
        "selected_top_k": selected_top_k,
        "retrieve_top_n": retrieve_top_n,
        "hybrid_vector_top_n": vector_top_n,
        "hybrid_bm25_top_n": bm25_top_n,
        "pretrained_reranker_model": reranker_model_name,
        "retrieval_supervision_eval_file": str(supervision_file),
    }
    (out_dir / "baseline_frozen_config_reranker.json").write_text(
        json.dumps(frozen_cfg, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results_baseline_eval_grid_reranker_fast.json").write_text(
        json.dumps({"runs": per_k_results, "selected_top_k": selected_top_k}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results_baseline_eval_main_reranker_fast.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen baseline+rereanker selected: top_k={selected_top_k}", flush=True)


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


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
    _print_storage_diagnostics()
    frozen_cfg_path = Path(
        os.environ.get(
            "BASELINE_FROZEN_CONFIG_RERANKER",
            "experiments/pilot/baseline_frozen_config_reranker.json",
        )
    )
    if not frozen_cfg_path.exists():
        raise FileNotFoundError(
            f"Frozen baseline+rereanker config not found: {frozen_cfg_path}. "
            "Run baseline val reranker selection job first."
        )
    frozen_cfg = json.loads(frozen_cfg_path.read_text(encoding="utf-8"))
    top_k = int(frozen_cfg["selected_top_k"])
    retrieve_top_n = int(frozen_cfg.get("retrieve_top_n", 40))
    vector_top_n = int(frozen_cfg.get("hybrid_vector_top_n", 20))
    bm25_top_n = int(frozen_cfg.get("hybrid_bm25_top_n", 20))
    pretrained_reranker_model = str(
        frozen_cfg.get("pretrained_reranker_model", "BAAI/bge-reranker-base")
    )
    supervision_file = Path(
        os.environ.get(
            "RETRIEVAL_SUPERVISION_TEST_FILE",
            "data/processed/retrieval_supervision_test_reranker.jsonl",
        )
    )
    supervision_stats_file = Path(
        os.environ.get(
            "RETRIEVAL_SUPERVISION_TEST_STATS_FILE",
            "data/processed/retrieval_supervision_test_reranker_stats.json",
        )
    )
    build_supervision_cmd = [
        py,
        "-m",
        "src.data_pipeline.build_retrieval_supervision",
        "--qa-file",
        "data/processed/qa_test_main.jsonl",
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

    cmd = [
        py,
        "-m",
        "src.evaluation.eval_baseline",
        "--test-file",
        "data/processed/qa_test_main.jsonl",
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
        pretrained_reranker_model,
        "--retrieval-supervision-file",
        str(supervision_file),
        "--judge",
        "--judge-max-samples",
        "50",
        "--out-file",
        "experiments/pilot/results_baseline_test_main_reranker_fast.json",
        "--predictions-file",
        "experiments/pilot/predictions_baseline_test_main_reranker_fast.jsonl",
    ]
    if os.environ.get("BASELINE_NO_QUANT_JUDGE", "").strip() == "1":
        cmd.append("--no-quant-judge")
    if os.environ.get("BASELINE_NO_QUANT_GENERATOR", "").strip() == "1":
        cmd.append("--no-quant-generator")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


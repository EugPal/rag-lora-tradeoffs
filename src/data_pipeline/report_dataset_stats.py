from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def jsonl_count(path: Path) -> int:
    return len(read_jsonl(path))


def distribution(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not value:
            continue
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset manifest/statistics report.")
    parser.add_argument("--urls-file", type=Path, default=Path("data/raw/fastapi_urls.txt"))
    parser.add_argument("--html-dir", type=Path, default=Path("data/raw/fastapi_html"))
    parser.add_argument("--pages-file", type=Path, default=Path("data/processed/fastapi_pages.jsonl"))
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--silver-file", type=Path, default=Path("data/processed/qa_silver_filtered.jsonl"))
    parser.add_argument("--gold-file", type=Path, default=Path("data/processed/qa_gold_full.jsonl"))
    parser.add_argument("--gold-val-file", type=Path, default=Path("data/processed/qa_gold_val.jsonl"))
    parser.add_argument("--gold-test-file", type=Path, default=Path("data/processed/qa_gold_test.jsonl"))
    parser.add_argument("--silver-train-file", type=Path, default=Path("data/processed/qa_silver_train.jsonl"))
    parser.add_argument("--real-user-full-file", type=Path, default=Path("data/processed/qa_real_user_full.jsonl"))
    parser.add_argument("--real-user-train-file", type=Path, default=Path("data/processed/qa_real_user_train.jsonl"))
    parser.add_argument("--real-user-val-file", type=Path, default=Path("data/processed/qa_real_user_val.jsonl"))
    parser.add_argument("--real-user-test-file", type=Path, default=Path("data/processed/qa_real_user_test.jsonl"))
    parser.add_argument("--mixed-train-file", type=Path, default=Path("data/processed/qa_train_mixed.jsonl"))
    parser.add_argument("--mixed-eval-file", type=Path, default=Path("data/processed/qa_eval_mixed.jsonl"))
    parser.add_argument("--mixed-test-file", type=Path, default=Path("data/processed/qa_test_mixed.jsonl"))
    parser.add_argument("--silver-pages-file", type=Path, default=Path("data/processed/silver_pages.txt"))
    parser.add_argument("--gold-pages-file", type=Path, default=Path("data/processed/gold_pages.txt"))
    parser.add_argument("--page-split-summary", type=Path, default=Path("data/processed/page_split_summary.json"))
    parser.add_argument("--url-stats-file", type=Path, default=Path("data/processed/url_list_stats.json"))
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/dataset_manifest.json"),
    )
    args = parser.parse_args()

    logger = setup_logging("report_dataset_stats")
    html_files = sorted(args.html_dir.glob("*.html")) if args.html_dir.exists() else []
    silver_rows = read_jsonl(args.silver_file)
    silver_train_rows = read_jsonl(args.silver_train_file)
    gold_rows = read_jsonl(args.gold_file)
    real_user_rows = read_jsonl(args.real_user_full_file)
    page_split = {}
    if args.page_split_summary.exists():
        page_split = json.loads(args.page_split_summary.read_text(encoding="utf-8"))
    url_stats = {}
    if args.url_stats_file.exists():
        url_stats = json.loads(args.url_stats_file.read_text(encoding="utf-8"))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "urls_file": str(args.urls_file),
            "html_dir": str(args.html_dir),
            "pages_file": str(args.pages_file),
            "docs_file": str(args.docs_file),
        },
        "counts": {
            "urls": count_lines(args.urls_file),
            "html_pages": len(html_files),
            "parsed_pages": jsonl_count(args.pages_file),
            "doc_chunks": jsonl_count(args.docs_file),
            "qa_silver_filtered": jsonl_count(args.silver_file),
            "qa_gold_full": jsonl_count(args.gold_file),
            "qa_gold_val": jsonl_count(args.gold_val_file),
            "qa_gold_test": jsonl_count(args.gold_test_file),
            "qa_silver_train": jsonl_count(args.silver_train_file),
            "qa_real_user_full": jsonl_count(args.real_user_full_file),
            "qa_real_user_train": jsonl_count(args.real_user_train_file),
            "qa_real_user_val": jsonl_count(args.real_user_val_file),
            "qa_real_user_test": jsonl_count(args.real_user_test_file),
            "qa_train_mixed": jsonl_count(args.mixed_train_file),
            "qa_eval_mixed": jsonl_count(args.mixed_eval_file),
            "qa_test_mixed": jsonl_count(args.mixed_test_file),
            "silver_pages": count_lines(args.silver_pages_file),
            "gold_pages": count_lines(args.gold_pages_file),
        },
        "hashes": {
            "urls_sha256": sha256_file(args.urls_file),
            "url_stats_sha256": sha256_file(args.url_stats_file),
            "pages_sha256": sha256_file(args.pages_file),
            "docs_sha256": sha256_file(args.docs_file),
            "silver_sha256": sha256_file(args.silver_file),
            "gold_sha256": sha256_file(args.gold_file),
            "real_user_sha256": sha256_file(args.real_user_full_file),
            "mixed_train_sha256": sha256_file(args.mixed_train_file),
            "mixed_eval_sha256": sha256_file(args.mixed_eval_file),
            "mixed_test_sha256": sha256_file(args.mixed_test_file),
            "silver_pages_sha256": sha256_file(args.silver_pages_file),
            "gold_pages_sha256": sha256_file(args.gold_pages_file),
        },
        "distributions": {
            "silver_filtered_categories": distribution(silver_rows, "category"),
            "silver_filtered_sections": distribution(silver_rows, "section"),
            "silver_train_categories": distribution(silver_train_rows, "category"),
            "gold_full_sections": distribution(gold_rows, "section"),
            "real_user_categories": distribution(real_user_rows, "category"),
            "real_user_sections": distribution(real_user_rows, "section"),
        },
        "page_split_summary": page_split,
        "url_list_stats": url_stats,
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest to %s", args.out_file)


if __name__ == "__main__":
    main()

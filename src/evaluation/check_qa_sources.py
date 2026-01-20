from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io_utils import read_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA source_chunk ids.")
    parser.add_argument("--qa-file", type=Path, default=Path("data/processed/qa_small.jsonl"))
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    args = parser.parse_args()

    logger = setup_logging("check_qa_sources")
    qa_rows = read_jsonl(args.qa_file)
    doc_rows = read_jsonl(args.docs_file)

    doc_ids = {row.get("id") for row in doc_rows}
    missing = [row.get("id") for row in qa_rows if row.get("source_chunk") not in doc_ids]

    if missing:
        logger.warning("Missing source_chunk for %d QA rows.", len(missing))
        for qa_id in missing:
            logger.warning("Missing source_chunk in QA id=%s", qa_id)
        raise SystemExit(1)

    logger.info("All QA rows have valid source_chunk ids. (%d rows)", len(qa_rows))


if __name__ == "__main__":
    main()

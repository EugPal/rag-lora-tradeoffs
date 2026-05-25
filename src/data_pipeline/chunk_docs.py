from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.io_utils import ensure_dir, read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def split_paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs = [line for line in lines if line]
    return paragraphs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        para_words = len(paragraph.split())
        if current_words + para_words > chunk_size and current:
            chunks.append(" ".join(current))
            if overlap > 0:
                overlap_words = 0
                overlap_paras: list[str] = []
                for prev in reversed(current):
                    overlap_paras.insert(0, prev)
                    overlap_words += len(prev.split())
                    if overlap_words >= overlap:
                        break
                current = overlap_paras[:]
                current_words = sum(len(p.split()) for p in current)
            else:
                current = []
                current_words = 0
        current.append(paragraph)
        current_words += para_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_pages(in_file: Path) -> list[dict]:
    return read_jsonl(in_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk docs into docs.jsonl.")
    parser.add_argument(
        "--in-file",
        type=Path,
        default=Path("data/processed/fastapi_pages.jsonl"),
    )
    parser.add_argument("--out-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    logger = setup_logging("chunk_docs")
    pages = load_pages(args.in_file)
    if not pages:
        logger.warning("No pages found in %s", args.in_file)
        return

    rows = []
    pbar = tqdm(pages, desc="chunk_docs", unit="page", disable=not sys.stderr.isatty())
    for page_idx, page in enumerate(pbar):
        text = page.get("text", "")
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        for chunk_idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": f"{page.get('id', 'page')}-{chunk_idx}",
                    "source": page.get("source", "unknown"),
                    "text": chunk,
                    "chunk_index": chunk_idx,
                    "doc_index": page_idx,
                }
            )
    ensure_dir(args.out_file.parent)
    write_jsonl(args.out_file, rows)
    logger.info("Wrote %d chunks to %s", len(rows), args.out_file)


if __name__ == "__main__":
    main()

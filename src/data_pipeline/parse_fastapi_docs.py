from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils.io_utils import ensure_dir, write_jsonl
from src.utils.logging_utils import setup_logging


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content and layout elements.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return soup.get_text("\n", strip=True)
    return main.get_text("\n", strip=True)


def normalize_text(text: str) -> str:
    boilerplate_substrings = [
        "waiting list",
        "subscribe",
        "sponsor",
        "table of contents",
        "initializing search",
        "fastapi and friends newsletter",
        "material for mkdocs",
    ]
    cleaned_lines: list[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        lower = line.lower()
        if any(s in lower for s in boilerplate_substrings):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def parse_dir(in_dir: Path, out_file: Path) -> None:
    logger = setup_logging("parse_fastapi_docs")
    ensure_dir(out_file.parent)
    html_files = sorted(in_dir.glob("*.html"))
    if not html_files:
        logger.warning("No HTML files found in %s", in_dir)
        return
    rows = []
    pbar = tqdm(html_files, desc="parse_fastapi_docs", unit="file", disable=not sys.stderr.isatty())
    for html_file in pbar:
        raw_text = html_to_text(html_file.read_text(encoding="utf-8", errors="ignore"))
        text = normalize_text(raw_text)
        rows.append(
            {
                "id": html_file.stem,
                "source": html_file.name,
                "text": text,
            }
        )
    write_jsonl(out_file, rows)
    logger.info("Parsed %d pages -> %s", len(rows), out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse FastAPI HTML into plain text.")
    parser.add_argument("--in-dir", type=Path, default=Path("data/raw/fastapi_html"))
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/fastapi_pages.jsonl"),
    )
    args = parser.parse_args()
    parse_dir(args.in_dir, args.out_file)


if __name__ == "__main__":
    main()

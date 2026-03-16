from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from tqdm import tqdm

from src.utils.io_utils import ensure_dir, write_jsonl
from src.utils.logging_utils import setup_logging


BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"}
NOISE_CLASS_PREFIXES = (
    "md-nav",
    "md-sidebar",
    "md-tabs",
    "md-header",
    "md-search",
    "md-path",
)
EXCLUDED_PAGE_IDS = {"release-notes"}


def _is_noise_tag(tag: Tag) -> bool:
    classes = tag.get("class", [])
    return any(any(cls.startswith(prefix) for prefix in NOISE_CLASS_PREFIXES) for cls in classes)


def _has_noise_ancestor(tag: Tag, root: Tag) -> bool:
    current = tag
    while current is not None and current is not root:
        if _is_noise_tag(current):
            return True
        current = current.parent if isinstance(current.parent, Tag) else None
    return False


def _collect_inline_text(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in BLOCK_TAGS:
            # Nested block tags are emitted separately by the main traversal.
            continue
        text = _collect_inline_text(child)
        if text:
            parts.append(text)
    return " ".join(" ".join(parts).split())


def _iter_text_blocks(root: Tag) -> list[str]:
    blocks: list[str] = []

    def visit(node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if _has_noise_ancestor(child, root):
                continue
            if child.name in BLOCK_TAGS:
                text = _collect_inline_text(child)
                if text:
                    blocks.append(text)
            visit(child)

    # Traverse the tree in document order and emit each block once, while
    # keeping inline formatting inside the surrounding sentence.
    visit(root)
    return blocks


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content and layout elements.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return soup.get_text(" ", strip=True)
    blocks = _iter_text_blocks(main)
    if not blocks:
        return main.get_text(" ", strip=True)
    return "\n".join(blocks)


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
        if html_file.stem in EXCLUDED_PAGE_IDS:
            continue
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

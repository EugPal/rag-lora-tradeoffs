from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging

ALLOWED_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "tr"}
NOISE_SUBSTRINGS = (
    "table of contents",
    "edit this page",
    "create issue",
    "feedback",
    "was this page helpful",
    "thanks for the feedback",
    "open an issue in the github repository",
    "last modified",
)


def is_selected_block(tag: Tag) -> bool:
    if tag.name not in ALLOWED_TAGS:
        return False
    parent = tag.parent
    while isinstance(parent, Tag):
        if parent.name in ALLOWED_TAGS:
            return False
        parent = parent.parent
    return True


def extract_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if not isinstance(main, Tag):
        return []

    lines: list[str] = []
    for tag in main.find_all(ALLOWED_TAGS):
        if not is_selected_block(tag):
            continue
        text = tag.get_text(" ", strip=True)
        text = " ".join(text.split())
        if not text:
            continue
        if tag.name and tag.name.startswith("h") and len(tag.name) == 2 and tag.name[1].isdigit():
            level = min(6, max(1, int(tag.name[1])))
            lines.append(f'{"#" * level} {text}')
        elif tag.name == "li":
            lines.append(f"- {text}")
        elif tag.name == "tr":
            lines.append(text.replace("  ", " | "))
        else:
            lines.append(text)
    return lines


def normalize_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        lowered = line.lower().strip()
        if not lowered:
            continue
        if lowered.startswith("## feedback") or lowered == "# feedback":
            break
        if any(noise in lowered for noise in NOISE_SUBSTRINGS):
            continue
        if lowered in {"yes", "no"}:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(line.strip())
    return cleaned


def title_from_html(html: str, lines: list[str]) -> str:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        text = " ".join(h1.get_text(" ", strip=True).split())
        if text:
            return text
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    title_tag = soup.find("title")
    if title_tag:
        text = " ".join(title_tag.get_text(" ", strip=True).split())
        return text.replace(" | Kubernetes", "").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Kubernetes HTML pages into structured plain text.")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("data/raw/kubernetes_html_manifest.jsonl"),
    )
    parser.add_argument("--in-dir", type=Path, default=Path("data/raw/kubernetes_html"))
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/kubernetes_pages.jsonl"),
    )
    args = parser.parse_args()

    logger = setup_logging("parse_kubernetes_docs")
    manifest_rows = read_jsonl(args.manifest_file)
    if not manifest_rows:
        logger.warning("No manifest rows found in %s", args.manifest_file)
        return

    parsed_rows: list[dict] = []
    pbar = tqdm(manifest_rows, desc="parse_kubernetes_docs", unit="page", disable=not sys.stderr.isatty())
    for row in pbar:
        html_file = args.in_dir / row["html_file"]
        if not html_file.exists():
            logger.warning("Missing HTML file: %s", html_file)
            continue
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        raw_lines = extract_lines(html)
        lines = normalize_lines(raw_lines)
        if not lines:
            continue
        parsed_rows.append(
            {
                "id": row["id"],
                "url": row["url"],
                "path": row["path"],
                "section": row["section"],
                "source": row["html_file"],
                "title": title_from_html(html, lines),
                "text": "\n".join(lines),
            }
        )

    write_jsonl(args.out_file, parsed_rows)
    logger.info("Parsed %d Kubernetes pages to %s", len(parsed_rows), args.out_file)


if __name__ == "__main__":
    main()

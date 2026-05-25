from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging

HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^\n]{0,120}\u00B6$")
REFERENCE_ANCHOR_RE = re.compile(
    r"^(?:class|exception|async\b|classmethod\b|property\b|alias of\b|[A-Za-z_][\w.]*\s*\()"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
INVENTORY_PHRASES = (
    "subpackages",
    "submodules",
    "module contents",
    "subpackage",
    "submodule",
)


def normalize_line(line: str) -> str:
    return " ".join(line.split())


def word_count(text: str) -> int:
    return len(text.split())


def clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    previous = ""
    for raw in text.splitlines():
        line = normalize_line(raw)
        if not line:
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return lines


def infer_page_kind(page: dict) -> str:
    title = (page.get("title") or "").lower()
    url = (page.get("url") or "").lower()
    if "/sdk/python/api" in url:
        if url.endswith("/api"):
            return "sdk_index"
        if " package" in title:
            return "sdk_package"
        return "sdk_reference"
    if any(token in url for token in ("/specification", "/definitions")):
        return "spec"
    if any(token in url for token in ("/topics/", "/tutorial", "/guides/", "/whats-new", "/announcing-")):
        return "narrative"
    return "narrative"


def should_exclude_page(page: dict, page_kind: str) -> str | None:
    title = (page.get("title") or "").lower()
    url = (page.get("url") or "").lower()
    text = (page.get("text") or "").lower()
    inventory_hits = sum(text.count(phrase) for phrase in INVENTORY_PHRASES)
    if page_kind == "sdk_index":
        return "sdk_api_index"
    if url.endswith("/sdk/python/api/a2a.html") and inventory_hits >= 8:
        return "root_package_inventory"
    if title == "a2a sdk" and inventory_hits >= 4:
        return "sdk_overview_inventory"
    return None


def is_heading_line(line: str) -> bool:
    if line.endswith("\u00B6"):
        return True
    return bool(HEADING_RE.match(line))


def is_reference_anchor(line: str) -> bool:
    return bool(REFERENCE_ANCHOR_RE.match(line)) and word_count(line) <= 24


def section_merge_min_words(page_kind: str) -> int:
    if page_kind == "spec":
        return 90
    if page_kind == "sdk_package":
        return 72
    if page_kind == "sdk_reference":
        return 60
    return 48


def chunk_budget_words(page_kind: str, default_max_words: int) -> int:
    if page_kind == "spec":
        return max(default_max_words, 260)
    if page_kind == "sdk_package":
        return max(default_max_words, 200)
    return default_max_words


def chunk_overlap_words(page_kind: str, narrative_overlap_words: int, reference_overlap_words: int) -> int:
    if page_kind == "spec":
        return max(narrative_overlap_words, 30)
    if page_kind.startswith("sdk"):
        return reference_overlap_words
    return narrative_overlap_words


def min_chunk_words(page_kind: str) -> int:
    if page_kind == "spec":
        return 80
    if page_kind == "sdk_package":
        return 60
    if page_kind == "sdk_reference":
        return 50
    return 40


def split_long_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    pieces: list[str] = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        piece_words = words[start : start + max_words]
        if not piece_words:
            break
        pieces.append(" ".join(piece_words))
        if start + max_words >= len(words):
            break
        start += step
    return pieces


def merge_small_sections(sections: list[dict[str, str]], min_words: int) -> list[dict[str, str]]:
    if not sections:
        return []
    merged: list[dict[str, str]] = [sections[0].copy()]
    for section in sections[1:]:
        if word_count(merged[-1]["text"]) < min_words:
            merged[-1]["text"] = f"{merged[-1]['text']}\n{section['text']}"
        else:
            merged.append(section.copy())
    return merged


def build_sections(lines: list[str], page_kind: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_anchor: str | None = None

    def flush() -> None:
        nonlocal current_lines, current_anchor
        if not current_lines:
            return
        text = "\n".join(current_lines)
        sections.append({"anchor": current_anchor or current_lines[0], "text": text})
        current_lines = []
        current_anchor = None

    for line in lines:
        boundary = is_heading_line(line)
        if page_kind.startswith("sdk"):
            boundary = boundary or is_reference_anchor(line)
        if boundary and current_lines:
            flush()
        if not current_lines:
            current_anchor = line if boundary else None
        current_lines.append(line)

    flush()
    return merge_small_sections(sections, min_words=section_merge_min_words(page_kind))


def tail_overlap_sections(sections: list[dict[str, str]], overlap_words: int) -> list[dict[str, str]]:
    if overlap_words <= 0 or not sections:
        return []
    combined = "\n\n".join(section["text"] for section in sections)
    words = combined.split()
    if len(words) <= overlap_words:
        overlap_text = combined
    else:
        overlap_text = " ".join(words[-overlap_words:])
    return [{"anchor": sections[-1]["anchor"], "text": overlap_text}]


def dedupe_sentences(text: str) -> str:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    if len(pieces) <= 1:
        return text
    deduped: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        key = normalize_line(piece).lower()
        if len(key) > 40 and key in seen:
            continue
        seen.add(key)
        deduped.append(piece)
    return " ".join(deduped)


def normalize_chunk_text(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    deduped_lines: list[str] = []
    seen_lines: set[str] = set()
    for line in lines:
        key = line.lower()
        if len(key) > 40 and key in seen_lines:
            continue
        seen_lines.add(key)
        deduped_lines.append(line)
    normalized = "\n\n".join(deduped_lines)
    return normalize_line(dedupe_sentences(normalized))


def merge_tiny_chunks(chunks: list[dict[str, str]], min_words_needed: int, max_words: int) -> list[dict[str, str]]:
    if not chunks:
        return []
    merged: list[dict[str, str]] = [chunks[0].copy()]
    for chunk in chunks[1:]:
        chunk_words = word_count(chunk["text"])
        prev_words = word_count(merged[-1]["text"])
        if chunk_words < min_words_needed and prev_words + chunk_words <= max_words + min_words_needed // 2:
            merged[-1]["text"] = f"{merged[-1]['text']}\n\n{chunk['text']}"
        else:
            merged.append(chunk.copy())
    return merged


def dedupe_adjacent_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    previous_text = ""
    for chunk in chunks:
        text = normalize_chunk_text(chunk["text"])
        if not text or text == previous_text:
            continue
        deduped.append({"anchor": chunk["anchor"], "text": text})
        previous_text = text
    return deduped


def pack_sections(
    sections: list[dict[str, str]],
    page_kind: str,
    max_words: int,
    overlap_words: int,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    current: list[dict[str, str]] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        text = "\n\n".join(section["text"] for section in current)
        chunks.append({"anchor": current[0]["anchor"], "text": text})
        current = []
        current_words = 0

    for section in sections:
        section_words = word_count(section["text"])
        if section_words > max_words:
            if current:
                flush()
            for piece in split_long_text(section["text"], max_words=max_words, overlap_words=overlap_words):
                chunks.append({"anchor": section["anchor"], "text": piece})
            continue

        if current and current_words + section_words > max_words:
            overlap = tail_overlap_sections(current, overlap_words)
            flush()
            current = overlap
            current_words = sum(word_count(item["text"]) for item in current)

        current.append(section.copy())
        current_words += section_words

    flush()
    chunks = merge_tiny_chunks(chunks, min_words_needed=min_chunk_words(page_kind), max_words=max_words)
    return dedupe_adjacent_chunks(chunks)


def build_chunks_for_page(
    page: dict,
    max_words: int,
    narrative_overlap_words: int,
    reference_overlap_words: int,
) -> list[dict[str, str]]:
    page_kind = infer_page_kind(page)
    lines = clean_lines(page.get("text", ""))
    if not lines:
        return []
    sections = build_sections(lines, page_kind=page_kind)
    budget = chunk_budget_words(page_kind, max_words)
    overlap_words = chunk_overlap_words(page_kind, narrative_overlap_words, reference_overlap_words)
    return pack_sections(sections, page_kind=page_kind, max_words=budget, overlap_words=overlap_words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a section-aware A2A chunk corpus.")
    parser.add_argument(
        "--in-file",
        type=Path,
        default=Path("data/processed/fresh_start/a2a_pages_reparsed.jsonl"),
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/fresh_start/docs_a2a_reparsed_semantic_v1_1.jsonl"),
    )
    parser.add_argument(
        "--excluded-pages-file",
        type=Path,
        default=Path("data/processed/fresh_start/a2a_pages_reparsed_semantic_v1_1_excluded.jsonl"),
    )
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--narrative-overlap-words", type=int, default=20)
    parser.add_argument("--reference-overlap-words", type=int, default=0)
    parser.add_argument(
        "--exclude-noisy-pages",
        action="store_true",
        help="Drop obvious SDK index pages that mostly contain inventories.",
    )
    args = parser.parse_args()

    logger = setup_logging("build_semantic_a2a_corpus")
    pages = read_jsonl(args.in_file)
    if not pages:
        logger.warning("No pages found in %s", args.in_file)
        return

    rows: list[dict] = []
    excluded_rows: list[dict] = []
    pbar = tqdm(pages, desc="build_semantic_a2a_corpus", unit="page", disable=not sys.stderr.isatty())
    for page in pbar:
        page_kind = infer_page_kind(page)
        exclusion_reason = should_exclude_page(page, page_kind) if args.exclude_noisy_pages else None
        if exclusion_reason:
            excluded_rows.append(
                {
                    "page_id": page["id"],
                    "title": page.get("title"),
                    "url": page.get("url"),
                    "page_kind": page_kind,
                    "reason": exclusion_reason,
                }
            )
            continue

        chunks = build_chunks_for_page(
            page,
            max_words=args.max_words,
            narrative_overlap_words=args.narrative_overlap_words,
            reference_overlap_words=args.reference_overlap_words,
        )
        for chunk_idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": f"{page['id']}-{chunk_idx}",
                    "page_id": page["id"],
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "text": chunk["text"],
                    "chunk_index": chunk_idx,
                    "page_kind": page_kind,
                    "section_anchor": chunk["anchor"],
                }
            )

    write_jsonl(args.out_file, rows)
    write_jsonl(args.excluded_pages_file, excluded_rows)
    logger.info("Wrote %d semantic chunks to %s", len(rows), args.out_file)
    logger.info("Excluded %d pages to %s", len(excluded_rows), args.excluded_pages_file)


if __name__ == "__main__":
    main()

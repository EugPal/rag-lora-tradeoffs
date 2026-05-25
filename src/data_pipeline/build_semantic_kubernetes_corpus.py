from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def normalize_line(line: str) -> str:
    return " ".join(line.split())


def word_count(text: str) -> int:
    return len(text.split())


def infer_page_kind(page: dict) -> str:
    url = (page.get("url") or "").lower()
    if "/docs/reference/generated/" in url or "/docs/reference/kubernetes-api/" in url:
        return "spec"
    if "/docs/reference/" in url:
        return "reference"
    if "/docs/tasks/" in url:
        return "task"
    if "/docs/tutorials/" in url:
        return "tutorial"
    if "/docs/setup/" in url:
        return "setup"
    if "/docs/concepts/" in url:
        return "concept"
    if "/docs/contribute/" in url:
        return "contribute"
    return "narrative"


def clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = normalize_line(raw)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def is_heading(line: str) -> bool:
    return line.startswith("#")


def section_merge_min_words(page_kind: str) -> int:
    if page_kind == "spec":
        return 100
    if page_kind == "reference":
        return 70
    return 50


def chunk_budget_words(page_kind: str, default_max_words: int) -> int:
    if page_kind == "spec":
        return max(default_max_words, 240)
    if page_kind == "reference":
        return max(default_max_words, 200)
    return default_max_words


def chunk_overlap_words(page_kind: str, narrative_overlap_words: int, reference_overlap_words: int) -> int:
    if page_kind in {"spec", "reference"}:
        return reference_overlap_words
    return narrative_overlap_words


def min_chunk_words(page_kind: str) -> int:
    if page_kind == "spec":
        return 90
    if page_kind == "reference":
        return 60
    return 40


def merge_small_sections(sections: list[dict[str, str]], min_words: int) -> list[dict[str, str]]:
    if not sections:
        return []
    merged: list[dict[str, str]] = [sections[0].copy()]
    for section in sections[1:]:
        if word_count(merged[-1]["text"]) < min_words:
            merged[-1]["text"] = f'{merged[-1]["text"]}\n{section["text"]}'
        else:
            merged.append(section.copy())
    return merged


def build_sections(lines: list[str], page_kind: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_anchor: str | None = None

    def flush() -> None:
        nonlocal current_anchor, current_lines
        if not current_lines:
            return
        sections.append({"anchor": current_anchor or current_lines[0], "text": "\n".join(current_lines)})
        current_anchor = None
        current_lines = []

    for line in lines:
        if is_heading(line) and current_lines:
            flush()
        if not current_lines:
            current_anchor = line if is_heading(line) else None
        current_lines.append(line)
    flush()
    return merge_small_sections(sections, section_merge_min_words(page_kind))


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


def tail_overlap_sections(sections: list[dict[str, str]], overlap_words: int) -> list[dict[str, str]]:
    if overlap_words <= 0 or not sections:
        return []
    combined = "\n\n".join(section["text"] for section in sections)
    words = combined.split()
    overlap_text = combined if len(words) <= overlap_words else " ".join(words[-overlap_words:])
    return [{"anchor": sections[-1]["anchor"], "text": overlap_text}]


def dedupe_adjacent_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    previous = ""
    for chunk in chunks:
        text = normalize_line(chunk["text"])
        if not text or text == previous:
            continue
        deduped.append({"anchor": chunk["anchor"], "text": text})
        previous = text
    return deduped


def merge_tiny_chunks(chunks: list[dict[str, str]], min_words_needed: int, max_words: int) -> list[dict[str, str]]:
    if not chunks:
        return []
    merged: list[dict[str, str]] = [chunks[0].copy()]
    for chunk in chunks[1:]:
        cw = word_count(chunk["text"])
        pw = word_count(merged[-1]["text"])
        if cw < min_words_needed and pw + cw <= max_words + max(20, min_words_needed // 2):
            merged[-1]["text"] = f'{merged[-1]["text"]}\n\n{chunk["text"]}'
        else:
            merged.append(chunk.copy())
    return merged


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
        chunks.append({"anchor": current[0]["anchor"], "text": "\n\n".join(item["text"] for item in current)})
        current = []
        current_words = 0

    for section in sections:
        sw = word_count(section["text"])
        if sw > max_words:
            if current:
                flush()
            for piece in split_long_text(section["text"], max_words=max_words, overlap_words=overlap_words):
                chunks.append({"anchor": section["anchor"], "text": piece})
            continue
        if current and current_words + sw > max_words:
            overlap = tail_overlap_sections(current, overlap_words)
            flush()
            current = overlap
            current_words = sum(word_count(item["text"]) for item in current)
        current.append(section.copy())
        current_words += sw
    flush()
    chunks = merge_tiny_chunks(chunks, min_chunk_words(page_kind), max_words)
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
    sections = build_sections(lines, page_kind)
    return pack_sections(
        sections,
        page_kind=page_kind,
        max_words=chunk_budget_words(page_kind, max_words),
        overlap_words=chunk_overlap_words(page_kind, narrative_overlap_words, reference_overlap_words),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a section-aware Kubernetes chunk corpus.")
    parser.add_argument(
        "--in-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/kubernetes_pages.jsonl"),
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/docs_kubernetes_semantic_v1.jsonl"),
    )
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--narrative-overlap-words", type=int, default=24)
    parser.add_argument("--reference-overlap-words", type=int, default=0)
    args = parser.parse_args()

    logger = setup_logging("build_semantic_kubernetes_corpus")
    pages = read_jsonl(args.in_file)
    if not pages:
        logger.warning("No pages found in %s", args.in_file)
        return

    rows: list[dict] = []
    pbar = tqdm(pages, desc="build_semantic_kubernetes_corpus", unit="page", disable=not sys.stderr.isatty())
    for page in pbar:
        page_kind = infer_page_kind(page)
        chunks = build_chunks_for_page(
            page,
            max_words=args.max_words,
            narrative_overlap_words=args.narrative_overlap_words,
            reference_overlap_words=args.reference_overlap_words,
        )
        for chunk_idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": f'{page["id"]}-{chunk_idx}',
                    "page_id": page["id"],
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "section": page.get("section"),
                    "text": chunk["text"],
                    "chunk_index": chunk_idx,
                    "page_kind": page_kind,
                    "section_anchor": chunk["anchor"],
                }
            )

    write_jsonl(args.out_file, rows)
    logger.info("Wrote %d Kubernetes chunks to %s", len(rows), args.out_file)


if __name__ == "__main__":
    main()

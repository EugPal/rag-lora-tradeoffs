from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
BAD_ANSWER_SUBSTRINGS = (
    "http://",
    "https://",
    "last modified",
    "thanks for the feedback",
    "edit this page",
    "create issue",
    "table of contents",
)
GENERIC_SUBJECTS = {"this", "it", "they", "these", "those", "there"}
TOKEN_STOPWORDS = {"what", "when", "where", "which", "that", "this", "these", "those", "documentation", "kubernetes"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    return " ".join(text.split())


def load_page_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def sentence_split(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def clean_subject(subject: str) -> str:
    subject = re.sub(r"^[Aa]n?\s+", "", subject.strip())
    subject = re.sub(r"^[Tt]he\s+", "", subject)
    return subject.strip(" .,:;")


def good_answer(sentence: str) -> bool:
    if not sentence:
        return False
    lower = sentence.lower()
    if any(bad in lower for bad in BAD_ANSWER_SUBSTRINGS):
        return False
    if sentence.startswith("#") or sentence.startswith("- "):
        return False
    if "`" in sentence or "{" in sentence or "}" in sentence:
        return False
    words = sentence.split()
    if len(words) < 8 or len(words) > 40:
        return False
    digit_count = sum(ch.isdigit() for ch in sentence)
    if digit_count / max(1, len(sentence)) > 0.15:
        return False
    return True


def definition_candidate(sentence: str) -> tuple[str, str] | None:
    match = re.match(r"^([A-Z][A-Za-z0-9()/.+\\-]*(?: [A-Z][A-Za-z0-9()/.+\\-]*){0,7}|[A-Z][^.!?]{0,60}?)\\s+(is|are)\\s+.+", sentence)
    if not match:
        return None
    subject = clean_subject(match.group(1))
    if not subject or normalize(subject) in GENERIC_SUBJECTS:
        return None
    if len(subject.split()) > 8:
        return None
    verb = match.group(2).lower()
    if verb == "is":
        return f"What is {subject}?", sentence
    return f"What are {subject}?", sentence


def capability_candidate(sentence: str) -> tuple[str, str] | None:
    match = re.match(
        r"^([A-Z][A-Za-z0-9()/.+\\-]*(?: [A-Z][A-Za-z0-9()/.+\\-]*){0,7}|[A-Z][^.!?]{0,60}?)\\s+(provides|allows|supports|enables|lets)\\s+.+",
        sentence,
    )
    if not match:
        return None
    subject = clean_subject(match.group(1))
    if not subject or normalize(subject) in GENERIC_SUBJECTS:
        return None
    verb = match.group(2).lower()
    if verb == "provides":
        question = f"What does {subject} provide?"
    elif verb == "allows":
        question = f"What does {subject} allow?"
    elif verb == "supports":
        question = f"What does {subject} support?"
    elif verb == "enables":
        question = f"What does {subject} enable?"
    else:
        question = f"What does {subject} let you do?"
    return question, sentence


def content_tokens(text: str) -> set[str]:
    tokens = {token for token in normalize(text).split() if len(token) >= 4 and token not in TOKEN_STOPWORDS}
    singularized = {token[:-1] for token in tokens if token.endswith("s") and len(token) > 4}
    return tokens | singularized


def anchor_candidate(anchor: str, sentence: str, page_kind: str) -> tuple[str, str] | None:
    anchor_text = anchor.lstrip("#").strip()
    if not anchor_text or len(anchor_text.split()) > 10:
        return None
    anchor_norm = normalize(anchor_text)
    if anchor_norm in {"overview", "concepts", "reference", "setup", "tasks", "tutorials", "design"}:
        return None
    if anchor_norm.startswith(("about ", "using ", "migrating ", "configuring ", "enabling ", "direct ", "manual ")):
        return None
    if " versus " in anchor_norm or anchor_norm.startswith("self registration"):
        return None
    anchor_tokens = content_tokens(anchor_text)
    overlap = anchor_tokens & content_tokens(sentence)
    if not anchor_tokens or (len(overlap) / max(1, len(anchor_tokens))) < 0.5:
        return None
    lower = sentence.lower()
    if page_kind not in {"concept", "reference", "spec"}:
        return None
    if any(marker in lower for marker in [" is ", " are ", " refers to ", " represented by ", " consists of "]):
        if anchor_text.endswith("s") and not anchor_text.endswith("ss"):
            return f"What are {anchor_text}?", sentence
        return f"What is {anchor_text}?", sentence
    if any(marker in lower for marker in [" provides ", " allows ", " supports ", " enables ", " lets ", " manages "]):
        return f"What does {anchor_text} do?", sentence
    return None


def generate_candidates(chunk: dict) -> list[tuple[str, str, str]]:
    text = chunk.get("text", "")
    anchor = str(chunk.get("section_anchor") or "")
    page_kind = str(chunk.get("page_kind") or "")
    sentences = [s for s in sentence_split(text) if good_answer(s)]
    candidates: list[tuple[str, str, str]] = []
    for sentence in sentences[:3]:
        for builder, label in ((definition_candidate, "definition"), (capability_candidate, "capability")):
            candidate = builder(sentence)
            if candidate is not None:
                candidates.append((candidate[0], candidate[1], label))
                break
        else:
            candidate = anchor_candidate(anchor, sentence, page_kind)
            if candidate is not None:
                candidates.append((candidate[0], candidate[1], "anchor"))
    return candidates


def assign_split(page_id: str, train_pages: set[str], eval_pages: set[str], test_pages: set[str]) -> str | None:
    if page_id in train_pages:
        return "train"
    if page_id in eval_pages:
        return "eval"
    if page_id in test_pages:
        return "test"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-based Kubernetes QA datasets without LLMs.")
    parser.add_argument(
        "--docs-file",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/docs_kubernetes_semantic_v1.jsonl"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/page_split_60_20_20"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/page_split_60_20_20"),
    )
    args = parser.parse_args()

    logger = setup_logging("build_kubernetes_qa_dataset")
    docs_rows = read_jsonl(args.docs_file)
    if not docs_rows:
        logger.warning("No docs rows found in %s", args.docs_file)
        return

    train_pages = load_page_ids(args.split_dir / "page_ids_train.txt")
    eval_pages = load_page_ids(args.split_dir / "page_ids_eval.txt")
    test_pages = load_page_ids(args.split_dir / "page_ids_test.txt")

    split_rows = {"train": [], "eval": [], "test": []}
    stats = {
        "scanned_chunks": 0,
        "kept_rows": 0,
        "by_split": {"train": 0, "eval": 0, "test": 0},
        "by_pattern": {},
    }
    seen_pairs: set[tuple[str, str]] = set()

    for chunk in docs_rows:
        stats["scanned_chunks"] += 1
        page_id = str(chunk.get("page_id") or "")
        split = assign_split(page_id, train_pages, eval_pages, test_pages)
        if split is None:
            continue
        for question, answer, pattern in generate_candidates(chunk):
            pair_key = (normalize(question), normalize(answer))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            qa_row = {
                "id": f'kubernetes-{split}-{len(split_rows[split]):05d}',
                "question": question,
                "answer": answer,
                "context_policy": "retriever_only",
                "source_chunk": chunk["id"],
                "source_page": page_id,
                "page_kind": chunk.get("page_kind"),
                "provenance": f"rule_based_{pattern}",
            }
            split_rows[split].append(qa_row)
            stats["kept_rows"] += 1
            stats["by_split"][split] += 1
            stats["by_pattern"][pattern] = stats["by_pattern"].get(pattern, 0) + 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "qa_kubernetes_train_manual_v1.jsonl", split_rows["train"])
    write_jsonl(args.out_dir / "qa_kubernetes_eval_manual_v1.jsonl", split_rows["eval"])
    write_jsonl(args.out_dir / "qa_kubernetes_test_manual_v1.jsonl", split_rows["test"])
    (args.out_dir / "qa_kubernetes_manual_v1_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote Kubernetes QA train=%d eval=%d test=%d",
        len(split_rows["train"]),
        len(split_rows["eval"]),
        len(split_rows["test"]),
    )


if __name__ == "__main__":
    main()

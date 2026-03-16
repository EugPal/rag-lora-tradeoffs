from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from src.data_pipeline.dataset_utils import infer_category, infer_section, normalize, source_page_id
from tqdm import tqdm

from src.rag.generator import BaseGenerator, GenerationConfig, HFGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


BAD_SUBSTRINGS = [
    "subscribe",
    "waiting list",
    "back to top",
    "http://",
    "https://",
]


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str | None:
    t = _strip_code_fences(text)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def _is_bad_text(text: str) -> bool:
    low = text.lower()
    if any(b in low for b in BAD_SUBSTRINGS):
        return True
    if re.search(r"\d{4,}", text):
        return True
    digits = sum(ch.isdigit() for ch in text)
    if len(text) > 0 and (digits / max(1, len(text))) > 0.15:
        return True
    return False


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_question(title: str) -> str:
    q = title.strip()
    q = re.sub(r"^\[(solved|closed)\]\s*", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()
    if not q.endswith("?"):
        q = f"{q}?"
    return q


def _extract_input_record(row: dict) -> tuple[str, str, str]:
    # Supports both raw discussions schema and processed real-user schema.
    raw_question = str(row.get("question", "") or row.get("title", "")).strip()
    question = clean_question(raw_question) if raw_question else ""

    rec_id = str(row.get("id", "")).strip()
    if not rec_id:
        discussion_number = row.get("discussion_number")
        if discussion_number is not None and str(discussion_number).strip():
            rec_id = f"real-user-{discussion_number}"
    if not rec_id:
        h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        rec_id = f"real-user-{h}"

    discussion_url = str(row.get("discussion_url", "") or row.get("url", "")).strip()
    return rec_id, question, discussion_url


def pick_answer(question: str, context: str) -> str | None:
    q_words = set(normalize(question).split())
    best: tuple[int, str] | None = None
    for sent in sentence_split(context):
        low = sent.lower()
        if any(b in low for b in BAD_SUBSTRINGS):
            continue
        words = sent.split()
        if not (8 <= len(words) <= 50):
            continue
        overlap = len(set(normalize(sent).split()) & q_words)
        if overlap <= 0:
            continue
        cand = " ".join(words[:50])
        if best is None or overlap > best[0]:
            best = (overlap, cand)
    if best:
        return best[1]
    return None


def _valid_llm_answer_reason(
    answer_quote: str,
    source_chunk: str,
    chunk_ids: list[str],
    contexts_by_chunk: dict[str, str],
) -> str | None:
    a = answer_quote.strip()
    if not a:
        return "empty_answer"
    if source_chunk not in chunk_ids:
        return "source_chunk_not_in_top_k"
    if len(a.split()) < 8 or len(a.split()) > 80:
        return "bad_answer_length"
    if _is_bad_text(a):
        return "bad_text"
    ctx = contexts_by_chunk.get(source_chunk, "")
    if not ctx or a not in ctx:
        return "quote_not_substring"
    return None


def _llm_pick_answer(
    generator: HFGenerator,
    question: str,
    chunk_ids: list[str],
    contexts: list[str],
    reason_counts: dict[str, int] | None = None,
) -> tuple[str, str] | None:
    context_blocks = []
    for idx, (chunk_id, ctx) in enumerate(zip(chunk_ids, contexts), start=1):
        context_blocks.append(f"[{idx}] chunk_id={chunk_id}\n{ctx}")
    prompt = (
        "You are creating doc-grounded QA from retrieved FastAPI documentation.\n"
        "Return ONLY valid JSON, no markdown.\n\n"
        f"QUESTION:\n{question}\n\n"
        "RETRIEVED_CONTEXTS:\n"
        + "\n\n".join(context_blocks)
        + "\n\nTask:\n"
        "- Select ONE best supporting chunk_id from the provided contexts.\n"
        "- Extract ONE exact answer quote from that selected context.\n"
        "- The quote must be contiguous text copied verbatim.\n\n"
        "Output JSON schema (must match exactly):\n"
        '{\n  "source_chunk": "<one chunk_id from input>",\n  "answer_quote": "<exact quote>"\n}\n'
    )
    raw = generator.generate(prompt)
    payload_str = _extract_first_json_object(raw)
    if not payload_str:
        if reason_counts is not None:
            reason_counts["no_json_object"] = reason_counts.get("no_json_object", 0) + 1
        return None
    try:
        payload = json.loads(payload_str)
    except Exception:
        if reason_counts is not None:
            reason_counts["json_parse_error"] = reason_counts.get("json_parse_error", 0) + 1
        return None
    source_chunk = str(payload.get("source_chunk", "")).strip()
    answer_quote = str(payload.get("answer_quote", "")).strip()
    if not source_chunk or not answer_quote:
        if reason_counts is not None:
            reason_counts["missing_fields"] = reason_counts.get("missing_fields", 0) + 1
        return None
    contexts_by_chunk = {cid: ctx for cid, ctx in zip(chunk_ids, contexts)}
    reason = _valid_llm_answer_reason(
        answer_quote=answer_quote,
        source_chunk=source_chunk,
        chunk_ids=chunk_ids,
        contexts_by_chunk=contexts_by_chunk,
    )
    if reason is not None:
        if reason_counts is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return None
    return source_chunk, answer_quote


def main() -> None:
    ap = argparse.ArgumentParser(description="Build real-user QA grounded in official docs.")
    ap.add_argument(
        "--discussions-file",
        "--in-file",
        dest="discussions_file",
        type=Path,
        default=Path("data/raw/github/fastapi_questions_answered.jsonl"),
        help="Input JSONL with either raw discussions (title/url/discussion_number) or processed real-user QA rows.",
    )
    ap.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    ap.add_argument("--index-file", type=Path, default=Path("data/embeddings/docs_embeddings.faiss"))
    ap.add_argument("--embeddings-file", type=Path, default=Path("data/embeddings/docs_embeddings.npy"))
    ap.add_argument("--out-file", type=Path, default=Path("data/processed/qa_real_user_full.jsonl"))
    ap.add_argument("--max-items", type=int, default=240)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--answer-mode",
        type=str,
        choices=["heuristic", "llm_quote"],
        default="heuristic",
        help="How to build answer from retrieved contexts.",
    )
    ap.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-max-tokens", type=int, default=256)
    ap.add_argument("--llm-max-attempts", type=int, default=2)
    ap.add_argument(
        "--show-progress",
        action="store_true",
        help="Force tqdm progress display even in non-interactive environments.",
    )
    ap.add_argument(
        "--stats-file",
        type=Path,
        default=None,
        help="Optional output JSON for build stats. Defaults to <out-file stem>_stats.json.",
    )
    args = ap.parse_args()

    logger = setup_logging("build_real_user_qa")
    rows = read_jsonl(args.discussions_file)
    if not rows:
        logger.warning("No input rows found in %s", args.discussions_file)
        return

    pipeline = RagPipeline(
        RagConfig(
            docs_path=args.docs_file,
            index_path=args.index_file,
            embeddings_path=args.embeddings_file,
            top_k=max(1, args.top_k),
            use_hf_generator=False,
        ),
        generator=BaseGenerator(),
    )

    llm_generator = None
    if args.answer_mode == "llm_quote":
        llm_generator = HFGenerator(
            GenerationConfig(
                model_name=args.llm_model,
                max_tokens=args.llm_max_tokens,
                temperature=0.0,
            )
        )

    stats: dict[str, object] = {
        "answer_mode": args.answer_mode,
        "max_items": args.max_items,
        "top_k": max(1, args.top_k),
        "scanned_questions": 0,
        "retrieved_ok": 0,
        "kept": 0,
        "skipped_duplicate_question": 0,
        "skipped_no_question": 0,
        "skipped_no_retrieval": 0,
        "skipped_no_source_page": 0,
        "skipped_no_answer": 0,
        "llm_failed": 0,
        "llm_fail_reasons": {},
    }
    llm_reasons: dict[str, int] = stats["llm_fail_reasons"]  # type: ignore[assignment]

    out: list[dict] = []
    seen_q: set[str] = set()
    pbar = tqdm(
        rows,
        desc="build_real_user_qa",
        unit="question",
        disable=(not args.show_progress and not sys.stderr.isatty()),
    )
    for row in pbar:
        stats["scanned_questions"] = int(stats["scanned_questions"]) + 1
        if len(out) >= args.max_items:
            break
        rec_id, question, discussion_url = _extract_input_record(row)
        if not question:
            stats["skipped_no_question"] = int(stats["skipped_no_question"]) + 1
            continue
        qn = normalize(question)
        if not qn or qn in seen_q:
            stats["skipped_duplicate_question"] = int(stats["skipped_duplicate_question"]) + 1
            continue
        contexts, scored = pipeline.retrieve(question)
        if not scored or not contexts:
            stats["skipped_no_retrieval"] = int(stats["skipped_no_retrieval"]) + 1
            continue
        stats["retrieved_ok"] = int(stats["retrieved_ok"]) + 1

        chunk_ids = [doc_id for doc_id, _ in scored]
        if args.answer_mode == "llm_quote":
            if llm_generator is None:
                raise RuntimeError("LLM mode is selected but generator is not initialized.")
            picked = None
            for _ in range(max(1, args.llm_max_attempts)):
                picked = _llm_pick_answer(
                    generator=llm_generator,
                    question=question,
                    chunk_ids=chunk_ids,
                    contexts=contexts,
                    reason_counts=llm_reasons,
                )
                if picked is not None:
                    break
            if picked is None:
                stats["llm_failed"] = int(stats["llm_failed"]) + 1
                continue
            source_chunk, answer = picked
        else:
            source_chunk = scored[0][0]
            answer = pick_answer(question, contexts[0])
            if not answer:
                stats["skipped_no_answer"] = int(stats["skipped_no_answer"]) + 1
                continue

        source_page = source_page_id(source_chunk)
        if not source_page:
            stats["skipped_no_source_page"] = int(stats["skipped_no_source_page"]) + 1
            continue
        out.append(
            {
                "id": rec_id,
                "question": question,
                "answer": answer,
                "source_chunk": source_chunk,
                "source_page": source_page,
                "section": infer_section(source_page),
                "category": infer_category(question, answer, source_page),
                "provenance": (
                    "github_discussion_answered_llm_quote"
                    if args.answer_mode == "llm_quote"
                    else "github_discussion_answered"
                ),
                "discussion_url": discussion_url,
                "retrieval_top_k": max(1, args.top_k),
            }
        )
        seen_q.add(qn)
        stats["kept"] = int(stats["kept"]) + 1

    write_jsonl(args.out_file, out)
    logger.info("Built real-user QA: %d rows -> %s", len(out), args.out_file)
    stats_file = args.stats_file
    if stats_file is None:
        stats_file = args.out_file.with_name(f"{args.out_file.stem}_stats.json")
    stats_file.write_text(
        json.dumps(
            {
                "discussions_file": str(args.discussions_file),
                "docs_file": str(args.docs_file),
                "out_file": str(args.out_file),
                "stats": stats,
                "llm_model": args.llm_model if args.answer_mode == "llm_quote" else None,
                "llm_max_tokens": args.llm_max_tokens if args.answer_mode == "llm_quote" else None,
                "llm_max_attempts": args.llm_max_attempts if args.answer_mode == "llm_quote" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote real-user QA build stats to %s", stats_file)


if __name__ == "__main__":
    main()

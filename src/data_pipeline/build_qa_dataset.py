from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

from src.rag.generator import GenerationConfig, HFGenerator
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def split_dataset(rows: list[dict], seed: int, ratios: tuple[float, float, float]):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


BAD_SUBSTRINGS = [
    "back to top",
    "next",
    "previous",
    "waiting list",
    "subscribe",
    "initializing search",
    "http://",
    "https://",
    "<html",
    "<script",
    "function(",
    "json.parse",
    "copied from",
]


def pick_answer(sentences: list[str], min_words: int, max_words: int) -> str | None:
    for s in sentences:
        low = s.lower()
        if any(b in low for b in BAD_SUBSTRINGS):
            continue
        words = s.split()
        if not (min_words <= len(words) <= max_words):
            continue
        return " ".join(words[:max_words]).strip()
    return None


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Remove leading ```json / ``` and trailing fences if present.
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    Best-effort extraction of the first JSON object from a model output.
    This avoids failing when the model adds a prefix/suffix.
    """
    t = _strip_code_fences(text)
    start = t.find("{")
    if start < 0:
        return None
    # Simple brace matching.
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
    # Heuristic: long digit runs often come from line-number dumps.
    if re.search(r"\d{4,}", text):
        return True
    # Heuristic: too many digits.
    digits = sum(ch.isdigit() for ch in text)
    if len(text) > 0 and (digits / max(1, len(text))) > 0.15:
        return True
    return False


def _valid_llm_qa_reason(question: str, answer_quote: str, context: str) -> str | None:
    q = question.strip()
    a = answer_quote.strip()
    if not q or not a:
        return "empty_question_or_answer"
    if len(q.split()) < 6 or len(q.split()) > 30:
        return "bad_question_length"
    if len(a.split()) < 8 or len(a.split()) > 60:
        return "bad_answer_length"
    if q.lower().startswith("according to the documentation"):
        return "generic_question_prefix"
    if "what does it say about" in q.lower():
        return "generic_question_anchor"
    if _is_bad_text(q) or _is_bad_text(a):
        return "bad_text"
    # Grounding: exact substring match.
    if a not in context:
        return "quote_not_substring"
    # Avoid trivial echo questions.
    qn = re.sub(r"\s+", " ", q.lower()).strip()
    an = re.sub(r"\s+", " ", a.lower()).strip()
    if qn == an or (an and an in qn):
        return "trivial_echo"
    return None


def _llm_generate_qa(
    generator: HFGenerator,
    context: str,
    reason_counts: dict[str, int] | None = None,
) -> tuple[str, str] | None:
    system = (
        "You generate dataset items for a RAG system. "
        "You must be strictly grounded in the provided CONTEXT. "
        "Return ONLY valid JSON. No markdown, no explanations."
    )
    prompt = (
        f"{system}\n\n"
        "CONTEXT:\n<<<\n"
        f"{context}\n"
        ">>>\n\n"
        "Task:\n"
        "- Create ONE high-quality user question that a FastAPI developer might ask and that can be answered using ONLY the CONTEXT.\n"
        "- Then extract the answer as an EXACT QUOTE from the CONTEXT (1-2 sentences). Do not paraphrase.\n"
        "- The question must be specific and meaningful.\n"
        "- The quote must be a contiguous substring of the CONTEXT.\n\n"
        "Output JSON schema (must match exactly):\n"
        '{\n  "question": "...",\n  "answer_quote": "...",\n  "answer_sentence_count": 1 or 2,\n  "topic_keywords": ["...", "...", "..."]\n}\n'
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
    q = str(payload.get("question", "")).strip()
    a = str(payload.get("answer_quote", "")).strip()
    if not q or not a:
        if reason_counts is not None:
            reason_counts["missing_fields"] = reason_counts.get("missing_fields", 0) + 1
        return None
    reason = _valid_llm_qa_reason(q, a, context)
    if reason is not None:
        if reason_counts is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return None
    return q, a


def build_qa_from_docs(
    rows: list[dict],
    max_qa: int,
    seed: int,
    min_answer_words: int,
    max_answer_words: int,
    use_llm: bool,
    llm_model: str,
    llm_max_tokens: int,
    llm_max_attempts: int,
    show_progress: bool,
    max_source_chunks: int | None,
) -> list[dict]:
    qa_rows = []
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    generator = None
    if use_llm:
        generator = HFGenerator(
            GenerationConfig(
                model_name=llm_model,
                max_tokens=llm_max_tokens,
                temperature=0.0,
            )
        )
    stats: dict[str, object] = {
        "scanned_chunks": 0,
        "kept": 0,
        "skipped_empty": 0,
        "skipped_no_sentences": 0,
        "skipped_no_answer": 0,
        "llm_failed": 0,
        "llm_fail_reasons": {},
    }
    llm_reasons: dict[str, int] = stats["llm_fail_reasons"]  # type: ignore[assignment]
    pbar = tqdm(
        rows,
        desc="build_qa_dataset",
        unit="chunk",
        disable=(not show_progress and not sys.stderr.isatty()),
    )
    for row in pbar:
        if max_source_chunks is not None and int(stats["scanned_chunks"]) >= max_source_chunks:
            break
        stats["scanned_chunks"] = int(stats["scanned_chunks"]) + 1
        context = row.get("text", "") or ""
        if not context.strip():
            stats["skipped_empty"] = int(stats["skipped_empty"]) + 1
            continue
        if use_llm and generator is not None:
            qa = None
            for _ in range(max(1, llm_max_attempts)):
                qa = _llm_generate_qa(generator, context, reason_counts=llm_reasons)
                if qa is not None:
                    break
            if qa is None:
                stats["llm_failed"] = int(stats["llm_failed"]) + 1
                continue
            question, answer = qa
        else:
            sentences = sentence_split(context)
            if not sentences:
                stats["skipped_no_sentences"] = int(stats["skipped_no_sentences"]) + 1
                continue
            answer = pick_answer(
                sentences,
                min_words=min_answer_words,
                max_words=max_answer_words,
            )
            if not answer:
                stats["skipped_no_answer"] = int(stats["skipped_no_answer"]) + 1
                continue
            anchor = " ".join(answer.split()[:6])
            question = f"According to the documentation, what does it say about {anchor}?"
        qa_rows.append(
            {
                "id": f"qa-{row.get('id', len(qa_rows))}",
                "question": question,
                "answer": answer,
                "source_chunk": row.get("id"),
                "provenance": "llm_quote" if use_llm else "heuristic_sentence",
            }
        )
        stats["kept"] = int(stats["kept"]) + 1
        if len(qa_rows) >= max_qa:
            break
    return qa_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA splits from docs.jsonl.")
    parser.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-qa", type=int, default=20)
    parser.add_argument("--min-answer-words", type=int, default=8)
    parser.add_argument("--max-answer-words", type=int, default=25)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use an instruction-tuned LLM to generate question + extractive quote answer (recommended for silver).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model name for LLM-based QA generation.",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=384)
    parser.add_argument("--llm-max-attempts", type=int, default=2)
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Force tqdm progress display even in non-interactive environments.",
    )
    parser.add_argument(
        "--max-source-chunks",
        type=int,
        default=-1,
        help="Cap how many input chunks are scanned (smoke/debug control). Use -1 for no cap.",
    )
    args = parser.parse_args()

    logger = setup_logging("build_qa_dataset")
    rows = read_jsonl(args.docs_file)
    if not rows:
        logger.warning("No docs found at %s", args.docs_file)
        return

    qa_rows, stats = build_qa_from_docs(
        rows,
        max_qa=args.max_qa,
        seed=args.seed,
        min_answer_words=args.min_answer_words,
        max_answer_words=args.max_answer_words,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
        llm_max_attempts=args.llm_max_attempts,
        show_progress=args.show_progress,
        max_source_chunks=(None if args.max_source_chunks < 0 else args.max_source_chunks),
    )
    if not qa_rows:
        logger.warning("No QA pairs generated from %s", args.docs_file)
        return

    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    train, val, test = split_dataset(qa_rows, args.seed, ratios)

    write_jsonl(args.out_dir / "qa_small.jsonl", qa_rows)
    write_jsonl(args.out_dir / "qa_train.jsonl", train)
    write_jsonl(args.out_dir / "qa_val.jsonl", val)
    write_jsonl(args.out_dir / "qa_test.jsonl", test)
    stats_out = args.out_dir / "qa_small_build_stats.json"
    stats_payload = {
        "mode": "llm_quote" if args.use_llm else "heuristic_sentence",
        "docs_file": str(args.docs_file),
        "max_qa": args.max_qa,
        "seed": args.seed,
        "llm_model": args.llm_model if args.use_llm else None,
        "llm_max_tokens": args.llm_max_tokens if args.use_llm else None,
        "llm_max_attempts": args.llm_max_attempts if args.use_llm else None,
        "max_source_chunks": (None if args.max_source_chunks < 0 else args.max_source_chunks),
        "stats": stats,
    }
    stats_out.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote qa_small=%d train=%d val=%d test=%d",
        len(qa_rows),
        len(train),
        len(val),
        len(test),
    )
    logger.info("Wrote QA build stats to %s", stats_out)


if __name__ == "__main__":
    main()

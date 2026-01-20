from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.evaluation.metrics import bertscore_f1, embedding_cosine, exact_match, f1_score
from src.evaluation.judge import JudgeConfig, LLMJudge
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline RAG.")
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/qa_test.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("experiments/baseline/results.json"))
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("experiments/baseline/predictions.jsonl"),
    )
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--bertscore", action="store_true", help="Enable BERTScore.")
    parser.add_argument("--bertscore-model", type=str, default="roberta-base")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring.")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument(
        "--judge-max-samples",
        type=int,
        default=200,
        help="Limit judge scoring to first N samples (use -1 for all).",
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    logger = setup_logging("eval_baseline")
    rows = read_jsonl(args.test_file)
    if not rows:
        logger.warning("No QA test data found at %s", args.test_file)
        return

    pipeline = RagPipeline(RagConfig(top_k=args.top_k))
    em_scores = []
    f1_scores = []
    embed_scores = []
    hit_scores = []  # source_chunk hit@k
    page_hit_scores = []  # source page (base id) hit@k
    mrr_scores = []  # source_chunk MRR@k
    page_mrr_scores = []  # source_page MRR@k
    judge_correctness = []
    judge_groundedness = []
    judge = LLMJudge(JudgeConfig(model_name=args.judge_model)) if args.judge else None
    def normalize(text: str) -> str:
        return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()

    def token_set(text: str) -> set[str]:
        tokens = [t for t in normalize(text) if len(t) > 2]
        return set(tokens)

    def base_chunk_id(chunk_id: str | None) -> str | None:
        if not chunk_id:
            return None
        # Remove trailing "-<digits>" to get page-level id.
        return re.sub(r"-\d+$", "", chunk_id)

    def overlap_count(answer: str, contexts: list[str]) -> int:
        answer_tokens = token_set(answer)
        if not answer_tokens:
            return 0
        max_overlap = 0
        for ctx in contexts:
            overlap = answer_tokens.intersection(token_set(ctx))
            max_overlap = max(max_overlap, len(overlap))
        return max_overlap

    def is_hit(answer: str, contexts: list[str], min_overlap: int = 3) -> bool:
        if not answer:
            return False
        answer_norm = " ".join(normalize(answer))
        for ctx in contexts:
            ctx_norm = " ".join(normalize(ctx))
            if answer_norm and answer_norm in ctx_norm:
                return True
        return overlap_count(answer, contexts) >= min_overlap

    def extract_double_quoted_spans(text: str) -> list[str]:
        # Extract "..." spans; keep moderately sized quotes to avoid grabbing whole contexts.
        spans = re.findall(r"\"([^\"]{8,200})\"", text)
        # Deduplicate while preserving order
        seen = set()
        out = []
        for s in spans:
            s = s.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def has_supported_quote(prediction: str, contexts: list[str]) -> bool:
        quotes = extract_double_quoted_spans(prediction)
        if not quotes:
            return False
        for q in quotes:
            if any(q in ctx for ctx in contexts):
                return True
        return False

    predictions = []
    pred_texts = []
    gold_texts = []
    quote_supported = []
    pbar = tqdm(rows, desc="eval_baseline", unit="qa", disable=not sys.stderr.isatty())
    for idx, row in enumerate(pbar):
        start = time.perf_counter()
        prediction, chunks, scored_ids = pipeline.answer_with_context(row["question"])
        latency_s = time.perf_counter() - start
        em_scores.append(exact_match(prediction, row["answer"]))
        f1_scores.append(f1_score(prediction, row["answer"]))
        embed_scores.append(embedding_cosine(prediction, row["answer"], model_name=args.embed_model))
        pred_texts.append(prediction)
        gold_texts.append(row["answer"])
        quote_supported.append(1.0 if has_supported_quote(prediction, chunks) else 0.0)
        source_chunk = row.get("source_chunk")
        retrieved_ids = [doc_id for doc_id, _score in scored_ids]
        source_hit = 1.0 if (source_chunk is not None and source_chunk in retrieved_ids) else 0.0
        if source_chunk is not None and source_chunk in retrieved_ids:
            rank = retrieved_ids.index(source_chunk) + 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)

        source_page = base_chunk_id(source_chunk)
        retrieved_pages = {base_chunk_id(doc_id) for doc_id in retrieved_ids}
        page_hit = 1.0 if (source_page is not None and source_page in retrieved_pages) else 0.0
        if source_page is not None:
            page_rank = 0
            for i, doc_id in enumerate(retrieved_ids, start=1):
                if base_chunk_id(doc_id) == source_page:
                    page_rank = i
                    break
            page_mrr_scores.append(1.0 / page_rank if page_rank else 0.0)
        else:
            page_mrr_scores.append(0.0)

        judge_result = None
        if judge is not None and (args.judge_max_samples < 0 or idx < args.judge_max_samples):
            judge_result = judge.judge(row["question"], prediction, chunks)
            overlap = overlap_count(row["answer"], chunks)
            raw_evidence = judge_result.get("evidence")
            if isinstance(raw_evidence, list):
                raw_evidence = " ".join(str(item) for item in raw_evidence)
            evidence = (raw_evidence or "").strip()
            evidence_in_context = any(evidence and evidence in ctx for ctx in chunks)

            # Make groundedness strict: high groundedness requires a literal quoteable evidence.
            if judge_result.get("groundedness") is not None and judge_result["groundedness"] >= 4:
                if not evidence_in_context:
                    judge_result["groundedness"] = 2

            # Make correctness strict too: high correctness requires quoteable evidence.
            if judge_result.get("correctness") is not None and judge_result["correctness"] >= 4:
                if not evidence_in_context:
                    judge_result["correctness"] = 3

            # If the labeled source chunk wasn't retrieved, groundedness can't be high.
            if source_chunk and source_chunk not in retrieved_ids:
                if judge_result.get("groundedness") is not None and judge_result["groundedness"] >= 4:
                    judge_result["groundedness"] = 2
                if judge_result.get("correctness") is not None and judge_result["correctness"] >= 4:
                    judge_result["correctness"] = 3

            if not evidence and overlap < 3:
                if judge_result.get("groundedness") is None or judge_result["groundedness"] >= 4:
                    judge_result["groundedness"] = 1
                if judge_result.get("correctness") is None or judge_result["correctness"] >= 4:
                    judge_result["correctness"] = 1
            if judge_result.get("correctness") is not None:
                judge_correctness.append(judge_result["correctness"])
            if judge_result.get("groundedness") is not None:
                judge_groundedness.append(judge_result["groundedness"])
        hit_scores.append(source_hit)
        page_hit_scores.append(page_hit)
        predictions.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "gold": row["answer"],
                "prediction": prediction,
                "chunks": chunks,
                "source_chunk": source_chunk,
                "source_chunk_hit": source_hit,
                "source_page": source_page,
                "source_page_hit": page_hit,
                "retrieval": [{"id": doc_id, "score": score} for doc_id, score in scored_ids],
                "latency_s": latency_s,
                "embed_cosine": embed_scores[-1],
                "bertscore_f1": None,
                "quote_supported": quote_supported[-1],
                "judge": judge_result,
            }
        )

    bert_scores = None
    if args.bertscore:
        bert_scores = bertscore_f1(
            pred_texts,
            gold_texts,
            model_type=args.bertscore_model,
        )
        for item, score in zip(predictions, bert_scores):
            item["bertscore_f1"] = score

    results = {
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "embed_cosine": sum(embed_scores) / len(embed_scores),
        "bertscore_f1": sum(bert_scores) / len(bert_scores) if bert_scores else None,
        "retrieval_hit_rate": sum(hit_scores) / len(hit_scores),
        "retrieval_hit_rate_page": sum(page_hit_scores) / len(page_hit_scores),
        "retrieval_mrr": sum(mrr_scores) / len(mrr_scores),
        "retrieval_mrr_page": sum(page_mrr_scores) / len(page_mrr_scores),
        "quote_support_rate": sum(quote_supported) / len(quote_supported),
        "judge_correctness_avg": sum(judge_correctness) / len(judge_correctness)
        if judge_correctness
        else None,
        "judge_groundedness_avg": sum(judge_groundedness) / len(judge_groundedness)
        if judge_groundedness
        else None,
        "judge_scored": len(judge_correctness),
        "qualitative_check": "manual spot-check recommended",
        "samples": len(rows),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_jsonl(args.predictions_file, predictions)
    logger.info("Saved results to %s", args.out_file)
    logger.info("Saved predictions to %s", args.predictions_file)


if __name__ == "__main__":
    main()

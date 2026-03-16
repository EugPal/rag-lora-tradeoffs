from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.evaluation.metrics import bertscore_f1, embedding_cosine, exact_match, f1_components
from src.evaluation.judge import JudgeConfig, LLMJudge
from src.rag.generator import GenerationConfig, HFGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline RAG.")
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to evaluation dataset (e.g., data/processed/qa_eval_main.jsonl or qa_test_main.jsonl).",
    )
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
        "--no-quant-judge",
        action="store_true",
        help="Disable 4-bit quantization for judge model (uses fp16 on GPU).",
    )
    parser.add_argument(
        "--no-quant-generator",
        action="store_true",
        help="Disable 4-bit quantization for generator model (uses fp16 on GPU).",
    )
    parser.add_argument(
        "--judge-max-samples",
        type=int,
        default=200,
        help="Limit judge scoring to first N samples (use -1 for all).",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Path to a PEFT LoRA adapter directory to load for generation.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hybrid-retrieval", action="store_true")
    parser.add_argument("--vector-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument(
        "--retrieve-top-n",
        type=int,
        default=0,
        help="Initial retriever depth before reranking (0 means use top-k).",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=None,
        help="Path to trained linear reranker JSON model.",
    )
    parser.add_argument(
        "--pretrained-reranker-model",
        type=str,
        default=None,
        help="HF model name/path for pretrained cross-encoder reranker (e.g. BAAI/bge-reranker-base).",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help="Batch size for pretrained reranker scoring.",
    )
    parser.add_argument(
        "--retrieval-supervision-file",
        type=Path,
        default=None,
        help="Optional JSONL with positive_chunks/negative_types for retrieval diagnostics.",
    )
    args = parser.parse_args()

    logger = setup_logging("eval_baseline")
    rows = read_jsonl(args.test_file)
    if not rows:
        logger.warning("No QA test data found at %s", args.test_file)
        return

    rows_with_label_context = sum(
        1
        for r in rows
        if r.get("source_chunk") is not None or r.get("context_chunks") is not None
    )
    logger.info(
        "Context policy: retriever-only for generation; dataset source_chunk/context_chunks are metric-only. rows_with_label_fields=%d/%d",
        rows_with_label_context,
        len(rows),
    )

    generator = None
    if args.lora_adapter:
        generator = HFGenerator(
            GenerationConfig(
                lora_adapter_dir=args.lora_adapter,
                use_4bit=not args.no_quant_generator,
            )
        )
    pipeline = RagPipeline(
        RagConfig(
            top_k=args.top_k,
            use_hybrid_retrieval=args.hybrid_retrieval,
            vector_top_n=args.vector_top_n,
            bm25_top_n=args.bm25_top_n,
            retrieve_top_n=args.retrieve_top_n,
            reranker_model_path=args.reranker_model,
            pretrained_reranker_model=args.pretrained_reranker_model,
            reranker_batch_size=args.reranker_batch_size,
            use_4bit_generator=not args.no_quant_generator,
        ),
        generator=generator,
    )
    em_scores = []
    f1_scores = []
    f1_precision_scores = []
    f1_recall_scores = []
    embed_scores = []
    hit_scores = []  # source_chunk hit@k
    page_hit_scores = []  # source page (base id) hit@k
    mrr_scores = []  # source_chunk MRR@k
    page_mrr_scores = []  # source_page MRR@k
    positive_recall_retrieve_n_scores = []  # Recall@retrieve_top_n over labeled positive_chunks
    positive_recall_scores = []  # Recall@k over labeled positive_chunks
    hard_negative_top1_errors = []  # top-1 belongs to hard negative
    judge_raw_correctness = []
    judge_raw_groundedness = []
    judge_checked_correctness = []
    judge_checked_groundedness = []
    supervision_by_id: dict[str, dict] = {}
    if args.retrieval_supervision_file:
        supervision_by_id = {
            str(r.get("id")): r
            for r in read_jsonl(args.retrieval_supervision_file)
            if r.get("id") is not None
        }
    judge = (
        LLMJudge(
            JudgeConfig(
                model_name=args.judge_model,
                use_4bit=not args.no_quant_judge,
            )
        )
        if args.judge
        else None
    )
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

    def extract_final_answer(text: str) -> str:
        if not text:
            return ""
        m = re.search(r"(?im)^\s*final[_\s-]*answer\s*:\s*(.+)$", text)
        if m:
            return m.group(1).strip()
        # Fallback to raw prediction when model didn't emit FINAL_ANSWER field.
        return text.strip()

    predictions = []
    pred_texts = []
    gold_texts = []
    quote_supported = []
    pbar = tqdm(rows, desc="eval_baseline", unit="qa", disable=not sys.stderr.isatty())
    for idx, row in enumerate(pbar):
        start = time.perf_counter()
        # Leakage guard: generation input is question + retriever contexts only.
        question = row["question"]
        prediction, chunks, scored_ids, initial_scored_ids = pipeline.answer_with_context_stages(
            question
        )
        latency_s = time.perf_counter() - start
        prediction_for_metrics = extract_final_answer(prediction)
        em_scores.append(exact_match(prediction_for_metrics, row["answer"]))
        f1_precision, f1_recall, f1_value = f1_components(prediction_for_metrics, row["answer"])
        f1_precision_scores.append(f1_precision)
        f1_recall_scores.append(f1_recall)
        f1_scores.append(f1_value)
        embed_scores.append(
            embedding_cosine(prediction_for_metrics, row["answer"], model_name=args.embed_model)
        )
        pred_texts.append(prediction_for_metrics)
        gold_texts.append(row["answer"])
        quote_supported.append(1.0 if has_supported_quote(prediction, chunks) else 0.0)
        source_chunk = row.get("source_chunk")
        supervision = supervision_by_id.get(str(row.get("id")))
        positive_chunks = row.get("positive_chunks") or (
            supervision.get("positive_chunks") if supervision else []
        )
        hard_negative_chunks = (
            (row.get("negative_types") or {}).get("hard", [])
            or ((supervision.get("negative_types") or {}).get("hard", []) if supervision else [])
        )
        retrieved_ids = [doc_id for doc_id, _score in scored_ids]
        retrieved_initial_ids = [doc_id for doc_id, _score in initial_scored_ids]
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
        if positive_chunks:
            positives = set(positive_chunks)
            positive_hits = sum(1 for chunk_id in positives if chunk_id in retrieved_ids)
            positive_recall_scores.append(positive_hits / max(1, len(positives)))
            positive_hits_initial = sum(
                1 for chunk_id in positives if chunk_id in retrieved_initial_ids
            )
            positive_recall_retrieve_n_scores.append(
                positive_hits_initial / max(1, len(positives))
            )
        if hard_negative_chunks:
            hard_negative_top1_errors.append(
                1.0 if (retrieved_ids and retrieved_ids[0] in set(hard_negative_chunks)) else 0.0
            )

        judge_result_raw = None
        judge_result_checked = None
        if judge is not None and (args.judge_max_samples < 0 or idx < args.judge_max_samples):
            judge_result_raw = judge.judge(row["question"], prediction_for_metrics, chunks)
            judge_result_checked = dict(judge_result_raw)
            overlap = overlap_count(row["answer"], chunks)
            raw_evidence = judge_result_checked.get("evidence")
            if isinstance(raw_evidence, list):
                raw_evidence = " ".join(str(item) for item in raw_evidence)
            evidence = (raw_evidence or "").strip()
            evidence_in_context = any(evidence and evidence in ctx for ctx in chunks)

            raw_correctness = judge_result_raw.get("correctness")
            raw_groundedness = judge_result_raw.get("groundedness")
            if raw_correctness is not None:
                judge_raw_correctness.append(raw_correctness)
            if raw_groundedness is not None:
                judge_raw_groundedness.append(raw_groundedness)

            # Soft check 1: high groundedness/correctness requires quoteable evidence.
            if not evidence_in_context:
                if judge_result_checked.get("groundedness") is not None:
                    if judge_result_checked["groundedness"] >= 5:
                        judge_result_checked["groundedness"] = 3
                    elif judge_result_checked["groundedness"] == 4:
                        judge_result_checked["groundedness"] = 3
                if judge_result_checked.get("correctness") is not None:
                    if judge_result_checked["correctness"] >= 5:
                        judge_result_checked["correctness"] = 4
                    elif judge_result_checked["correctness"] == 4:
                        judge_result_checked["correctness"] = 3

            # Soft check 2: if no chunk from labeled source page is retrieved, cap high scores.
            if source_page and source_page not in retrieved_pages:
                if judge_result_checked.get("groundedness") is not None:
                    if judge_result_checked["groundedness"] >= 4:
                        judge_result_checked["groundedness"] = min(
                            judge_result_checked["groundedness"], 3
                        )
                if judge_result_checked.get("correctness") is not None:
                    if judge_result_checked["correctness"] >= 5:
                        judge_result_checked["correctness"] = 4

            # Soft check 3: if no evidence and weak lexical support, degrade (don't hard-force to 1).
            if not evidence and overlap < 3:
                if judge_result_checked.get("groundedness") is None:
                    judge_result_checked["groundedness"] = 1
                else:
                    judge_result_checked["groundedness"] = max(
                        1, int(judge_result_checked["groundedness"]) - 2
                    )
                if judge_result_checked.get("correctness") is None:
                    judge_result_checked["correctness"] = 1
                else:
                    judge_result_checked["correctness"] = max(
                        1, int(judge_result_checked["correctness"]) - 1
                    )
            if judge_result_checked.get("correctness") is not None:
                judge_checked_correctness.append(judge_result_checked["correctness"])
            if judge_result_checked.get("groundedness") is not None:
                judge_checked_groundedness.append(judge_result_checked["groundedness"])
        hit_scores.append(source_hit)
        page_hit_scores.append(page_hit)
        predictions.append(
            {
                "id": row.get("id"),
                "question": question,
                "context_policy": "retriever_only",
                "gold": row["answer"],
                "prediction": prediction,
                "chunks": chunks,
                "source_chunk": source_chunk,
                "source_chunk_hit": source_hit,
                "source_page": source_page,
                "source_page_hit": page_hit,
                "retrieval": [{"id": doc_id, "score": score} for doc_id, score in scored_ids],
                "retrieval_initial": [
                    {"id": doc_id, "score": score} for doc_id, score in initial_scored_ids
                ],
                "positive_chunks": positive_chunks,
                "hard_negative_chunks": hard_negative_chunks,
                "positive_recall_at_retrieve_n": positive_recall_retrieve_n_scores[-1]
                if positive_chunks
                else None,
                "positive_recall_at_k": positive_recall_scores[-1] if positive_chunks else None,
                "hard_negative_top1_error": hard_negative_top1_errors[-1]
                if hard_negative_chunks
                else None,
                "latency_s": latency_s,
                "embed_cosine": embed_scores[-1],
                "bertscore_f1": None,
                "quote_supported": quote_supported[-1],
                "judge": judge_result_checked,
                "judge_raw": judge_result_raw,
                "judge_checked": judge_result_checked,
                "lora_adapter": args.lora_adapter,
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
        "lora_adapter": args.lora_adapter,
        "reranker_model": str(args.reranker_model) if args.reranker_model else None,
        "pretrained_reranker_model": args.pretrained_reranker_model,
        "reranker_batch_size": args.reranker_batch_size,
        "retrieve_top_n": args.retrieve_top_n,
        "hybrid_retrieval": args.hybrid_retrieval,
        "vector_top_n": args.vector_top_n,
        "bm25_top_n": args.bm25_top_n,
        "retrieval_supervision_file": str(args.retrieval_supervision_file)
        if args.retrieval_supervision_file
        else None,
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "f1_precision": sum(f1_precision_scores) / len(f1_precision_scores),
        "f1_recall": sum(f1_recall_scores) / len(f1_recall_scores),
        "embed_cosine": sum(embed_scores) / len(embed_scores),
        "bertscore_f1": sum(bert_scores) / len(bert_scores) if bert_scores else None,
        "retrieval_hit_rate": sum(hit_scores) / len(hit_scores),
        "retrieval_hit_rate_page": sum(page_hit_scores) / len(page_hit_scores),
        "retrieval_mrr": sum(mrr_scores) / len(mrr_scores),
        "retrieval_mrr_page": sum(page_mrr_scores) / len(page_mrr_scores),
        "retrieval_recall_positive_chunks_at_retrieve_n": sum(positive_recall_retrieve_n_scores)
        / len(positive_recall_retrieve_n_scores)
        if positive_recall_retrieve_n_scores
        else None,
        "rows_with_positive_chunks_at_retrieve_n": len(positive_recall_retrieve_n_scores),
        "retrieval_recall_positive_chunks_at_k": sum(positive_recall_scores)
        / len(positive_recall_scores)
        if positive_recall_scores
        else None,
        "rows_with_positive_chunks": len(positive_recall_scores),
        "hard_negative_top1_error_rate": sum(hard_negative_top1_errors)
        / len(hard_negative_top1_errors)
        if hard_negative_top1_errors
        else None,
        "rows_with_hard_negatives": len(hard_negative_top1_errors),
        "quote_support_rate": sum(quote_supported) / len(quote_supported),
        "judge_raw_correctness_avg": sum(judge_raw_correctness) / len(judge_raw_correctness)
        if judge_raw_correctness
        else None,
        "judge_raw_groundedness_avg": sum(judge_raw_groundedness) / len(judge_raw_groundedness)
        if judge_raw_groundedness
        else None,
        "judge_raw_scored": len(judge_raw_correctness),
        "judge_checked_correctness_avg": sum(judge_checked_correctness) / len(judge_checked_correctness)
        if judge_checked_correctness
        else None,
        "judge_checked_groundedness_avg": sum(judge_checked_groundedness) / len(judge_checked_groundedness)
        if judge_checked_groundedness
        else None,
        "judge_checked_scored": len(judge_checked_correctness),
        "accuracy_raw": sum(judge_raw_correctness) / len(judge_raw_correctness)
        if judge_raw_correctness
        else None,
        "groundedness_raw": sum(judge_raw_groundedness) / len(judge_raw_groundedness)
        if judge_raw_groundedness
        else None,
        "accuracy_checked": sum(judge_checked_correctness) / len(judge_checked_correctness)
        if judge_checked_correctness
        else None,
        "groundedness_checked": sum(judge_checked_groundedness) / len(judge_checked_groundedness)
        if judge_checked_groundedness
        else None,
        # Backward-compatible aliases: keep these as checked metrics.
        "judge_correctness_avg": sum(judge_checked_correctness) / len(judge_checked_correctness)
        if judge_checked_correctness
        else None,
        "judge_groundedness_avg": sum(judge_checked_groundedness) / len(judge_checked_groundedness)
        if judge_checked_groundedness
        else None,
        "judge_scored": len(judge_checked_correctness),
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

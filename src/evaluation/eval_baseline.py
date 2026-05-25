from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time

import numpy as np
import torch
from pathlib import Path

from tqdm import tqdm

from src.evaluation.metrics import bertscore_f1, embedding_cosine, exact_match, f1_components
from src.evaluation.judge import JudgeConfig, LLMJudge
from src.rag.generator import GenerationConfig, HFGenerator
from src.rag.rag_pipeline import (
    RagConfig,
    RagPipeline,
    build_messages,
    predict_answer_mode,
    resolve_answer_mode,
)
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline RAG.")
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to evaluation dataset (e.g., data/processed/fresh_start/page_split_60_20_20/qa_a2a_eval_manual_v3_3.jsonl or qa_a2a_test_manual_v1.jsonl).",
    )
    parser.add_argument("--out-file", type=Path, default=Path("experiments/baseline/results.json"))
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("experiments/baseline/predictions.jsonl"),
    )
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--generator-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Generator model name/path.",
    )
    parser.add_argument("--bertscore", action="store_true", help="Enable BERTScore.")
    parser.add_argument("--bertscore-model", type=str, default="roberta-base")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring.")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
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
    parser.add_argument(
        "--answer-mode-source",
        type=str,
        choices=["none", "router", "oracle", "exact", "normal", "explicit_grounded"],
        default="none",
        help="How to choose answer_mode during generation. none uses the neutral mainline prompt; router uses the built-in rule-based router; oracle uses the gold dataset label.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inference.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable model thinking mode if tokenizer supports it.",
    )
    parser.add_argument("--hybrid-retrieval", action="store_true")
    parser.add_argument("--vector-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument("--native-sparse-retrieval", action="store_true")
    parser.add_argument("--sparse-top-n", type=int, default=20)
    parser.add_argument(
        "--retrieve-top-n",
        type=int,
        default=30,
        help="Initial retriever depth before reranking.",
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
        default="BAAI/bge-reranker-v2-m3",
        help="HF model name/path for pretrained cross-encoder reranker.",
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    generator = HFGenerator(
        GenerationConfig(
            max_tokens=args.max_new_tokens,
            enable_thinking=(False if args.disable_thinking else None),
            model_name=args.generator_model,
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
            use_bge_m3_native_sparse=args.native_sparse_retrieval,
            sparse_top_n=args.sparse_top_n,
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
    oracle_em_scores = []
    oracle_f1_scores = []
    oracle_f1_precision_scores = []
    oracle_f1_recall_scores = []
    router_match_scores = []
    predicted_mode_counts = {"exact": 0, "normal": 0}
    generation_mode_counts = {"none": 0, "exact": 0, "normal": 0, "explicit_grounded": 0}
    mode_metrics: dict[str, dict[str, list[float]]] = {
        'exact': {'em': [], 'f1': [], 'oracle_em': [], 'oracle_f1': []},
        'normal': {'em': [], 'f1': [], 'oracle_em': [], 'oracle_f1': []},
    }
    embed_scores = []
    hit_scores = []  # source_chunk hit@k
    page_hit_scores = []  # source page (base id) hit@k
    mrr_scores = []  # source_chunk MRR@k
    page_mrr_scores = []  # source_page MRR@k
    positive_recall_retrieve_n_scores = []  # Recall@retrieve_top_n over labeled positive_chunks
    positive_recall_scores = []  # Recall@k over labeled positive_chunks
    claim_support_recall_retrieve_n_scores = []  # Claim recall@retrieve_top_n over support_claims
    claim_support_recall_scores = []  # Claim recall@k over support_claims
    full_claim_support_retrieve_n_scores = []  # All claims supported@retrieve_top_n
    full_claim_support_scores = []  # All claims supported@k
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


    def _normalize_mojibake_common(text: str) -> str:
        # Common UTF-8/CP1251 mojibake artifacts seen in predictions/logs.
        replacements = {
            "???????": "'",
            "??????": "'",
            "??????": '"',
            "??????": '"',
            "???????": "-",
            "???????": "-",
            "????????": "'",
            "???????": "'",
            "???????": '"',
            "???????": '"',
            "????????": "-",
            "????????": "-",
            "??": "",
        }
        out = text or ""
        for bad, good in replacements.items():
            out = out.replace(bad, good)
        return out

    def _normalize_for_match(text: str) -> str:
        text = _normalize_mojibake_common(text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_quoted_spans(text: str) -> list[str]:
        # Support both "..." and '...' quoted extractive spans.
        spans = []
        for m in re.finditer(r"\"([^\"]{8,240})\"|'([^']{8,240})'", text or ""):
            s = (m.group(1) or m.group(2) or "").strip()
            if s:
                spans.append(s)
        out = []
        seen = set()
        for s in spans:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def has_context_support(clean_prediction: str, raw_prediction: str, contexts: list[str]) -> bool:
        if not clean_prediction or clean_prediction == "NOT_FOUND":
            return False
        ctx_norms = [_normalize_for_match(ctx) for ctx in contexts if ctx]
        if not ctx_norms:
            return False

        # 1) Quoted support (double or single quotes) from raw prediction.
        for span in extract_quoted_spans(raw_prediction or ""):
            span_norm = _normalize_for_match(span)
            if span_norm and any(span_norm in ctx for ctx in ctx_norms):
                return True

        # 2) Unquoted exact support for cleaned prediction.
        pred_norm = _normalize_for_match(clean_prediction)
        if pred_norm and any(pred_norm in ctx for ctx in ctx_norms):
            return True

        # 3) Minimal unquoted extractive support via n-gram overlap.
        pred_tokens = [t for t in pred_norm.split() if len(t) >= 2]
        if len(pred_tokens) < 3:
            return False
        n = min(6, max(3, len(pred_tokens) // 2))
        pred_ngrams = {tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)}
        if not pred_ngrams:
            return False
        for ctx in ctx_norms:
            ctx_tokens = [t for t in ctx.split() if len(t) >= 2]
            if len(ctx_tokens) < n:
                continue
            ctx_ngrams = {tuple(ctx_tokens[i : i + n]) for i in range(len(ctx_tokens) - n + 1)}
            if pred_ngrams.intersection(ctx_ngrams):
                return True
        return False

    def _oracle_context_from_row(row: dict) -> tuple[list[str], str | None]:
        source_chunk = row.get("source_chunk")
        if isinstance(source_chunk, str):
            txt = pipeline._doc_lookup.get(source_chunk, "")
            if txt:
                return [txt], source_chunk
        return [], None

    def extract_final_answer(text: str) -> tuple[str, dict[str, bool]]:
        flags = {
            "multi_sentence_raw": False,
            "loop_cleaned": False,
            "not_found_after_cleanup": False,
        }
        if not text:
            flags["not_found_after_cleanup"] = True
            return "NOT_FOUND", flags

        # Accept label variants like FINAL_ANSWER, FINAL ANSWER, and forms with
        # non-word unicode modifiers before colon (e.g. FINAL_ANSWER\u0e4c:).
        label_pat = r"final(?:[^\w\r\n]|_)*answer(?:[^\w\r\n]|_)*"

        def _trim_on_markers(value: str) -> str:
            for marker in ("\nQuestion:", "\nContext ", "\nFINAL_ANSWER:", "\nExplanation:", "\nAnswer:"):
                idx = value.find(marker)
                if idx != -1:
                    value = value[:idx]
            return value

        def _cut_when_ngram_repeats(value: str, n: int = 4) -> tuple[str, bool]:
            words = re.findall(r"\S+", value)
            if len(words) < n * 2:
                return value, False
            seen = set()
            cut_at = None
            for i in range(len(words) - n + 1):
                ng = tuple(w.lower() for w in words[i : i + n])
                if ng in seen:
                    cut_at = i
                    break
                seen.add(ng)
            if cut_at is None:
                return value, False
            return " ".join(words[:cut_at]).strip(), True

        def _dedupe_consecutive_sentences(value: str) -> tuple[str, bool]:
            parts = re.split(r"(?<=[.!?])\s+", value)
            out = []
            prev_key = None
            changed = False
            for part in parts:
                s = part.strip()
                if not s:
                    continue
                key = re.sub(r"\W+", " ", s.lower()).strip()
                if key and key == prev_key:
                    changed = True
                    continue
                out.append(s)
                prev_key = key
            return " ".join(out).strip(), changed

        def _clean_tail(value: str) -> str:
            if re.search(rf"(?is)(?:\s*{label_pat}\s*:\s*){{2,}}", value):
                flags["loop_cleaned"] = True
            value = _normalize_mojibake_common(value)
            value = re.sub(rf"(?is)(?:\s*{label_pat}\s*:\s*)+", "", value).strip()
            value = re.sub(rf"(?is)^(?:\s*{label_pat}\s*)+", "", value).strip()
            value = _trim_on_markers(value)
            value, cut_repeated = _cut_when_ngram_repeats(value, n=4)
            if cut_repeated:
                flags["loop_cleaned"] = True
            value, deduped = _dedupe_consecutive_sentences(value)
            if deduped:
                flags["loop_cleaned"] = True
            value = re.sub(r"\s+", " ", value).strip(" -:;")
            # Eval golds are single-sentence extractive spans; keep first sentence only.
            parts = re.split(r"(?<=[.!?])\s+", value)
            first = (parts[0] if parts else "").strip()
            if first:
                value = first.strip(" -:;")
            # Degenerate cleanup fallback.
            if not value:
                flags["not_found_after_cleanup"] = True
                return "NOT_FOUND"
            if re.fullmatch(
                rf"(?is)(?:{label_pat}\s*:|not_found|not\s+found\s+in\s+context|n/?a|none|unknown|\.)+",
                value,
            ):
                flags["not_found_after_cleanup"] = True
                return "NOT_FOUND"
            return value

        raw_for_check = re.sub(rf"(?is)^\s*(?:{label_pat}\s*:\s*)+", "", text).strip()
        raw_for_check = _trim_on_markers(raw_for_check)
        raw_sentences = [s for s in re.split(r"(?<=[.!?])\s+", raw_for_check) if s.strip()]
        flags["multi_sentence_raw"] = len(raw_sentences) >= 2

        m = re.search(rf"(?im)^\s*{label_pat}\s*:\s*(.+)$", text)
        if m:
            return _clean_tail(m.group(1)), flags
        return _clean_tail(text), flags


    predictions = []
    pred_texts = []
    gold_texts = []
    quote_supported = []
    cleanup_multi_sentence_raw = []
    cleanup_loop_cleaned = []
    cleanup_not_found_after_cleanup = []
    latency_values = []
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    pbar = tqdm(rows, desc="eval_baseline", unit="qa", disable=not sys.stderr.isatty())
    for idx, row in enumerate(pbar):
        start = time.perf_counter()
        # Leakage guard: generation input is question + retriever contexts only.
        question = row["question"]
        gold_answer_mode = resolve_answer_mode(question, row.get("answer_mode") or "normal")
        predicted_answer_mode = predict_answer_mode(question)
        if args.answer_mode_source == "router":
            generation_answer_mode = predicted_answer_mode
        elif args.answer_mode_source == "oracle":
            generation_answer_mode = gold_answer_mode
        elif args.answer_mode_source == "none":
            generation_answer_mode = "none"
        else:
            generation_answer_mode = args.answer_mode_source
        predicted_mode_counts[predicted_answer_mode] += 1
        generation_mode_counts[generation_answer_mode] += 1
        router_match_scores.append(1.0 if predicted_answer_mode == gold_answer_mode else 0.0)
        prediction, chunks, scored_ids, initial_scored_ids = pipeline.answer_with_context_stages(
            question,
            answer_mode=generation_answer_mode,
        )
        latency_s = time.perf_counter() - start
        latency_values.append(latency_s)
        prediction_for_metrics, cleanup_flags = extract_final_answer(prediction)
        em_value = exact_match(prediction_for_metrics, row["answer"])
        em_scores.append(em_value)
        f1_precision, f1_recall, f1_value = f1_components(prediction_for_metrics, row["answer"])

        oracle_prediction_for_metrics = None
        oracle_source_chunk = None
        oracle_em = None
        oracle_f1_precision = None
        oracle_f1_recall = None
        oracle_f1 = None
        oracle_contexts, oracle_source_chunk = _oracle_context_from_row(row)
        if oracle_contexts:
            oracle_messages = build_messages(
                pipeline.config.prompt_system,
                question,
                oracle_contexts,
                answer_mode=gold_answer_mode,
            )
            oracle_raw_prediction = pipeline.generator.generate(oracle_messages)
            oracle_prediction_for_metrics, _oracle_cleanup = extract_final_answer(oracle_raw_prediction)
            oracle_em = exact_match(oracle_prediction_for_metrics, row["answer"])
            oracle_f1_precision, oracle_f1_recall, oracle_f1 = f1_components(
                oracle_prediction_for_metrics, row["answer"]
            )
            oracle_em_scores.append(oracle_em)
            oracle_f1_precision_scores.append(oracle_f1_precision)
            oracle_f1_recall_scores.append(oracle_f1_recall)
            oracle_f1_scores.append(oracle_f1)
        f1_precision_scores.append(f1_precision)
        f1_recall_scores.append(f1_recall)
        f1_scores.append(f1_value)
        if gold_answer_mode in mode_metrics:
            mode_metrics[gold_answer_mode]["em"].append(em_value)
            mode_metrics[gold_answer_mode]["f1"].append(f1_value)
            if oracle_em is not None:
                mode_metrics[gold_answer_mode]["oracle_em"].append(oracle_em)
            if oracle_f1 is not None:
                mode_metrics[gold_answer_mode]["oracle_f1"].append(oracle_f1)
        embed_score = embedding_cosine(
            prediction_for_metrics, row["answer"], model_name=args.embed_model
        )
        if embed_score is not None:
            embed_scores.append(embed_score)
        pred_texts.append(prediction_for_metrics)
        gold_texts.append(row["answer"])
        quote_supported.append(
            1.0 if has_context_support(prediction_for_metrics, prediction, chunks) else 0.0
        )
        cleanup_multi_sentence_raw.append(1.0 if cleanup_flags["multi_sentence_raw"] else 0.0)
        cleanup_loop_cleaned.append(1.0 if cleanup_flags["loop_cleaned"] else 0.0)
        cleanup_not_found_after_cleanup.append(
            1.0 if cleanup_flags["not_found_after_cleanup"] else 0.0
        )
        source_chunk = row.get("source_chunk")
        supervision = supervision_by_id.get(str(row.get("id")))
        positive_chunks = row.get("positive_chunks") or (
            supervision.get("positive_chunks") if supervision else []
        )
        support_claims = row.get("support_claims") or []
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
        claim_support_recall_at_k = None
        claim_support_recall_at_retrieve_n = None
        full_claim_support_at_k = None
        full_claim_support_at_retrieve_n = None
        if support_claims:
            claims_supported_at_k = 0
            claims_supported_at_retrieve_n = 0
            for claim in support_claims:
                supported_by = set(claim.get("supported_by") or [])
                if not supported_by:
                    continue
                if any(chunk_id in retrieved_ids for chunk_id in supported_by):
                    claims_supported_at_k += 1
                if any(chunk_id in retrieved_initial_ids for chunk_id in supported_by):
                    claims_supported_at_retrieve_n += 1
            total_claims = len(support_claims)
            claim_support_recall_at_k = claims_supported_at_k / total_claims
            claim_support_recall_at_retrieve_n = claims_supported_at_retrieve_n / total_claims
            full_claim_support_at_k = 1.0 if claims_supported_at_k == total_claims else 0.0
            full_claim_support_at_retrieve_n = (
                1.0 if claims_supported_at_retrieve_n == total_claims else 0.0
            )
            claim_support_recall_scores.append(claim_support_recall_at_k)
            claim_support_recall_retrieve_n_scores.append(claim_support_recall_at_retrieve_n)
            full_claim_support_scores.append(full_claim_support_at_k)
            full_claim_support_retrieve_n_scores.append(full_claim_support_at_retrieve_n)
        if hard_negative_chunks:
            hard_negative_top1_errors.append(
                1.0 if (retrieved_ids and retrieved_ids[0] in set(hard_negative_chunks)) else 0.0
            )

        judge_result_raw = None
        judge_result_checked = None
        if judge is not None and (args.judge_max_samples < 0 or idx < args.judge_max_samples):
            # Retry judge a few times if it returns an unparsable payload.
            judge_result_raw = None
            for _attempt in range(3):
                candidate = judge.judge(row["question"], prediction_for_metrics, chunks)
                corr = candidate.get("correctness")
                grd = candidate.get("groundedness")
                if corr is not None and grd is not None:
                    judge_result_raw = candidate
                    break
                judge_result_raw = candidate
            judge_result_checked = dict(judge_result_raw)
            overlap = overlap_count(row["answer"], chunks)
            raw_evidence = judge_result_checked.get("evidence")
            if isinstance(raw_evidence, list):
                raw_evidence = " ".join(str(item) for item in raw_evidence)
            evidence = (raw_evidence or "").strip()
            evidence_in_context = any(evidence and evidence in ctx for ctx in chunks)
            # Use the same support test as quote_support_rate to avoid over-penalizing
            # groundedness when support is unquoted but still extractive.
            context_supported = has_context_support(prediction_for_metrics, prediction, chunks)

            raw_correctness = judge_result_raw.get("correctness")
            raw_groundedness = judge_result_raw.get("groundedness")
            if raw_correctness is not None:
                judge_raw_correctness.append(raw_correctness)
            if raw_groundedness is not None:
                judge_raw_groundedness.append(raw_groundedness)

            # Soft check 1: only cap extreme highs when there is no direct evidence and no
            # unquoted extractive support in retrieved context.
            if not evidence_in_context and not context_supported:
                if judge_result_checked.get("groundedness") is not None:
                    if judge_result_checked["groundedness"] >= 5:
                        judge_result_checked["groundedness"] = 4
                if judge_result_checked.get("correctness") is not None:
                    if judge_result_checked["correctness"] >= 5:
                        judge_result_checked["correctness"] = 4

            # Soft check 2: if no chunk from labeled source page is retrieved, cap very high
            # groundedness claims but do not force to low floor values.
            if source_page and source_page not in retrieved_pages:
                if judge_result_checked.get("groundedness") is not None:
                    if judge_result_checked["groundedness"] >= 5:
                        judge_result_checked["groundedness"] = 4

            # Soft check 3: if both lexical overlap and support are weak, gently degrade.
            if overlap < 2 and not context_supported:
                if judge_result_checked.get("groundedness") is not None:
                    judge_result_checked["groundedness"] = max(
                        1, int(judge_result_checked["groundedness"]) - 1
                    )
                if judge_result_checked.get("correctness") is not None:
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
                "answer_mode": generation_answer_mode,
                "gold_answer_mode": gold_answer_mode,
                "predicted_answer_mode": predicted_answer_mode,
                "router_match_gold": predicted_answer_mode == gold_answer_mode,
                "context_policy": "retriever_only",
                "gold": row["answer"],
                "prediction": prediction_for_metrics,
                "oracle_prediction": oracle_prediction_for_metrics,
                "oracle_source_chunk": oracle_source_chunk,
                "oracle_context_used": bool(oracle_contexts),
                "oracle_em": oracle_em,
                "oracle_f1": oracle_f1,
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
                "support_claims": support_claims,
                "hard_negative_chunks": hard_negative_chunks,
                "positive_recall_at_retrieve_n": positive_recall_retrieve_n_scores[-1]
                if positive_chunks
                else None,
                "positive_recall_at_k": positive_recall_scores[-1] if positive_chunks else None,
                "claim_support_recall_at_retrieve_n": claim_support_recall_at_retrieve_n,
                "claim_support_recall_at_k": claim_support_recall_at_k,
                "full_claim_support_at_retrieve_n": full_claim_support_at_retrieve_n,
                "full_claim_support_at_k": full_claim_support_at_k,
                "hard_negative_top1_error": hard_negative_top1_errors[-1]
                if hard_negative_chunks
                else None,
                "latency_s": latency_s,
                "cosine_similarity": embed_score,
                "bertscore_f1": None,
                "quote_supported": quote_supported[-1],
                "cleanup_multi_sentence_raw": cleanup_flags["multi_sentence_raw"],
                "cleanup_loop_cleaned": cleanup_flags["loop_cleaned"],
                "cleanup_not_found_after_cleanup": cleanup_flags["not_found_after_cleanup"],
                "judge": judge_result_checked,
                "judge_raw": judge_result_raw,
                "judge_checked": judge_result_checked,
                "lora_adapter": args.lora_adapter,
            }
        )

    latency_mean_s = (sum(latency_values) / len(latency_values)) if latency_values else None
    latency_p95_s = None
    if latency_values:
        sorted_latency = sorted(latency_values)
        p95_idx = max(0, math.ceil(0.95 * len(sorted_latency)) - 1)
        latency_p95_s = sorted_latency[p95_idx]

    peak_gpu_memory_inference_gb = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            peak_gpu_memory_inference_gb = torch.cuda.max_memory_reserved() / (1024**3)
        except Exception:
            peak_gpu_memory_inference_gb = None

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
        "seed": args.seed,
        "answer_mode_source": args.answer_mode_source,
        "router_accuracy": (sum(router_match_scores) / len(router_match_scores)) if router_match_scores else None,
        "router_predicted_exact": predicted_mode_counts["exact"],
        "router_predicted_normal": predicted_mode_counts["normal"],
        "generation_none": generation_mode_counts["none"],
        "generation_exact": generation_mode_counts["exact"],
        "generation_normal": generation_mode_counts["normal"],
        "generation_explicit_grounded": generation_mode_counts["explicit_grounded"],
        "reranker_model": str(args.reranker_model) if args.reranker_model else None,
        "pretrained_reranker_model": args.pretrained_reranker_model,
        "reranker_batch_size": args.reranker_batch_size,
        "retrieve_top_n": args.retrieve_top_n,
        "hybrid_retrieval": args.hybrid_retrieval,
        "vector_top_n": args.vector_top_n,
        "bm25_top_n": args.bm25_top_n,
        "native_sparse_retrieval": args.native_sparse_retrieval,
        "sparse_top_n": args.sparse_top_n,
        "latency_mean_s": latency_mean_s,
        "latency_p95_s": latency_p95_s,
        "peak_gpu_memory_inference_gb": round(peak_gpu_memory_inference_gb, 3)
        if peak_gpu_memory_inference_gb is not None
        else None,
        "retrieval_supervision_file": str(args.retrieval_supervision_file)
        if args.retrieval_supervision_file
        else None,
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "f1_precision": sum(f1_precision_scores) / len(f1_precision_scores),
        "f1_recall": sum(f1_recall_scores) / len(f1_recall_scores),
        "oracle_em": (sum(oracle_em_scores) / len(oracle_em_scores)) if oracle_em_scores else None,
        "oracle_f1": (sum(oracle_f1_scores) / len(oracle_f1_scores)) if oracle_f1_scores else None,
        "oracle_f1_precision": (sum(oracle_f1_precision_scores) / len(oracle_f1_precision_scores))
        if oracle_f1_precision_scores
        else None,
        "oracle_f1_recall": (sum(oracle_f1_recall_scores) / len(oracle_f1_recall_scores))
        if oracle_f1_recall_scores
        else None,
        "oracle_rows": len(oracle_f1_scores),
        "exact_rows": len(mode_metrics["exact"]["f1"]),
        "count_exact": len(mode_metrics["exact"]["f1"]),
        "exact_em": (sum(mode_metrics["exact"]["em"]) / len(mode_metrics["exact"]["em"]))
        if mode_metrics["exact"]["em"]
        else None,
        "exact_f1": (sum(mode_metrics["exact"]["f1"]) / len(mode_metrics["exact"]["f1"]))
        if mode_metrics["exact"]["f1"]
        else None,
        "exact_oracle_rows": len(mode_metrics["exact"]["oracle_f1"]),
        "exact_oracle_em": (
            sum(mode_metrics["exact"]["oracle_em"]) / len(mode_metrics["exact"]["oracle_em"])
        )
        if mode_metrics["exact"]["oracle_em"]
        else None,
        "exact_oracle_f1": (
            sum(mode_metrics["exact"]["oracle_f1"]) / len(mode_metrics["exact"]["oracle_f1"])
        )
        if mode_metrics["exact"]["oracle_f1"]
        else None,
        "normal_rows": len(mode_metrics["normal"]["f1"]),
        "count_normal": len(mode_metrics["normal"]["f1"]),
        "normal_em": (sum(mode_metrics["normal"]["em"]) / len(mode_metrics["normal"]["em"]))
        if mode_metrics["normal"]["em"]
        else None,
        "normal_f1": (sum(mode_metrics["normal"]["f1"]) / len(mode_metrics["normal"]["f1"]))
        if mode_metrics["normal"]["f1"]
        else None,
        "normal_oracle_rows": len(mode_metrics["normal"]["oracle_f1"]),
        "normal_oracle_em": (
            sum(mode_metrics["normal"]["oracle_em"]) / len(mode_metrics["normal"]["oracle_em"])
        )
        if mode_metrics["normal"]["oracle_em"]
        else None,
        "normal_oracle_f1": (
            sum(mode_metrics["normal"]["oracle_f1"]) / len(mode_metrics["normal"]["oracle_f1"])
        )
        if mode_metrics["normal"]["oracle_f1"]
        else None,
        "cosine_similarity": (sum(embed_scores) / len(embed_scores)) if embed_scores else None,
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
        "claim_support_recall_at_retrieve_n": sum(claim_support_recall_retrieve_n_scores)
        / len(claim_support_recall_retrieve_n_scores)
        if claim_support_recall_retrieve_n_scores
        else None,
        "full_claim_support_rate_at_retrieve_n": sum(full_claim_support_retrieve_n_scores)
        / len(full_claim_support_retrieve_n_scores)
        if full_claim_support_retrieve_n_scores
        else None,
        "claim_support_recall_at_k": sum(claim_support_recall_scores)
        / len(claim_support_recall_scores)
        if claim_support_recall_scores
        else None,
        "full_claim_support_rate_at_k": sum(full_claim_support_scores)
        / len(full_claim_support_scores)
        if full_claim_support_scores
        else None,
        "rows_with_support_claims": len(claim_support_recall_scores),
        "hard_negative_top1_error_rate": sum(hard_negative_top1_errors)
        / len(hard_negative_top1_errors)
        if hard_negative_top1_errors
        else None,
        "rows_with_hard_negatives": len(hard_negative_top1_errors),
        "quote_support_rate": sum(quote_supported) / len(quote_supported),
        "multi_sentence_raw_rate": sum(cleanup_multi_sentence_raw) / len(cleanup_multi_sentence_raw),
        "loop_cleaned_rate": sum(cleanup_loop_cleaned) / len(cleanup_loop_cleaned),
        "not_found_after_cleanup_rate": sum(cleanup_not_found_after_cleanup)
        / len(cleanup_not_found_after_cleanup),
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






# Analyzing Quality-Latency-Resource Trade-offs in a Technical Documentation RAG Assistant with LoRA Adaptation

## Abstract

This work studies a retrieval-augmented generation (RAG) assistant for technical documentation under practical constraints. We focus on the interaction between answer quality and efficiency when applying LoRA-based adaptation to a fixed RAG pipeline. The document corpus contains 146 FastAPI documentation pages, split into silver and gold subsets and combined with real-user questions into mixed train/eval/test sets. We compare a strong non-LoRA baseline against LoRA configurations (L4-S, L4-F) under a controlled setup. The best baseline on test reaches `F1=0.3167`, `quote_support_rate=0.875`, and `judge_checked_correctness/groundedness=4.18/3.65`. In our current experiments, LoRA adapters improve some judge-side scores but underperform the baseline on string-overlap and citation support metrics. These results highlight that LoRA in RAG pipelines is not a guaranteed quality improvement and must be selected via explicit multi-metric trade-off analysis.

## 1. Introduction

RAG is a practical strategy for grounding LLM outputs in external documentation. LoRA enables parameter-efficient adaptation but can shift model behavior in non-trivial ways, especially in production-like QA settings where latency and reliability matter.  
This project investigates a concrete question: **which adaptation choices are practically useful when quality, latency, and resource limits are considered jointly?**

## 2. Data and Experimental Setup

### 2.1 Corpus and splits

- Documentation pages: `146`
- Page split: `silver_pages=108`, `gold_pages=38`
- Parsed chunks (current baseline index): `1601`
- Mixed datasets currently used:
  - `qa_train_mixed.jsonl`: `445`
  - `qa_eval_mixed.jsonl`: `60`
  - `qa_test_mixed.jsonl`: `88`

Mixed sets combine documentation-derived QA with real-user questions to better approximate realistic usage.

### 2.2 Baseline pipeline

The baseline keeps retrieval and prompting fixed and varies only model adaptation choices. Evaluation reports:
- lexical/semantic quality (`F1`, `embed_cosine`),
- retrieval grounding (`retrieval_hit_rate(_page)`, `retrieval_mrr(_page)`),
- citation reliability (`quote_support_rate`),
- LLM-judge metrics (raw and post-checked where available).

## 3. LoRA Configuration Space

The study emphasizes LoRA trade-offs through compact configurations, primarily:
- **L4-S** (rank 4, smaller training fraction),
- **L4-F** (rank 4, larger training fraction).

The baseline (no LoRA) is treated as the reference system.

## 4. Results

### 4.1 Best baseline (test)

From `experiments/pilot/results_baseline_test_mixed_fast_fp16_all.json`:

- `F1 = 0.3167`
- `embed_cosine = 0.5285`
- `retrieval_hit_rate_page = 0.8409`
- `retrieval_mrr_page = 0.7405`
- `quote_support_rate = 0.8750`
- `judge_checked_correctness = 4.1837`
- `judge_checked_groundedness = 3.6531`

This is the current strongest test baseline in the available runs.

### 4.2 LoRA test results (current)

From FP16 adapter runs:

- **L4-S (test)**: `F1=0.3067`, `quote_support_rate=0.5000`, `judge_checked=3.94/3.48`
- **L4-F (test)**: `F1=0.2363`, `quote_support_rate=0.4659`, `judge_checked=4.42/3.88`

### 4.3 Observed trade-offs

Relative to the best baseline:
- LoRA configurations can preserve or even improve some judge-side signals,
- but currently lose on key presentation-critical metrics (`F1`, `quote_support_rate`),
- indicating a quality-shift trade-off rather than a strict dominance over baseline.

## 5. Discussion

The current evidence suggests that adding LoRA adapters is sensitive to training setup and objective alignment in RAG.  
In this setting, **baseline quality remains stronger overall** for deployment-oriented criteria, while LoRA exhibits mixed behavior across metric families.

Practically, this means:
- evaluate LoRA with multiple metrics, not one scalar,
- prioritize citation reliability and answer faithfulness for documentation QA,
- use baseline as a robust reference point while LoRA continues as an optimization track.

## 6. Limitations

- Results are based on the current available experiment set (not an exhaustive LoRA sweep).
- Some metric families (string overlap vs judge-based scoring) can disagree and require careful interpretation.
- Additional controlled reruns may still shift conclusions for specific LoRA presets.

## 7. Conclusion

For the current stage, the project demonstrates a reproducible RAG pipeline and a clear empirical framework for LoRA trade-off analysis.  
The strongest verified configuration is still the non-LoRA baseline on test quality and citation support.  
LoRA remains promising but requires further controlled optimization before being presented as a net improvement over baseline.

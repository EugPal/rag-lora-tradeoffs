# Main Dataset Build Protocol (Reproducible)

This document describes the reproducible pipeline for building the main dataset:

- frozen documentation corpus from FastAPI docs
- section-aware page split (`silver_pages` / `gold_pages`)
- silver QA candidates and filtered silver train set with category quotas
- strict gold set and fixed splits
- dataset manifest with counts/hashes

All steps are implemented in repository scripts (no ad-hoc `python -c` required).

## 1) Build URL list

```bash
python -m src.data_pipeline.build_fastapi_url_list --scope all --out-file data/raw/fastapi_urls.txt --stats-out data/processed/url_list_stats.json
```

Output:

- `data/raw/fastapi_urls.txt`
- `data/processed/url_list_stats.json`

## 2) Download and parse docs

```bash
python -m src.data_pipeline.fetch_fastapi_docs --urls-file data/raw/fastapi_urls.txt --max-pages 10000 --sleep 0.05
python -m src.data_pipeline.parse_fastapi_docs --in-dir data/raw/fastapi_html --out-file data/processed/fastapi_pages.jsonl
```

Outputs:

- `data/raw/fastapi_html/*.html`
- `data/processed/fastapi_pages.jsonl`

## 3) Chunk docs and build index

```bash
python -m src.data_pipeline.chunk_docs --in-file data/processed/fastapi_pages.jsonl --out-file data/processed/docs.jsonl --chunk-size 96 --overlap 12
python -m src.rag.rag_pipeline --query "warmup"
```

Outputs:

- `data/processed/docs.jsonl`
- `data/embeddings/docs_embeddings.faiss`
- `data/embeddings/docs_embeddings.npy`

## 4) Build section-aware page split (train/eval isolation)

```bash
python -m src.data_pipeline.build_page_splits --docs-file data/processed/fastapi_pages.jsonl --silver-pages-out data/processed/silver_pages.txt --gold-pages-out data/processed/gold_pages.txt --summary-out data/processed/page_split_summary.json --gold-ratio 0.25 --seed 42
```

Outputs:

- `data/processed/silver_pages.txt` (pages allowed for Silver generation/train)
- `data/processed/gold_pages.txt` (pages reserved for Gold eval)
- `data/processed/page_split_summary.json` (counts by section)

Policy: `silver_pages` and `gold_pages` are page-disjoint by construction.

## 5) Generate and filter silver QA

```bash
python -m src.data_pipeline.build_qa_dataset --docs-file data/processed/docs.jsonl --out-dir data/processed --max-qa 5000 --seed 42 --min-answer-words 8 --max-answer-words 25
python -m src.data_pipeline.filter_qa_candidates --in-file data/processed/qa_small.jsonl --docs-file data/processed/docs.jsonl --silver-pages-file data/processed/silver_pages.txt --out-file data/processed/qa_silver_filtered.jsonl --target-size 350 --seed 42 --max-per-page 4
```

Outputs:

- `data/processed/qa_small.jsonl` (candidate pool)
- `data/processed/qa_silver_filtered.jsonl` (filtered silver set, with `source_page`, `section`, `category`)

Silver policy:

- automated pre-filter (length/noise/dedup/source validity) + manual review
- category quotas for Silver train target (quality-first; quotas are upper targets and may underfill for sparse categories):
  - endpoints/routing 15%
  - pydantic/validation 15%
  - dependencies/DI 15%
  - security/auth 15%
  - async/concurrency 10%
  - testing 10%
  - deployment/ops 8%
  - middleware/lifespan/background 7%
  - errors/debugging 5%

## 6) Build strict gold and fixed splits

```bash
python -m src.data_pipeline.build_gold_dataset --seed-file data/raw/qa_seed/seed.jsonl --candidates-file data/processed/qa_small.jsonl --gold-pages-file data/processed/gold_pages.txt --target-size 150 --max-per-page 2 --out-file data/processed/qa_gold_full.jsonl
python -m src.data_pipeline.build_main_splits --gold-file data/processed/qa_gold_full.jsonl --silver-file data/processed/qa_silver_filtered.jsonl --out-dir data/processed --gold-test-size 120 --gold-val-size 30 --silver-train-size 350 --seed 42 --strict-page-disjoint
```

Outputs:

- `data/processed/qa_gold_full.jsonl`
- `data/processed/qa_gold_val.jsonl`
- `data/processed/qa_gold_test.jsonl`
- `data/processed/qa_silver_train.jsonl`

Notes:

- Gold is built only from `gold_pages` (reserved pages).
- Silver train is sampled from `silver_pages` only and remains category-balanced.
- `--strict-page-disjoint` fails fast if any page overlap appears.
- Quality gate for this checkpoint: `qa_silver_filtered >= 350` and `qa_gold_full >= 150`.
- Recommended after automated build: manual review pass for `qa_silver_filtered.jsonl` and `qa_gold_full.jsonl`.

## 7) Validate source_chunk integrity

```bash
python -m src.evaluation.check_qa_sources --qa-file data/processed/qa_gold_val.jsonl --docs-file data/processed/docs.jsonl
python -m src.evaluation.check_qa_sources --qa-file data/processed/qa_gold_test.jsonl --docs-file data/processed/docs.jsonl
python -m src.evaluation.check_qa_sources --qa-file data/processed/qa_silver_train.jsonl --docs-file data/processed/docs.jsonl
```

## 8) Write dataset manifest

```bash
python -m src.data_pipeline.report_dataset_stats --out-file data/processed/dataset_manifest.json
```

Manifest contains row counts and SHA256 hashes of key artifacts for reproducibility.

## Optional: one-command Makefile target

```bash
make build-main-dataset
make report-dataset
```

## 9) Build real-user QA from FastAPI GitHub Discussions

Source:

- https://github.com/fastapi/fastapi/discussions/categories/questions?discussions_q=category%3AQuestions+is%3Aanswered

```bash
make build-real-user-qa
```

Outputs:

- `data/raw/github/fastapi_questions_answered.jsonl`
- `data/processed/qa_real_user_full.jsonl`
- `data/processed/qa_real_user_train.jsonl`
- `data/processed/qa_real_user_val.jsonl`
- `data/processed/qa_real_user_test.jsonl`

Notes:

- Questions are sourced from official answered FastAPI discussions.
- Full pass uses up to 40 discussions pages.
- Answers are grounded in official docs via retrieval over `docs.jsonl`.

## 10) Merge real-user QA into train/eval/test

```bash
make build-mixed-splits
```

Outputs:

- `data/processed/qa_train_mixed.jsonl` = `qa_silver_train` + `qa_real_user_train`
- `data/processed/qa_eval_mixed.jsonl` = `qa_gold_val` + `qa_real_user_val`
- `data/processed/qa_test_mixed.jsonl` = `qa_gold_test` + `qa_real_user_test`

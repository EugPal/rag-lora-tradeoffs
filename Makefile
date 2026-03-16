.PHONY: build-data build-index build-main-dataset build-real-user-qa build-real-user-qa-llm build-mixed-splits build-retrieval-supervision train-linear-reranker eval-linear-reranker report-dataset eval-baseline eval-lora run-all

build-data:
	python -m src.data_pipeline.fetch_fastapi_docs
	python -m src.data_pipeline.parse_fastapi_docs
	python -m src.data_pipeline.chunk_docs
	python -m src.data_pipeline.build_qa_dataset

build-main-dataset:
	python -m src.data_pipeline.build_fastapi_url_list --scope all --out-file data/raw/fastapi_urls.txt --stats-out data/processed/url_list_stats.json
	python -m src.data_pipeline.fetch_fastapi_docs --urls-file data/raw/fastapi_urls.txt --max-pages 10000 --sleep 0.05
	python -m src.data_pipeline.parse_fastapi_docs --in-dir data/raw/fastapi_html --out-file data/processed/fastapi_pages.jsonl
	python -m src.data_pipeline.chunk_docs --in-file data/processed/fastapi_pages.jsonl --out-file data/processed/docs.jsonl --chunk-size 128 --overlap 16
	python -m src.data_pipeline.build_page_splits --docs-file data/processed/fastapi_pages.jsonl --silver-pages-out data/processed/silver_pages.txt --gold-pages-out data/processed/gold_pages.txt --summary-out data/processed/page_split_summary.json --gold-ratio 0.25 --seed 42
	python -m src.data_pipeline.build_qa_dataset --docs-file data/processed/docs.jsonl --out-dir data/processed --max-qa 5000 --seed 42 --use-llm --llm-model Qwen/Qwen2.5-3B-Instruct --llm-max-tokens 384 --llm-max-attempts 2
	python -m src.data_pipeline.filter_qa_candidates --in-file data/processed/qa_small.jsonl --docs-file data/processed/docs.jsonl --silver-pages-file data/processed/silver_pages.txt --out-file data/processed/qa_silver_filtered.jsonl --target-size 350 --seed 42 --max-per-page 4
	python -m src.data_pipeline.build_gold_dataset --seed-file data/raw/qa_seed/seed.jsonl --candidates-file data/processed/qa_small.jsonl --gold-pages-file data/processed/gold_pages.txt --target-size 150 --max-per-page 2 --out-file data/processed/qa_gold_full.jsonl
	python -m src.data_pipeline.build_main_splits --gold-file data/processed/qa_gold_full.jsonl --silver-file data/processed/qa_silver_filtered.jsonl --out-dir data/processed --gold-test-size 120 --gold-val-size 30 --silver-train-size 350 --seed 42 --strict-page-disjoint

report-dataset:
	python -m src.data_pipeline.report_dataset_stats --out-file data/processed/dataset_manifest.json

build-real-user-qa:
	python -m src.data_pipeline.fetch_fastapi_discussions --max-pages 40 --out-file data/raw/github/fastapi_questions_answered.jsonl
	python -m src.data_pipeline.build_real_user_qa --discussions-file data/raw/github/fastapi_questions_answered.jsonl --docs-file data/processed/docs.jsonl --out-file data/processed/qa_real_user_full.jsonl --max-items 1000
	python -m src.data_pipeline.build_real_user_splits --in-file data/processed/qa_real_user_full.jsonl --out-dir data/processed --train-size 120 --val-size 30 --test-size 30 --seed 42

build-real-user-qa-llm:
	python -m src.data_pipeline.build_real_user_qa --discussions-file data/raw/github/fastapi_questions_answered.jsonl --docs-file data/processed/docs.jsonl --index-file data/embeddings/docs_embeddings.faiss --embeddings-file data/embeddings/docs_embeddings.npy --out-file data/processed/qa_real_user_llm_quote_500.jsonl --max-items 500 --top-k 5 --answer-mode llm_quote --llm-model Qwen/Qwen2.5-3B-Instruct --llm-max-tokens 256 --llm-max-attempts 2 --show-progress

build-mixed-splits:
	python -m src.data_pipeline.build_mixed_splits --silver-train data/processed/qa_silver_train.jsonl --gold-val data/processed/qa_gold_val.jsonl --gold-test data/processed/qa_gold_test.jsonl --real-train data/processed/qa_real_user_train.jsonl --real-val data/processed/qa_real_user_val.jsonl --real-test data/processed/qa_real_user_test.jsonl --out-dir data/processed --seed 42

build-retrieval-supervision:
	python -m src.data_pipeline.build_retrieval_supervision --qa-file data/processed/qa_eval_main.jsonl --docs-file data/processed/docs.jsonl --index-file data/embeddings/docs_embeddings.faiss --out-file data/processed/retrieval_supervision_eval.jsonl --stats-file data/processed/retrieval_supervision_eval_stats.json --top-n 50 --hard-negatives 3 --in-page-negatives 1 --random-negatives 1 --add-neighbor-positive --max-negatives-per-positive 5 --seed 42

train-linear-reranker:
	python -m src.retrieval.train_linear_reranker --supervision-file data/processed/retrieval_supervision_eval.jsonl --docs-file data/processed/docs.jsonl --index-file data/embeddings/docs_embeddings.faiss --embeddings-file data/embeddings/docs_embeddings.npy --out-file experiments/retrieval/linear_reranker_v1.json

eval-linear-reranker:
	python -m src.retrieval.eval_linear_reranker --supervision-file data/processed/retrieval_supervision_eval.jsonl --model-file experiments/retrieval/linear_reranker_v1.json --docs-file data/processed/docs.jsonl --index-file data/embeddings/docs_embeddings.faiss --embeddings-file data/embeddings/docs_embeddings.npy --out-file experiments/retrieval/linear_reranker_eval_v1.json

build-index:
	python -m src.rag.rag_pipeline --query "warmup"

eval-baseline:
	python -m src.evaluation.eval_baseline

eval-lora:
	python -m src.evaluation.eval_lora --preset L8-F

run-all: build-data build-index eval-baseline eval-lora

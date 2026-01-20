.PHONY: build-data build-index eval-baseline eval-lora run-all

build-data:
	python -m src.data_pipeline.fetch_fastapi_docs
	python -m src.data_pipeline.parse_fastapi_docs
	python -m src.data_pipeline.chunk_docs
	python -m src.data_pipeline.build_qa_dataset

build-index:
	python -m src.rag.rag_pipeline --query "warmup"

eval-baseline:
	python -m src.evaluation.eval_baseline

eval-lora:
	python -m src.evaluation.eval_lora --preset L8-F

run-all: build-data build-index eval-baseline eval-lora

$ErrorActionPreference = "Stop"

Write-Host "Step 1/6: Fetch FastAPI docs"
python -m src.data_pipeline.fetch_fastapi_docs

Write-Host "Step 2/6: Parse FastAPI docs"
python -m src.data_pipeline.parse_fastapi_docs

Write-Host "Step 3/6: Build chunks"
python -m src.data_pipeline.chunk_docs

Write-Host "Step 4/6: Build QA splits"
python -m src.data_pipeline.build_qa_dataset

Write-Host "Step 5/6: Build index (warmup query)"
python -m src.rag.rag_pipeline --query "warmup"

Write-Host "Step 6/6: Evaluate baseline"
python -m src.evaluation.eval_baseline

Write-Host "Done."

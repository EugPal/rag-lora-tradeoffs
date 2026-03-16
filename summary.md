# Project Summary

## Official Evaluation Metrics

We use exactly six core metrics for reporting:

- EM: exact match between normalized predicted answer and gold answer.
- F1: token overlap F1 between predicted answer and gold answer.
- Accuracy: judge_checked_correctness_avg (LLM-judge correctness after post-check).
- Groundedness: judge_checked_groundedness_avg (LLM-judge groundedness after post-check).
- Chunk Recall: 
etrieval_hit_rate (whether the gold source_chunk is retrieved in top-k).
- Embedding Similarity: embed_cosine between predicted answer and gold answer.

## Notes

- Retrieval labels (source_chunk) are used for metrics only.
- Generation context in baseline RAG eval comes from retriever output only.

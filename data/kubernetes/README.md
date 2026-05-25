# Kubernetes-docs RAG benchmark (local backup)

Small curated mirror of the data used in the paper. The canonical, easier
to use version lives on HuggingFace:
[`evgenypal/k8s-docs-rag-bench`](https://huggingface.co/datasets/evgenypal/k8s-docs-rag-bench).

This local copy exists so the repo is self-contained and can be cloned
without depending on HF availability.

## Layout

```
data/kubernetes/
├── corpus/
│   ├── pages.jsonl       # cleaned full pages from the Kubernetes docs
│   └── chunks.jsonl      # the same pages segmented into semantic chunks;
│                         # the RAG indexer consumes this file
└── qa/
    ├── split_summary.json   # page-level 60/20/20 split sizes by section
    ├── stats.json           # per-split QA composition (page kinds, sections)
    ├── train/  qa.jsonl + page_ids.txt
    ├── eval/   qa.jsonl + page_ids.txt
    └── test/   qa.jsonl + page_ids.txt
```

## Provenance

- **Source:** [kubernetes.io/docs/](https://kubernetes.io/docs/), snapshot
  taken on 2026-02-04.
- **Split protocol:** documents are randomly partitioned at the page level
  (60% / 20% / 20%) with a fixed seed; QA pairs inherit the split of their
  source page, so train / eval / test never share pages.
- **QA construction:** seeded with LLM-drafted question–answer pairs, then
  manually reviewed and cleaned. Only the final (v2) version is shipped
  here; intermediate audit traces have been dropped to keep the repo small.

See the paper and the HuggingFace dataset card for the full methodology.

## License

CC BY 4.0, in line with the upstream Kubernetes documentation license.

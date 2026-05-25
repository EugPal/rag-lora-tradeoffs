# RAG + LoRA Trade-offs on Kubernetes Docs

Code and benchmark backups for the study:

> **"Quality–Latency–Resource Trade-offs in a Technical-Documentation RAG
> Assistant with LoRA Adaptation"**
> Evgenii Palnikov, Elizaveta Gavrilova. HSE University, 2026.

The work studies how different LoRA configurations (rank, target modules,
training data size) interact with a retrieval-augmented generation pipeline
over the Kubernetes documentation corpus, and where they sit on the
Quality–Latency–Resource Pareto frontier.

The arXiv preprint will be added to this repository in a follow-up commit
once it is assigned an arXiv ID.

## Artifacts on HuggingFace

- Benchmark (corpus + QA + judge labels):
  [`evgenypal/k8s-docs-rag-bench`](https://huggingface.co/datasets/evgenypal/k8s-docs-rag-bench)
- LoRA adapters:
  - [`evgenypal/llama-3.2-3b-k8s-rag-lora-r64-qv`](https://huggingface.co/evgenypal/llama-3.2-3b-k8s-rag-lora-r64-qv)
  - [`evgenypal/llama-3.1-8b-k8s-rag-lora-r64-qv`](https://huggingface.co/evgenypal/llama-3.1-8b-k8s-rag-lora-r64-qv)

## What's in the repo

```
.
├── src/                                  # pipeline source code
│   ├── data_pipeline/                    # fetch / parse / chunk Kubernetes docs
│   ├── rag/                              # FAISS dense index, RAG generator
│   ├── retrieval/                        # BGE-M3 sparse + RRF + cross-encoder rerank
│   ├── lora/                             # LoRA training & inference
│   ├── evaluation/                       # F1, groundedness, offline LLM-judge
│   └── utils/
├── data/kubernetes/                      # 21 MB curated backup of the
│   │                                     # benchmark (also on HuggingFace)
│   ├── corpus/                           #   pages.jsonl + chunks.jsonl
│   └── qa/                               #   train/ eval/ test/ {qa.jsonl, page_ids.txt}
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

Large artifacts (raw HTML scrape, training-time checkpoints, full embedding
matrices) live outside git. The full benchmark and the selected adapters are
mirrored on HuggingFace (see links above).

## Quick start

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### Load the benchmark from HuggingFace

```python
from datasets import load_dataset

corpus = load_dataset("evgenypal/k8s-docs-rag-bench", "corpus", split="train")
qa     = load_dataset("evgenypal/k8s-docs-rag-bench", "qa")
labels = load_dataset("evgenypal/k8s-docs-rag-bench", "judge_labels", split="train")
```

### Run a LoRA adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id    = "meta-llama/Llama-3.2-3B-Instruct"
adapter_id = "evgenypal/llama-3.2-3b-k8s-rag-lora-r64-qv"

tok   = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

See the model cards on HuggingFace for a full RAG-style inference example.

## Reproducing the pipeline (high level)

1. **Fetch + parse Kubernetes docs** — `src/data_pipeline/fetch_kubernetes_docs.py`,
   `src/data_pipeline/parse_kubernetes_docs.py` (snapshot date is recorded in
   the dataset card on HuggingFace).
2. **Chunk** — `src/data_pipeline/build_semantic_kubernetes_corpus.py` produces
   `data/kubernetes/corpus/chunks.jsonl`.
3. **Build QA splits** — `src/data_pipeline/build_kubernetes_page_split_60_20_20.py`
   and `build_kubernetes_qa_dataset.py`. The curated splits are checked in
   under `data/kubernetes/qa/{train,eval,test}/` and are also published in
   the HuggingFace dataset.
4. **Index** — dense FAISS index via `src/rag/embeddings.py` + `src/rag/index.py`;
   BGE-M3 native sparse via `src/retrieval/bge_m3_sparse_retriever.py`. The
   main pipeline fuses dense and BGE-M3 sparse with Reciprocal Rank Fusion,
   then applies a fine-tuned cross-encoder reranker.
5. **Train LoRA** — `src/lora/train_lora.py` (configs in `src/config/`).
6. **Evaluate** — `src/evaluation/eval_lora.py` produces predictions; offline
   LLM-judge in `src/evaluation/judge_api_offline.py` scores correctness and
   groundedness.

## Citing

If this code or data is useful, please cite the paper (BibTeX coming after
arXiv assignment) and the HuggingFace artifacts.

## License

Code in this repository: MIT — see [`LICENSE`](LICENSE).
The published benchmark (`k8s-docs-rag-bench`) and the LoRA adapters are
released under **CC BY 4.0**; see the corresponding HuggingFace repos for the
full license texts.

The Kubernetes documentation, from which the corpus is derived, is © the
Kubernetes Authors and is published under CC BY 4.0
(<https://github.com/kubernetes/website/blob/main/LICENSE>).

from __future__ import annotations

import argparse
import json
import math
import random
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from src.lora.lora_config import get_preset
from src.rag.generator import BaseGenerator
from src.rag.rag_pipeline import RagConfig, RagPipeline, build_prompt
from src.utils.io_utils import read_jsonl, write_yaml
from src.utils.logging_utils import setup_logging


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "1850e6",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with Path("debug-1850e6.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


def _stdout_debug(logger, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    try:
        logger.info(
            "[DEBUG1850e6][%s][%s] %s | %s",
            hypothesis_id,
            location,
            message,
            json.dumps(data, ensure_ascii=False, sort_keys=True),
        )
    except Exception:
        pass
    # endregion


def _target_module_suffixes(target_modules: list[str]) -> tuple[list[str], list[str]]:
    # Llama-style module names used by HF models.
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    attn_qv = ["q_proj", "v_proj"]
    attn_qk = ["q_proj", "k_proj"]
    ffn = ["gate_proj", "up_proj", "down_proj"]

    attn_out: list[str] = []
    if "attention" in target_modules:
        attn_out.extend(attn)
    if "attention_qv" in target_modules:
        attn_out.extend(attn_qv)
    if "attention_qk" in target_modules:
        attn_out.extend(attn_qk)
    # Preserve order while deduplicating in case both flags are present.
    attn_out = list(dict.fromkeys(attn_out))

    ffn_out: list[str] = ffn if "ffn" in target_modules else []
    return attn_out, ffn_out


def _top_layer_indices(model) -> list[int] | None:
    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        return None
    # "top" layers: adapt the upper 50% transformer blocks.
    top_n = max(1, num_layers // 2)
    start = max(0, num_layers - top_n)
    return list(range(start, num_layers))


def _build_target_modules(model, target_layers: str, target_modules: list[str]) -> list[str]:
    """
    PEFT normally matches by suffix (e.g. "q_proj"). To restrict to top layers
    robustly across PEFT versions, we pass fully-qualified module names for those layers.
    """
    attn_suffixes, ffn_suffixes = _target_module_suffixes(target_modules)
    if target_layers == "all":
        return [*attn_suffixes, *ffn_suffixes]

    layer_idxs = _top_layer_indices(model) or []
    full: list[str] = []
    for i in layer_idxs:
        for s in attn_suffixes:
            full.append(f"model.layers.{i}.self_attn.{s}")
        for s in ffn_suffixes:
            full.append(f"model.layers.{i}.mlp.{s}")
    return full


def _format_supervised_target(answer: str) -> str:
    answer = (answer or "").strip()
    # Keep training target aligned with inference format from rag_pipeline.build_prompt().
    if not answer:
        return "NOT_FOUND"
    return answer


class PromptQADataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        pipeline: RagPipeline,
        system_prompt: str,
        top_k: int,
        max_length: int,
    ) -> None:
        self.items = []
        self._retrieval_cache: dict[str, tuple[list[str], list[str]]] = {}
        for row in rows:
            q = row["question"]
            a = row["answer"]
            answer_mode = row.get("answer_mode") or "normal"
            contexts, target_answer = self._contexts_and_target_for_row(
                row=row,
                question=q,
                answer=a,
                pipeline=pipeline,
                top_k=top_k,
            )

            # Ensure the answer is not truncated away: fit prompt into a token budget.
            answer_ids = tokenizer(target_answer, add_special_tokens=False).get("input_ids", [])
            reserve_for_answer = min(256, max(64, len(answer_ids) + 8))
            prompt_budget = max(128, max_length - reserve_for_answer)

            contexts_used: list[str] = []
            for ctx in contexts[:top_k]:
                # Cheap per-context truncation to reduce prompt bloat.
                ctx = ctx[:2000]
                candidate = contexts_used + [ctx]
                candidate_prompt = build_prompt(
                    system_prompt,
                    q,
                    candidate,
                    answer_mode=answer_mode,
                )
                prompt_len = len(
                    tokenizer(candidate_prompt, add_special_tokens=False).get("input_ids", [])
                )
                if prompt_len > prompt_budget:
                    break
                contexts_used = candidate
            prompt = build_prompt(
                system_prompt,
                q,
                contexts_used,
                answer_mode=answer_mode,
            )
            target = _format_supervised_target(target_answer)
            full_text = prompt + "\n" + target

            enc_full = tokenizer(
                full_text,
                return_attention_mask=True,
            )
            enc_prompt = tokenizer(
                prompt,
                return_attention_mask=False,
            )
            labels = enc_full["input_ids"][:]
            prompt_len = len(enc_prompt["input_ids"])
            labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
            # Skip examples where the answer got fully truncated/masked.
            if all(x == -100 for x in labels):
                continue

            self.items.append(
                {
                    "input_ids": enc_full["input_ids"],
                    "attention_mask": enc_full["attention_mask"],
                    "labels": labels,
                }
            )

    def _lookup_chunk_texts(self, pipeline: RagPipeline, chunk_ids: list[str]) -> list[str]:
        contexts: list[str] = []
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                continue
            text = pipeline._doc_lookup.get(chunk_id.strip(), "")
            if text:
                contexts.append(text)
        return contexts

    def _retrieved_contexts(
        self, pipeline: RagPipeline, question: str, top_k: int
    ) -> tuple[list[str], list[str]]:
        cached = self._retrieval_cache.get(question)
        if cached is not None:
            return cached
        contexts, results = pipeline.retrieve(question)
        retrieved_ids = [doc_id for doc_id, _score in results][:top_k]
        retrieved_contexts = contexts[:top_k]
        self._retrieval_cache[question] = (retrieved_contexts, retrieved_ids)
        return retrieved_contexts, retrieved_ids

    def _singlehop_contexts_and_target(
        self,
        question: str,
        answer: str,
        single_source_chunk: str,
        pipeline: RagPipeline,
        top_k: int,
    ) -> tuple[list[str], str]:
        gold_text = pipeline._doc_lookup.get(single_source_chunk, "")
        retrieved_contexts, retrieved_ids = self._retrieved_contexts(pipeline, question, top_k)
        if not gold_text:
            return (retrieved_contexts or [], answer)

        # Use a milder single-hop mixture: mostly clean gold context, sometimes gold with distractors.
        mode_roll = random.random()
        if mode_roll < 0.60:
            return [gold_text], answer

        distractors = [
            ctx
            for ctx, chunk_id in zip(retrieved_contexts, retrieved_ids)
            if chunk_id != single_source_chunk
        ][:2]
        contexts = [gold_text, *distractors]
        random.shuffle(contexts)
        return contexts[:top_k], answer

    def _contexts_and_target_for_row(
        self,
        row: dict,
        question: str,
        answer: str,
        pipeline: RagPipeline,
        top_k: int,
    ) -> tuple[list[str], str]:
        source_chunk_ids = row.get("source_chunks")
        if isinstance(source_chunk_ids, list) and source_chunk_ids:
            contexts = self._lookup_chunk_texts(pipeline, source_chunk_ids)
            if contexts:
                return contexts, answer
        elif isinstance(source_chunk_ids, str) and source_chunk_ids.strip():
            contexts = self._lookup_chunk_texts(pipeline, [source_chunk_ids])
            if contexts:
                return contexts, answer

        single_source_chunk = row.get("source_chunk")
        if isinstance(single_source_chunk, str) and single_source_chunk.strip():
            return self._singlehop_contexts_and_target(
                question=question,
                answer=answer,
                single_source_chunk=single_source_chunk.strip(),
                pipeline=pipeline,
                top_k=top_k,
            )

        retrieved_contexts, _retrieved_ids = self._retrieved_contexts(pipeline, question, top_k)
        return retrieved_contexts, answer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

def _cuda_driver_api_version() -> int | None:
    try:
        getter = getattr(torch._C, "_cuda_getDriverVersion", None)
        if getter is None:
            return None
        value = getter()
        return int(value) if value is not None else None
    except Exception:
        return None


def _fail_fast_if_cuda_unavailable() -> None:
    if torch.cuda.is_available():
        return
    details = {
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_built": bool(torch.backends.cuda.is_built()),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_driver_api_version": _cuda_driver_api_version(),
    }
    raise RuntimeError(
        "CUDA is not available on this training node. "
        "Failing fast to avoid CPU-only LoRA training. "
        f"Diagnostics: {details}"
    )


def _collate_pad(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


class _CheckpointTimingCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._train_start_perf: float | None = None
        self.checkpoints: list[dict] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_perf = time.perf_counter()
        return control

    def on_save(self, args, state, control, **kwargs):
        if self._train_start_perf is None:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        elapsed_s = time.perf_counter() - self._train_start_perf
        row = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "elapsed_train_wall_clock_s": elapsed_s,
            "checkpoint_dir": str(checkpoint_dir),
            "exists": checkpoint_dir.exists(),
        }
        self.checkpoints.append(row)
        try:
            (self.output_dir / "checkpoint_timing.json").write_text(
                json.dumps({"checkpoints": self.checkpoints}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return control



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for the RAG generator (PEFT).")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/qa_train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/lora"))
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--embed-top-k", type=int, default=8, help="top-k contexts to include in training prompt")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="no",
        choices=["no", "epoch", "steps"],
        help="Trainer checkpoint save strategy.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Checkpoint save frequency in steps when --save-strategy=steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="Max number of checkpoints to keep (0 means unlimited).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to a trainer checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization for base model loading.",
    )
    args = parser.parse_args()
    run_id = f"train-{int(time.time())}"

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    logger = setup_logging("train_lora")
    total_wall_clock_start = time.perf_counter()
    _fail_fast_if_cuda_unavailable()
    _stdout_debug(
        logger,
        "H6",
        "src/lora/train_lora.py:main:startup",
        "Process start paths",
        {
            "cwd": os.getcwd(),
            "debug_log_abs": str(Path("debug-1850e6.log").resolve()),
            "train_file_abs": str(args.train_file.resolve()),
        },
    )
    _debug_log(
        run_id,
        "H1,H4",
        "src/lora/train_lora.py:main:args",
        "Parsed train arguments",
        {
            "preset": args.preset,
            "embed_top_k": args.embed_top_k,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "gradient_checkpointing": args.gradient_checkpointing,
            "no_quantization": args.no_quantization,
        },
    )
    config = get_preset(args.preset)
    train_rows = read_jsonl(args.train_file)
    # Keep all questions for every preset; S/F differ by layer coverage, not data fraction.
    used_rows = len(train_rows)

    run_dir = args.out_dir / args.preset
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config_used.yaml", asdict(config))

    if not train_rows:
        logger.warning("No training data found at %s", args.train_file)
        return

    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Base model loading mode:
    # - default: 4-bit quantized (QLoRA-style)
    # - --no-quantization: full-precision base on GPU (bf16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_4bit = not args.no_quantization
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="cpu",
            )
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # With gradient checkpointing, force input embeddings to require grad;
    # otherwise backward can fail with no grad_fn in full-precision LoRA mode.
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = _build_target_modules(model, config.target_layers, config.target_modules)
    peft_cfg = PeftLoraConfig(
        r=config.rank,
        lora_alpha=max(8, config.rank * 2),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    _debug_log(
        run_id,
        "H3,H5",
        "src/lora/train_lora.py:main:model_init",
        "Model initialized and PEFT attached",
        {
            "use_4bit": use_4bit,
            "cuda_available": torch.cuda.is_available(),
            "is_gradient_checkpointing_model": bool(getattr(model, "is_gradient_checkpointing", False)),
            "mem_alloc_gb": (
                round(torch.cuda.memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else None
            ),
            "mem_reserved_gb": (
                round(torch.cuda.memory_reserved() / (1024**3), 3) if torch.cuda.is_available() else None
            ),
            "mem_max_alloc_gb": (
                round(torch.cuda.max_memory_allocated() / (1024**3), 3)
                if torch.cuda.is_available()
                else None
            ),
        },
    )

    # Build prompts using current RAG retriever/index.
    rag = RagPipeline(RagConfig(top_k=args.embed_top_k), generator=BaseGenerator())
    system_prompt = rag.config.prompt_system

    train_ds = PromptQADataset(
        train_rows,
        tokenizer=tokenizer,
        pipeline=rag,
        system_prompt=system_prompt,
        top_k=args.embed_top_k,
        max_length=args.max_length,
    )
    _debug_log(
        run_id,
        "H4",
        "src/lora/train_lora.py:main:dataset_stats",
        "Built train dataset",
        {
            "train_ds_len": len(train_ds),
            "sample_input_len": (len(train_ds[0]["input_ids"]) if len(train_ds) > 0 else 0),
        },
    )

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer_out"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        save_total_limit=(args.save_total_limit if args.save_total_limit > 0 else None),
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    _debug_log(
        run_id,
        "H5",
        "src/lora/train_lora.py:main:training_args",
        "TrainingArguments configured",
        {
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "bf16": training_args.bf16,
            "fp16": training_args.fp16,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=_collate_pad(tokenizer),
    )
    checkpoint_timing_cb = _CheckpointTimingCallback(run_dir)
    trainer.add_callback(checkpoint_timing_cb)
    resume_checkpoint = args.resume_from_checkpoint.strip() or None
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    train_wall_clock_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_wall_clock_s = time.perf_counter() - train_wall_clock_start
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    peak_gpu_memory_training_gb = (
        torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else None
    )
    _debug_log(
        run_id,
        "H3",
        "src/lora/train_lora.py:main:post_train_mem",
        "Post-train GPU memory snapshot",
        {
            "mem_alloc_gb": (
                round(torch.cuda.memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else None
            ),
            "mem_reserved_gb": (
                round(torch.cuda.memory_reserved() / (1024**3), 3) if torch.cuda.is_available() else None
            ),
            "mem_max_alloc_gb": (
                round(torch.cuda.max_memory_allocated() / (1024**3), 3)
                if torch.cuda.is_available()
                else None
            ),
        },
    )

    # Save adapter + tokenizer
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    total_wall_clock_s = time.perf_counter() - total_wall_clock_start

    metrics = {
        "preset": args.preset,
        "status": "training_completed",
        "model_name": args.model_name,
        "use_4bit": use_4bit,
        "train_rows": len(train_ds),
        "val_rows": 0,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
        "training_wall_clock_s": train_wall_clock_s,
        "training_wall_clock_minutes": (train_wall_clock_s / 60.0),
        "total_wall_clock_s": total_wall_clock_s,
        "peak_gpu_memory_training_gb": round(peak_gpu_memory_training_gb, 3)
        if peak_gpu_memory_training_gb is not None
        else None,
        "checkpoint_timing_file": str(run_dir / "checkpoint_timing.json"),
        "checkpoints_recorded": len(checkpoint_timing_cb.checkpoints),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text("Training completed.\n", encoding="utf-8")
    logger.info("Saved LoRA adapter to %s", adapter_dir)
    _stdout_debug(
        logger,
        "H6,H9",
        "src/lora/train_lora.py:main:artifact_paths",
        "Artifact path checks after save",
        {
            "adapter_dir_abs": str(adapter_dir.resolve()),
            "adapter_dir_exists": adapter_dir.exists(),
            "adapter_config_abs": str((adapter_dir / "adapter_config.json").resolve()),
            "adapter_config_exists": (adapter_dir / "adapter_config.json").exists(),
            "metrics_abs": str((run_dir / "metrics.json").resolve()),
            "metrics_exists": (run_dir / "metrics.json").exists(),
            "debug_log_abs": str(Path("debug-1850e6.log").resolve()),
            "debug_log_exists": Path("debug-1850e6.log").exists(),
            "debug_log_size_bytes": (
                Path("debug-1850e6.log").stat().st_size if Path("debug-1850e6.log").exists() else None
            ),
        },
    )
    _debug_log(
        run_id,
        "H1,H2",
        "src/lora/train_lora.py:main:artifacts",
        "Artifact existence and sizes",
        {
            "run_dir": str(run_dir),
            "adapter_dir_exists": adapter_dir.exists(),
            "adapter_config_exists": (adapter_dir / "adapter_config.json").exists(),
            "adapter_model_exists": (adapter_dir / "adapter_model.safetensors").exists(),
            "adapter_model_size_mb": (
                round((adapter_dir / "adapter_model.safetensors").stat().st_size / (1024 * 1024), 3)
                if (adapter_dir / "adapter_model.safetensors").exists()
                else None
            ),
            "tokenizer_json_exists": (adapter_dir / "tokenizer.json").exists(),
            "tokenizer_json_size_mb": (
                round((adapter_dir / "tokenizer.json").stat().st_size / (1024 * 1024), 3)
                if (adapter_dir / "tokenizer.json").exists()
                else None
            ),
            "metrics_exists": (run_dir / "metrics.json").exists(),
            "metrics_size_bytes": (
                (run_dir / "metrics.json").stat().st_size if (run_dir / "metrics.json").exists() else None
            ),
            "cwd": os.getcwd(),
        },
    )


if __name__ == "__main__":
    main()









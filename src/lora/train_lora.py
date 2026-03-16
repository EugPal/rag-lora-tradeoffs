from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
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
    ffn = ["gate_proj", "up_proj", "down_proj"]
    attn_out: list[str] = attn if "attention" in target_modules else []
    ffn_out: list[str] = ffn if "ffn" in target_modules else []
    return attn_out, ffn_out


def _top_layer_indices(model) -> list[int] | None:
    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        return None
    # "top" layers: adapt the last N transformer blocks.
    top_n = min(12, num_layers)
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
    # Use the same extractive text as both evidence quote and final answer.
    quoted = answer.replace('"', '\\"')
    return f'Quotes:\n- "{quoted}"\nAnswer:\n{answer}'


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
        for row in rows:
            q = row["question"]
            a = row["answer"]
            # Retrieve contexts to mimic RAG-time prompting.
            contexts, _results = pipeline.retrieve(q)

            # Ensure the answer is not truncated away: fit prompt into a token budget.
            answer_ids = tokenizer(a, add_special_tokens=False).get("input_ids", [])
            reserve_for_answer = min(256, max(64, len(answer_ids) + 8))
            prompt_budget = max(128, max_length - reserve_for_answer)

            contexts_used: list[str] = []
            for ctx in contexts[:top_k]:
                # Cheap per-context truncation to reduce prompt bloat.
                ctx = ctx[:2000]
                candidate = contexts_used + [ctx]
                candidate_prompt = build_prompt(system_prompt, q, candidate)
                prompt_len = len(
                    tokenizer(candidate_prompt, add_special_tokens=False).get("input_ids", [])
                )
                if prompt_len > prompt_budget:
                    break
                contexts_used = candidate
            prompt = build_prompt(system_prompt, q, contexts_used)
            target = _format_supervised_target(a)
            full_text = prompt + "\n" + target

            enc_full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            enc_prompt = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


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

    logger = setup_logging("train_lora")
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
    used_rows = max(1, int(len(train_rows) * config.data_fraction)) if train_rows else 0
    train_rows = train_rows[:used_rows]

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
        save_strategy="no",
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
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
    train_result = trainer.train()
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

    metrics = {
        "preset": args.preset,
        "status": "training_completed",
        "model_name": args.model_name,
        "use_4bit": use_4bit,
        "train_rows": len(train_ds),
        "val_rows": 0,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
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

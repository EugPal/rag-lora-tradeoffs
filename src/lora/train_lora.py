from __future__ import annotations

import argparse
import json
import math
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
            full_text = prompt + "\n" + a

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
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/qa_val.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/lora"))
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--embed-top-k", type=int, default=8, help="top-k contexts to include in training prompt")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    logger = setup_logging("train_lora")
    config = get_preset(args.preset)
    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
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

    # Base model in 4-bit for efficient LoRA training.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    model.config.use_cache = False

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
    eval_ds = (
        PromptQADataset(
            val_rows,
            tokenizer=tokenizer,
            pipeline=rag,
            system_prompt=system_prompt,
            top_k=args.embed_top_k,
            max_length=args.max_length,
        )
        if val_rows
        else None
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
        eval_strategy="no" if eval_ds is None else "epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate_pad(tokenizer),
    )
    train_result = trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "preset": args.preset,
        "status": "training_completed",
        "model_name": args.model_name,
        "train_rows": len(train_ds),
        "val_rows": len(eval_ds) if eval_ds is not None else 0,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text("Training completed.\n", encoding="utf-8")
    logger.info("Saved LoRA adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()

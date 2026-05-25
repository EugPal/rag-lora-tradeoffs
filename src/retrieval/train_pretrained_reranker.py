from __future__ import annotations

import argparse
import json
import random
import inspect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.io_utils import read_jsonl


def _score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[1] > 1:
        return logits[:, -1]
    return logits.reshape(-1)


@dataclass
class PairwiseExample:
    row_id: str
    group_id: str
    query: str
    positive_text: str
    negative_text: str
    source_positive: str
    source_negative: str
    example_type: str
    review_status: str


class PairwiseRerankerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: list[PairwiseExample],
        tokenizer,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max(32, int(max_length))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.examples[idx]
        pos = self.tokenizer(
            row.query,
            row.positive_text,
            truncation=True,
            max_length=self.max_length,
        )
        neg = self.tokenizer(
            row.query,
            row.negative_text,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            'pos_input_ids': pos['input_ids'],
            'pos_attention_mask': pos['attention_mask'],
            'neg_input_ids': neg['input_ids'],
            'neg_attention_mask': neg['attention_mask'],
        }


class PairwiseCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        pos_features = [
            {
                'input_ids': feature['pos_input_ids'],
                'attention_mask': feature['pos_attention_mask'],
            }
            for feature in features
        ]
        neg_features = [
            {
                'input_ids': feature['neg_input_ids'],
                'attention_mask': feature['neg_attention_mask'],
            }
            for feature in features
        ]
        pos_batch = self.tokenizer.pad(pos_features, padding=True, return_tensors='pt')
        neg_batch = self.tokenizer.pad(neg_features, padding=True, return_tensors='pt')
        return {
            'pos_input_ids': pos_batch['input_ids'],
            'pos_attention_mask': pos_batch['attention_mask'],
            'neg_input_ids': neg_batch['input_ids'],
            'neg_attention_mask': neg_batch['attention_mask'],
        }


class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos_inputs = {
            'input_ids': inputs['pos_input_ids'],
            'attention_mask': inputs['pos_attention_mask'],
        }
        neg_inputs = {
            'input_ids': inputs['neg_input_ids'],
            'attention_mask': inputs['neg_attention_mask'],
        }
        pos_scores = _score_from_logits(model(**pos_inputs).logits)
        neg_scores = _score_from_logits(model(**neg_inputs).logits)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if return_outputs:
            return loss, {'pos_scores': pos_scores, 'neg_scores': neg_scores}
        return loss


def _load_docs_lookup(docs_file: Path) -> dict[str, str]:
    docs = read_jsonl(docs_file)
    return {row['id']: row.get('text', '') for row in docs if row.get('id')}


def _build_examples(pairwise_file: Path, docs_lookup: dict[str, str]) -> tuple[list[PairwiseExample], int]:
    rows = read_jsonl(pairwise_file)
    examples: list[PairwiseExample] = []
    skipped = 0
    for row in rows:
        query = str(row.get('query') or '').strip()
        positive_id = str(row.get('positive') or '').strip()
        negative_id = str(row.get('negative') or '').strip()
        positive_text = docs_lookup.get(positive_id, '').strip()
        negative_text = docs_lookup.get(negative_id, '').strip()
        if not query or not positive_text or not negative_text:
            skipped += 1
            continue
        examples.append(
            PairwiseExample(
                row_id=str(row.get('id') or ''),
                group_id=str(row.get('source_supervision_id') or row.get('id') or ''),
                query=query,
                positive_text=positive_text,
                negative_text=negative_text,
                source_positive=positive_id,
                source_negative=negative_id,
                example_type=str(row.get('type') or ''),
                review_status=str(row.get('review_status') or ''),
            )
        )
    return examples, skipped


def _split_group_ids(
    group_ids: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if not group_ids:
        return set(), set()
    if len(group_ids) == 1 or val_fraction <= 0.0:
        return set(group_ids), set()
    rng = random.Random(seed)
    ordered = list(group_ids)
    rng.shuffle(ordered)
    raw_val = int(round(len(ordered) * val_fraction))
    val_count = min(len(ordered) - 1, max(1, raw_val))
    val_ids = set(ordered[:val_count])
    train_ids = set(ordered[val_count:])
    return train_ids, val_ids


def _evaluate_pairwise(model, dataset: PairwiseRerankerDataset, collator: PairwiseCollator) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    device = model.device
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collator)
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_scores = _score_from_logits(
                model(input_ids=batch['pos_input_ids'], attention_mask=batch['pos_attention_mask']).logits
            )
            neg_scores = _score_from_logits(
                model(input_ids=batch['neg_input_ids'], attention_mask=batch['neg_attention_mask']).logits
            )
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            acc = (pos_scores > neg_scores).float().mean()
            losses.append(float(loss.detach().cpu().item()))
            accs.append(float(acc.detach().cpu().item()))
    return {
        'pairwise_loss': sum(losses) / len(losses),
        'pairwise_accuracy': sum(accs) / len(accs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Train LoRA adapter for pretrained cross-encoder reranker on pairwise supervision.')
    ap.add_argument('--pairwise-file', type=Path, required=True)
    ap.add_argument('--docs-file', type=Path, required=True)
    ap.add_argument('--model-name', type=str, default='BAAI/bge-reranker-v2-m3')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--max-groups', type=int, default=-1)
    ap.add_argument('--no-fp16', action='store_true')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.1)
    ap.add_argument('--lora-target-modules', type=str, default='query,key,value')
    ap.add_argument('--modules-to-save', type=str, default='classifier')
    args = ap.parse_args()

    docs_lookup = _load_docs_lookup(args.docs_file)
    examples, skipped_rows = _build_examples(args.pairwise_file, docs_lookup)
    if not examples:
        raise ValueError('No trainable pairwise rows were built from the input file.')

    grouped: dict[str, list[PairwiseExample]] = defaultdict(list)
    for ex in examples:
        grouped[ex.group_id].append(ex)
    group_ids = sorted(grouped)
    if args.max_groups > 0:
        group_ids = group_ids[: args.max_groups]
        grouped = {gid: grouped[gid] for gid in group_ids}

    train_group_ids, val_group_ids = _split_group_ids(
        group_ids=group_ids,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    train_examples = [ex for gid in train_group_ids for ex in grouped[gid]]
    val_examples = [ex for gid in val_group_ids for ex in grouped[gid]]
    if not train_examples:
        raise ValueError('Train split is empty after grouping/splitting.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    if getattr(model.config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    modules_to_save = [m.strip() for m in args.modules_to_save.split(',') if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=lora_target_modules,
        modules_to_save=(modules_to_save or None),
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_dataset = PairwiseRerankerDataset(train_examples, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = PairwiseRerankerDataset(val_examples, tokenizer=tokenizer, max_length=args.max_length)
    collator = PairwiseCollator(tokenizer)

    trainer_out = args.out_dir / 'trainer_out'
    adapter_out = args.out_dir / 'adapter'
    metrics_file = args.out_dir / 'metrics.json'
    trainer_out.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        'output_dir': str(trainer_out),
        'learning_rate': float(args.lr),
        'weight_decay': float(args.weight_decay),
        'num_train_epochs': float(args.epochs),
        'per_device_train_batch_size': int(args.batch_size),
        'per_device_eval_batch_size': int(args.batch_size),
        'gradient_accumulation_steps': int(args.grad_accum),
        'logging_steps': int(args.logging_steps),
        'save_strategy': 'epoch',
        'remove_unused_columns': False,
        'report_to': [],
        'save_total_limit': int(args.save_total_limit),
        'seed': int(args.seed),
        'fp16': torch.cuda.is_available() and not args.no_fp16,
        'dataloader_pin_memory': torch.cuda.is_available(),
    }
    # Keep checkpointing each epoch, but disable Trainer's built-in eval loop.
    # Pairwise batches use custom keys (pos_/neg_) that default prediction_step
    # cannot route into model(**inputs) safely across transformer versions.
    eval_strategy_value = 'no'
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if 'evaluation_strategy' in ta_params:
        training_kwargs['evaluation_strategy'] = eval_strategy_value
    elif 'eval_strategy' in ta_params:
        training_kwargs['eval_strategy'] = eval_strategy_value

    training_args = TrainingArguments(**training_kwargs)

    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()

    adapter_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    train_loss = None
    if getattr(train_result, 'training_loss', None) is not None:
        train_loss = float(train_result.training_loss)
    elif isinstance(getattr(train_result, 'metrics', None), dict):
        metrics = train_result.metrics
        if metrics.get('train_loss') is not None:
            train_loss = float(metrics['train_loss'])
        elif metrics.get('loss') is not None:
            train_loss = float(metrics['loss'])
    if train_loss is None:
        for item in reversed(getattr(trainer.state, 'log_history', [])):
            if isinstance(item, dict) and item.get('train_loss') is not None:
                train_loss = float(item['train_loss'])
                break

    summary = {
        'model_name': args.model_name,
        'pairwise_file': str(args.pairwise_file),
        'docs_file': str(args.docs_file),
        'output_dir': str(args.out_dir),
        'adapter_dir': str(adapter_out),
        'trainer_output_dir': str(trainer_out),
        'total_examples': len(examples),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'total_groups': len(group_ids),
        'train_groups': len(train_group_ids),
        'val_groups': len(val_group_ids),
        'skipped_rows': skipped_rows,
        'epochs': float(args.epochs),
        'learning_rate': float(args.lr),
        'batch_size': int(args.batch_size),
        'grad_accum': int(args.grad_accum),
        'max_length': int(args.max_length),
        'val_fraction': float(args.val_fraction),
        'seed': int(args.seed),
        'fp16': bool(torch.cuda.is_available() and not args.no_fp16),
        'lora_r': int(args.lora_r),
        'lora_alpha': int(args.lora_alpha),
        'lora_dropout': float(args.lora_dropout),
        'lora_target_modules': lora_target_modules,
        'modules_to_save': modules_to_save,
        'train_loss': train_loss,
    }
    if val_examples:
        summary['validation'] = _evaluate_pairwise(model, val_dataset, collator)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

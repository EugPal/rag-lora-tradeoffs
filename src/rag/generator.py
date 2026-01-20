from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"


class BaseGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def generate(self, prompt: str) -> str:
        # Lightweight placeholder generator. Replace with a real LLM if needed.
        lines = prompt.strip().splitlines()
        question = lines[-1] if lines else ""
        return (
            "Answer (stub): This is a placeholder response for the query "
            f"'{question[:80]}'."
        )


class HFGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        do_sample = self.config.temperature > 0
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if do_sample else None,
            top_p=self.config.top_p if do_sample else None,
            do_sample=do_sample,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

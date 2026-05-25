from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4
    enable_thinking: bool | None = None
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_dir: str | None = None
    use_4bit: bool = True


def _stringify_prompt(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, list):
        return "\n".join(msg.get("content", "") for msg in prompt)
    return prompt


class BaseGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def generate(self, prompt: str | list[dict[str, str]]) -> str:
        # Lightweight placeholder generator. Replace with a real LLM if needed.
        text = _stringify_prompt(prompt)
        lines = text.strip().splitlines()
        question = lines[-1] if lines else ""
        return (
            "Answer (stub): This is a placeholder response for the query "
            f"'{question[:80]}'."
        )


class HFGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # Prefer 4-bit quantization on CUDA when enabled; otherwise use fp16 on GPU.
        if torch.cuda.is_available() and self.config.use_4bit:
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
        elif torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",
            )
        # Optional: attach a PEFT LoRA adapter.
        if self.config.lora_adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.config.lora_adapter_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def generate(self, prompt: str | list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            chat_template_kwargs = {
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            if self.config.enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = self.config.enable_thinking
            try:
                encoded = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
            except TypeError:
                # Backward compatibility for tokenizers without enable_thinking.
                chat_template_kwargs.pop("enable_thinking", None)
                encoded = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
            except ValueError:
                # Some tokenizers expose apply_chat_template but do not define chat_template.
                text = _stringify_prompt(prompt)
                encoded = self.tokenizer(text, return_tensors="pt")
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
        else:
            text = _stringify_prompt(prompt)
            encoded = self.tokenizer(text, return_tensors="pt")
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
            repetition_penalty=max(1.0, float(self.config.repetition_penalty)),
            no_repeat_ngram_size=max(0, int(self.config.no_repeat_ngram_size)),
            do_sample=do_sample,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()



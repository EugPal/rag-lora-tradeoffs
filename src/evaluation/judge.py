from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    max_tokens: int = 256
    # Default to deterministic scoring for reproducibility.
    temperature: float = 0.0
    top_p: float = 1.0
    log_first_n: int = 3


class LLMJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self._logged = 0
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

    def _format_prompt(self, question: str, answer: str, contexts: list[str]) -> str:
        context_block = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        return (
            "You are a strict evaluator for a RAG system.\n"
            "Important rules:\n"
            "- Score 5 is rare and given ONLY if every claim is explicitly supported.\n"
            "- If ANY claim is not supported, correctness <= 3.\n"
            "- Evidence MUST be a non-empty verbatim quote copied from one of the Contexts.\n"
            "- If you cannot provide a verbatim quote, set groundedness=1.\n"
            "- If evidence is empty, set groundedness=1 and correctness<=3.\n"
            "- When in doubt, choose the LOWER score.\n\n"
            "Examples:\n"
            "Bad answer:\n"
            "Answer: \"FastAPI runs on Django.\"\n"
            "Score: correctness=1, groundedness=1, evidence=\"FastAPI is a modern, fast...\"  # if present; otherwise groundedness=1\n"
            "Rationale: Not supported by contexts.\n\n"
            "Partial answer:\n"
            "Answer: \"FastAPI is installed with pip.\"\n"
            "Score: correctness=3, groundedness=1, evidence=\"...\"  # only if the quote exists in Contexts\n"
            "Rationale: Context does not mention installation.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"{context_block}\n\n"
            "Scoring rubric:\n"
            "- correctness 5: fully correct and complete per contexts\n"
            "- correctness 3: partially correct or missing key details\n"
            "- correctness 1: incorrect or mostly unsupported\n"
            "- groundedness 5: all claims supported by contexts (quoteable)\n"
            "- groundedness 3: mixed support, some minor unsupported claims\n"
            "- groundedness 1: largely unsupported\n\n"
            "Return ONLY JSON (no markdown, no extra text) with keys:\n"
            "- correctness: integer 1-5\n"
            "- groundedness: integer 1-5\n"
            "- evidence: a short NON-EMPTY verbatim quote from the Contexts\n"
            "- rationale: short string\n"
        )

    def judge(self, question: str, answer: str, contexts: list[str]) -> dict:
        prompt = self._format_prompt(question, answer, contexts)
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

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            top_p=self.config.top_p if self.config.temperature > 0 else None,
            do_sample=self.config.temperature > 0,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if self._logged < self.config.log_first_n:
            print(f"[judge raw] {text}")
            self._logged += 1
        return parse_judge_output(text)


def parse_judge_output(text: str) -> dict:
    fenced_match = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        correctness_match = re.search(r"correctness\s*:\s*([1-5])", text, flags=re.IGNORECASE)
        grounded_match = re.search(r"groundedness\s*:\s*([1-5])", text, flags=re.IGNORECASE)
        rationale_match = re.search(r"rationale\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        evidence_match = re.search(r"evidence\s*:\s*(.*)", text, flags=re.IGNORECASE)
        return {
            "correctness": int(correctness_match.group(1)) if correctness_match else None,
            "groundedness": int(grounded_match.group(1)) if grounded_match else None,
            "evidence": (evidence_match.group(1) if evidence_match else "")[:200],
            "rationale": (rationale_match.group(1) if rationale_match else text)[:200],
        }
    try:
        raw = match.group(0)
        # Best-effort cleanup: remove trailing commas before closing braces/brackets
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(raw)
        return {
            "correctness": data.get("correctness"),
            "groundedness": data.get("groundedness"),
            "evidence": data.get("evidence", "")[:200],
            "rationale": data.get("rationale", "")[:200],
        }
    except json.JSONDecodeError:
        # Fallback: parse scalar fields even if JSON is malformed/truncated.
        correctness_match = re.search(r"correctness\"?\s*:\s*([1-5])", text, flags=re.IGNORECASE)
        grounded_match = re.search(r"groundedness\"?\s*:\s*([1-5])", text, flags=re.IGNORECASE)
        evidence_match = re.search(r"evidence\"?\s*:\s*\"(.*?)\"", text, flags=re.IGNORECASE | re.DOTALL)
        rationale_match = re.search(r"rationale\"?\s*:\s*\"(.*?)\"", text, flags=re.IGNORECASE | re.DOTALL)
        return {
            "correctness": int(correctness_match.group(1)) if correctness_match else None,
            "groundedness": int(grounded_match.group(1)) if grounded_match else None,
            "evidence": (evidence_match.group(1) if evidence_match else "")[:200],
            "rationale": (rationale_match.group(1) if rationale_match else text)[:200],
        }

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_tokens: int = 256
    # Default to deterministic scoring for reproducibility.
    temperature: float = 0.0
    top_p: float = 1.0
    log_first_n: int = 3
    use_4bit: bool = True


class LLMJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self._logged = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.config.use_4bit:
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
        else:
            if torch.cuda.is_available():
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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _format_prompt(self, question: str, answer: str, contexts: list[str]) -> str:
        context_block = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        return (
            "You are a strict evaluator for a RAG system.\n"
            "Important rules:\n"
            "- Score 5 is rare and given ONLY if every claim is explicitly supported by the contexts.\n"
            "- If ANY key claim is unsupported, correctness should be <= 3.\n"
            "- Evidence should be a short supporting span (quote or near-verbatim phrase) from contexts when possible.\n"
            "- If exact quote is unavailable but support is clear in contexts, do NOT force groundedness=1.\n"
            "- Use groundedness=1 only when support is largely missing.\n"
            "- When in doubt, choose the LOWER score.\n\n"
            "Examples:\n"
            "Bad answer:\n"
            "Answer: \"FastAPI runs on Django.\"\n"
            "Score: correctness=1, groundedness=1, evidence=\"...\"\n"
            "Rationale: Not supported by contexts.\n\n"
            "Partial answer:\n"
            "Answer: \"FastAPI is installed with pip.\"\n"
            "Score: correctness=3, groundedness=2 or 3, evidence=\"...\"\n"
            "Rationale: Some support may exist, but key details are missing.\n\n"
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
            "- evidence: a short supporting span from the Contexts (quote or near-verbatim)\n"
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


def _safe_short_text(value: object, max_len: int = 200) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    else:
        value = str(value)
    return value[:max_len]


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
            "evidence": _safe_short_text(evidence_match.group(1) if evidence_match else ""),
            "rationale": _safe_short_text(rationale_match.group(1) if rationale_match else text),
        }
    try:
        raw = match.group(0)
        # Best-effort cleanup: remove trailing commas before closing braces/brackets
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(raw)
        return {
            "correctness": data.get("correctness"),
            "groundedness": data.get("groundedness"),
            "evidence": _safe_short_text(data.get("evidence", "")),
            "rationale": _safe_short_text(data.get("rationale", "")),
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
            "evidence": _safe_short_text(evidence_match.group(1) if evidence_match else ""),
            "rationale": _safe_short_text(rationale_match.group(1) if rationale_match else text),
        }

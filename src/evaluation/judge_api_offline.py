from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import ssl
import time
from http.client import RemoteDisconnected
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from src.utils.io_utils import ensure_dir, read_jsonl
from src.utils.logging_utils import setup_logging


JUDGE_PROMPT_VERSION = "groundedness_v1_raw"

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator for a retrieval-augmented question answering system.

You will be given:
- a question,
- retrieved context,
- a model answer.

Your job is to evaluate the answer using only the retrieved context.

Evaluate two dimensions:

1. correctness
How correct the answer is with respect to the retrieved context.

2. groundedness
How well the answer is supported by the retrieved context and whether it avoids unsupported claims.

Important rules:
- Use only the retrieved context.
- Do not use outside knowledge.
- Be strict.
- Score 5 should be rare.
- If support is ambiguous, prefer the lower score.
- If the answer contains unsupported claims, groundedness must be reduced.
- If the answer is factually wrong with respect to the context, correctness must be reduced.
- If the answer is correct but not fully supported by the context, correctness may be higher than groundedness.
- If the answer is well supported by the context but incomplete, groundedness may be higher than correctness.

Scoring rubric:

Correctness:
1 = incorrect
2 = mostly incorrect
3 = partially correct
4 = mostly correct, with minor issues
5 = fully correct with respect to the context

Groundedness:
1 = unsupported by context
2 = weakly supported, with substantial unsupported content
3 = partially supported, with noticeable unsupported or overgeneralized content
4 = mostly supported, with only minor unsupported content
5 = fully supported by context, with no meaningful unsupported claims

Return ONLY valid JSON in exactly this format:
{{
  "correctness": <integer 1-5>,
  "groundedness": <integer 1-5>,
  "evidence": "<short supporting span from the retrieved context, or empty string if none>",
  "rationale": "<brief explanation>"
}}

Question:
{question}

Retrieved context:
{context}

Model answer:
{answer}
"""


@dataclass
class JobMeta:
    job_id: str
    config: str
    base_model_name_or_path: str
    rank: int | None
    target_mode: str
    checkpoint_dir: str | None
    predictions_file: str
    results_file: str
    adapter_config_file: str | None
    f1: float | None
    em: float | None
    latency_mean_s: float | None
    peak_gpu_memory_inference_gb: float | None


def format_context(chunks: list[str]) -> str:
    if not chunks:
        return "[Chunk 1]\n"
    return "\n\n".join(f"[Chunk {idx}]\n{chunk}" for idx, chunk in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: list[str], answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=(question or "").strip(),
        context=format_context(chunks),
        answer=(answer or "").strip(),
    )


def _to_int_1_5(value: Any) -> int | None:
    try:
        ivalue = int(value)
    except Exception:
        return None
    return ivalue if 1 <= ivalue <= 5 else None


def _short_text(value: Any, max_len: int = 300) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    else:
        value = str(value)
    return value.strip()[:max_len]


def parse_judge_output(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        raw = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            data = json.loads(raw)
            return {
                "correctness": _to_int_1_5(data.get("correctness")),
                "groundedness": _to_int_1_5(data.get("groundedness")),
                "evidence": _short_text(data.get("evidence")),
                "rationale": _short_text(data.get("rationale")),
            }
        except json.JSONDecodeError:
            pass
    correctness_match = re.search(r'correctness"?\s*:\s*([1-5])', text, flags=re.IGNORECASE)
    grounded_match = re.search(r'groundedness"?\s*:\s*([1-5])', text, flags=re.IGNORECASE)
    evidence_match = re.search(r'evidence"?\s*:\s*"?(.*?)"?(?:,|\n|$)', text, flags=re.IGNORECASE | re.DOTALL)
    rationale_match = re.search(r'rationale"?\s*:\s*"?(.*?)"?(?:\n|$)', text, flags=re.IGNORECASE | re.DOTALL)
    return {
        "correctness": int(correctness_match.group(1)) if correctness_match else None,
        "groundedness": int(grounded_match.group(1)) if grounded_match else None,
        "evidence": _short_text(evidence_match.group(1) if evidence_match else ""),
        "rationale": _short_text(rationale_match.group(1) if rationale_match else ""),
    }


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def call_openai_responses_api(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    store: bool,
    timeout_s: int,
    retry_count: int,
    retry_base_sleep_s: float,
    retry_max_sleep_s: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "store": store,
    }
    data = json.dumps(body).encode("utf-8")
    req = Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            with urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = extract_response_text(payload)
            return text, payload
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retriable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            last_error = RuntimeError(f"Judge API HTTPError {exc.code}: {detail}")
            if not retriable or attempt >= retry_count:
                raise last_error from exc
        except (URLError, ssl.SSLError, TimeoutError, RemoteDisconnected, ConnectionResetError) as exc:
            last_error = RuntimeError(f"Judge API connection failed: {exc}")
            if attempt >= retry_count:
                raise last_error from exc
        sleep_s = min(retry_max_sleep_s, retry_base_sleep_s * (2 ** (attempt - 1)))
        sleep_s += random.uniform(0.0, min(1.0, sleep_s * 0.25))
        time.sleep(sleep_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Judge API request failed without an explicit error")


def build_job_summary_row(
    *,
    job: JobMeta,
    rows: list[dict[str, Any]],
    sample_total: int,
    judge_model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "config": job.config,
        "base_model_name_or_path": job.base_model_name_or_path,
        "rank": job.rank,
        "target_mode": job.target_mode,
        "checkpoint_dir": job.checkpoint_dir,
        "source_prediction_file": job.predictions_file,
        "source_results_file": job.results_file,
        "f1": job.f1,
        "em": job.em,
        "latency_mean_s": job.latency_mean_s,
        "peak_gpu_memory_inference_gb": job.peak_gpu_memory_inference_gb,
        "judge_model": judge_model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "samples_total": sample_total,
        "samples_scored": len([r for r in rows if r.get("correctness") is not None]),
        "judge_correctness_avg": aggregate_scores(rows, "correctness"),
        "judge_groundedness_avg": aggregate_scores(rows, "groundedness"),
        "correctness_pass@4": pass_rate(rows, "correctness", 4),
        "groundedness_pass@4": pass_rate(rows, "groundedness", 4),
    }


def infer_target_mode(target_modules: list[str]) -> str:
    target_set = set(target_modules or [])
    if target_set == {"q_proj", "v_proj"}:
        return "qv_only"
    if target_set == {"q_proj", "k_proj", "o_proj", "v_proj"}:
        return "full_attention"
    return ",".join(sorted(target_set))


def infer_model_size(base_model_name_or_path: str) -> str:
    if "8B" in base_model_name_or_path:
        return "8B"
    if "3B" in base_model_name_or_path:
        return "3B"
    return base_model_name_or_path


def load_metadata_by_job(metadata_file: Path | None) -> dict[str, dict[str, Any]]:
    if metadata_file is None or not metadata_file.exists():
        return {}
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    if isinstance(data, list):
        for row in data:
            job_id = str(row.get("job_id") or "").strip()
            if job_id:
                out[job_id] = row
    return out


def discover_job_files(job_dir: Path) -> tuple[Path, Path, Path | None]:
    adapter_config_file = job_dir / "adapter_config.json"
    predictions_file = job_dir / "predictions.jsonl"
    results_file = job_dir / "results.json"
    if predictions_file.exists() and results_file.exists():
        return predictions_file, results_file, (adapter_config_file if adapter_config_file.exists() else None)

    prediction_candidates = sorted(job_dir.glob("predictions*.jsonl"))
    result_candidates = sorted(job_dir.glob("results*.json"))
    if len(prediction_candidates) == 1 and len(result_candidates) == 1:
        return prediction_candidates[0], result_candidates[0], (adapter_config_file if adapter_config_file.exists() else None)

    missing = []
    if not prediction_candidates:
        missing.append("predictions*.jsonl")
    if not result_candidates:
        missing.append("results*.json")
    if missing:
        raise FileNotFoundError(", ".join(missing))
    raise RuntimeError(
        f"Ambiguous job files in {job_dir}: predictions={len(prediction_candidates)} results={len(result_candidates)}"
    )


def infer_baseline_base_model(results: dict[str, Any]) -> str:
    peak_vram = results.get("peak_gpu_memory_inference_gb")
    try:
        peak = float(peak_vram)
    except Exception:
        peak = None
    if peak is not None and peak >= 18.0:
        return "meta-llama/Llama-3.1-8B-Instruct"
    return "meta-llama/Llama-3.2-3B-Instruct"


def load_job_meta(job_dir: Path, metadata_by_job: dict[str, dict[str, Any]]) -> JobMeta:
    predictions_file, results_file, adapter_config_file = discover_job_files(job_dir)
    results = json.loads(results_file.read_text(encoding="utf-8"))
    meta = metadata_by_job.get(job_dir.name, {})

    if adapter_config_file is not None and adapter_config_file.exists():
        adapter_config = json.loads(adapter_config_file.read_text(encoding="utf-8"))
        base_model_name_or_path = adapter_config.get("base_model_name_or_path", "")
        rank = adapter_config.get("r")
        target_mode = infer_target_mode(adapter_config.get("target_modules") or [])
        config = f"{infer_model_size(base_model_name_or_path)} r{rank} {target_mode}"
    else:
        base_model_name_or_path = str(meta.get("base_model") or infer_baseline_base_model(results))
        rank = meta.get("rank")
        target_mode = str(meta.get("target_label") or ("baseline" if results.get("lora_adapter") is None else "unknown"))
        if target_mode == "baseline":
            config = f"{infer_model_size(base_model_name_or_path)} baseline"
        elif rank is None:
            config = str(meta.get("config") or f"{infer_model_size(base_model_name_or_path)} {target_mode}")
        else:
            config = f"{infer_model_size(base_model_name_or_path)} r{rank} {target_mode}"

    return JobMeta(
        job_id=job_dir.name,
        config=config,
        base_model_name_or_path=base_model_name_or_path,
        rank=rank,
        target_mode=target_mode,
        checkpoint_dir=results.get("lora_adapter"),
        predictions_file=str(predictions_file),
        results_file=str(results_file),
        adapter_config_file=str(adapter_config_file) if adapter_config_file is not None else None,
        f1=results.get("f1"),
        em=results.get("em"),
        latency_mean_s=results.get("latency_mean_s"),
        peak_gpu_memory_inference_gb=results.get("peak_gpu_memory_inference_gb"),
    )


def aggregate_scores(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def pass_rate(rows: list[dict[str, Any]], field: str, threshold: int) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(1 for value in values if value >= threshold) / len(values)


def format_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return '[{}]'.format('-' * width)
    filled = int(width * current / total)
    filled = max(0, min(width, filled))
    return '[{}{}]'.format('#' * filled, '-' * (width - filled))


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=True) + '\n')


def build_summary_payload(
    *,
    args: argparse.Namespace,
    excluded: set[str],
    skipped_jobs: list[dict[str, Any]],
    ready_jobs: list[JobMeta],
    total_scored: int,
    prompt_sha256: str,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        'judge_model': args.model,
        'prompt_version': JUDGE_PROMPT_VERSION,
        'prompt_sha256': prompt_sha256,
        'temperature': args.temperature,
        'store': args.store,
        'api_key_env': args.api_key_env,
        'pareto_test_dir': str(args.pareto_test_dir),
        'excluded_jobs': sorted(excluded),
        'skipped_jobs': skipped_jobs,
        'jobs_total': len(ready_jobs) + len(skipped_jobs),
        'jobs_ready': len(ready_jobs),
        'examples_scored': total_scored,
        'rows': summary_rows,
    }


def write_summary_file(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline API judge over saved best-checkpoint predictions.")
    parser.add_argument("--pareto-test-dir", type=Path, default=Path("actual/datasets/pareto_test_best"))
    parser.add_argument("--metadata-file", type=Path, default=Path("actual/datasets/pareto_front/pareto_front_points.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-jobs", type=int, default=-1)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--api-retry-count", type=int, default=8)
    parser.add_argument("--api-retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--api-retry-max-sleep", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--exclude-job", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("judge_api_offline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("actual/datasets/judge_api_runs") / timestamp
    ensure_dir(out_dir)
    prompt_sha256 = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

    excluded = set(args.exclude_job)
    metadata_by_job = load_metadata_by_job(args.metadata_file)
    ready_jobs: list[JobMeta] = []
    skipped_jobs: list[dict[str, Any]] = []

    for job_dir in sorted(args.pareto_test_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        if job_dir.name in excluded:
            skipped_jobs.append({"job_id": job_dir.name, "reason": "user_excluded"})
            continue
        try:
            ready_jobs.append(load_job_meta(job_dir, metadata_by_job))
        except Exception as exc:
            skipped_jobs.append({"job_id": job_dir.name, "reason": str(exc)})
            continue

    if skipped_jobs:
        logger.warning("Skipped %d jobs with missing files or exclusions", len(skipped_jobs))
    if args.fail_on_missing and skipped_jobs:
        raise RuntimeError(f"Skipped jobs detected: {skipped_jobs}")
    if not ready_jobs:
        raise RuntimeError("No ready jobs found in pareto_test_best")
    if args.max_jobs > 0:
        ready_jobs = ready_jobs[: args.max_jobs]

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is required")

    results_path = out_dir / "judge_results.jsonl"
    summary_path = out_dir / "judge_summary.json"
    if args.overwrite and results_path.exists():
        results_path.unlink()
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    existing_rows = read_jsonl(results_path) if results_path.exists() else []
    rows_by_job: dict[str, list[dict[str, Any]]] = {}
    seen_example_ids_by_job: dict[str, set[str]] = {}
    for row in existing_rows:
        job_id = str(row.get("job_id") or "")
        example_id = str(row.get("example_id") or "")
        if not job_id or not example_id:
            continue
        rows_by_job.setdefault(job_id, []).append(row)
        seen_example_ids_by_job.setdefault(job_id, set()).add(example_id)

    total_scored = sum(1 for row in existing_rows if row.get("correctness") is not None)
    summary_rows: list[dict[str, Any]] = []
    for job in ready_jobs:
        existing_job_rows = rows_by_job.get(job.job_id, [])
        if existing_job_rows:
            sample_rows = read_jsonl(job.predictions_file)
            if args.max_samples > 0:
                sample_rows = sample_rows[: args.max_samples]
            summary_rows.append(
                build_job_summary_row(
                    job=job,
                    rows=existing_job_rows,
                    sample_total=len(sample_rows),
                    judge_model=args.model,
                    prompt_sha256=prompt_sha256,
                )
            )

    total_jobs = len(ready_jobs)
    logger.info("Starting judge run: %d jobs", total_jobs)
    write_summary_file(
        summary_path,
        build_summary_payload(
            args=args,
            excluded=excluded,
            skipped_jobs=skipped_jobs,
            ready_jobs=ready_jobs,
            total_scored=total_scored,
            prompt_sha256=prompt_sha256,
            summary_rows=summary_rows,
        ),
    )
    for job_idx, job in enumerate(ready_jobs, start=1):
        prediction_rows = read_jsonl(job.predictions_file)
        if args.max_samples > 0:
            prediction_rows = prediction_rows[: args.max_samples]
        sample_total = len(prediction_rows)
        processed_example_ids = seen_example_ids_by_job.get(job.job_id, set())
        logger.info(
            "Starting job %d/%d: %s (%s) on %d samples | already_scored=%d",
            job_idx,
            total_jobs,
            job.job_id,
            job.config,
            sample_total,
            len(processed_example_ids),
        )

        per_job_rows: list[dict[str, Any]] = list(rows_by_job.get(job.job_id, []))
        for sample_idx, row in enumerate(prediction_rows, start=1):
            example_id = str(row.get("id") or "")
            if example_id and example_id in processed_example_ids:
                if (
                    sample_idx == sample_total
                    or sample_idx == 1
                    or (args.progress_every > 0 and sample_idx % args.progress_every == 0)
                ):
                    logger.info(
                        "Job %d/%d %s %s %d/%d (%.1f%%) | scored_total=%d | resumed_skip",
                        job_idx,
                        total_jobs,
                        job.job_id,
                        format_progress_bar(sample_idx, sample_total),
                        sample_idx,
                        sample_total,
                        (100.0 * sample_idx / sample_total) if sample_total else 100.0,
                        total_scored,
                    )
                continue
            prompt = build_prompt(
                question=row.get("question", ""),
                chunks=row.get("chunks") or [],
                answer=row.get("prediction", ""),
            )
            last_text = ""
            last_payload: dict[str, Any] = {}
            parsed: dict[str, Any] = {
                "correctness": None,
                "groundedness": None,
                "evidence": "",
                "rationale": "",
            }
            for attempt in range(1, args.retry_count + 1):
                last_text, last_payload = call_openai_responses_api(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    store=args.store,
                    timeout_s=args.timeout_s,
                    retry_count=args.api_retry_count,
                    retry_base_sleep_s=args.api_retry_base_sleep,
                    retry_max_sleep_s=args.api_retry_max_sleep,
                )
                parsed = parse_judge_output(last_text)
                if parsed["correctness"] is not None and parsed["groundedness"] is not None:
                    break
                logger.warning(
                    "Unparseable judge output for %s/%s attempt %d",
                    job.job_id,
                    row.get("id"),
                    attempt,
                )
                time.sleep(1.0)

            result_row = {
                "job_id": job.job_id,
                "config": job.config,
                "base_model_name_or_path": job.base_model_name_or_path,
                "rank": job.rank,
                "target_mode": job.target_mode,
                "checkpoint_dir": job.checkpoint_dir,
                "source_prediction_file": job.predictions_file,
                "source_results_file": job.results_file,
                "source_adapter_config_file": job.adapter_config_file,
                "example_id": row.get("id"),
                "question": row.get("question"),
                "prediction": row.get("prediction"),
                "gold": row.get("gold"),
                "chunks": row.get("chunks") or [],
                "correctness": parsed["correctness"],
                "groundedness": parsed["groundedness"],
                "evidence": parsed["evidence"],
                "rationale": parsed["rationale"],
                "judge_text": last_text,
                "judge_response": last_payload,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "judge_model": args.model,
            }
            per_job_rows.append(result_row)
            rows_by_job[job.job_id] = per_job_rows
            if example_id:
                seen_example_ids_by_job.setdefault(job.job_id, set()).add(example_id)
            append_jsonl_row(results_path, result_row)
            if parsed["correctness"] is not None:
                total_scored += 1
            if (
                sample_idx == sample_total
                or sample_idx == 1
                or (args.progress_every > 0 and sample_idx % args.progress_every == 0)
            ):
                logger.info(
                    "Job %d/%d %s %s %d/%d (%.1f%%) | scored_total=%d",
                    job_idx,
                    total_jobs,
                    job.job_id,
                    format_progress_bar(sample_idx, sample_total),
                    sample_idx,
                    sample_total,
                    (100.0 * sample_idx / sample_total) if sample_total else 100.0,
                    total_scored,
                )

        logger.info(
            "Finished job %d/%d: %s | correctness_avg=%s groundedness_avg=%s",
            job_idx,
            total_jobs,
            job.job_id,
            aggregate_scores(per_job_rows, "correctness"),
            aggregate_scores(per_job_rows, "groundedness"),
        )
        summary_rows = [row for row in summary_rows if row.get("job_id") != job.job_id]
        summary_rows.append(
            build_job_summary_row(
                job=job,
                rows=per_job_rows,
                sample_total=len(prediction_rows),
                judge_model=args.model,
                prompt_sha256=prompt_sha256,
            )
        )
        write_summary_file(
            summary_path,
            build_summary_payload(
                args=args,
                excluded=excluded,
                skipped_jobs=skipped_jobs,
                ready_jobs=ready_jobs,
                total_scored=total_scored,
                prompt_sha256=prompt_sha256,
                summary_rows=summary_rows,
            ),
        )
        logger.info("Checkpointed outputs after job %d/%d", job_idx, total_jobs)

    logger.info("Saved per-example results to %s", results_path)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()

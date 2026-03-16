from __future__ import annotations

import hashlib
import re


CATEGORY_SHARES: dict[str, float] = {
    "endpoints_routing": 0.15,
    "pydantic_validation": 0.15,
    "dependencies_di": 0.15,
    "security_auth": 0.15,
    "async_concurrency": 0.10,
    "testing": 0.10,
    "deployment_ops": 0.08,
    "middleware_lifespan_background": 0.07,
    "errors_debugging": 0.05,
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def source_page_id(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    return re.sub(r"-\d+$", "", chunk_id)


def infer_section(page_id: str | None) -> str:
    if not page_id:
        return "general"
    # Handle localized page IDs, e.g. "es_tutorial_path-params".
    m = re.match(r"^([a-z]{2}(?:-[a-z]+)?)_(.+)$", page_id)
    if m:
        page_id = m.group(2)
    if page_id.startswith("tutorial_"):
        return "tutorial"
    if page_id.startswith("advanced_"):
        return "advanced"
    if page_id.startswith("reference_"):
        return "reference"
    if page_id.startswith("how-to_"):
        return "how_to"
    if page_id.startswith("deployment_"):
        return "deployment"
    return "general"


def infer_category(question: str, answer: str, page_id: str | None) -> str:
    text = normalize(f"{question} {answer} {page_id or ''}")

    if any(k in text for k in ["oauth", "jwt", "security", "bearer", "auth"]):
        return "security_auth"
    if any(k in text for k in ["depends", "dependency", "sub dependency"]):
        return "dependencies_di"
    if any(k in text for k in ["async", "await", "concurrency", "event loop"]):
        return "async_concurrency"
    if any(k in text for k in ["testclient", "pytest", "testing", "test "]):
        return "testing"
    if any(k in text for k in ["middleware", "lifespan", "backgroundtasks", "background task"]):
        return "middleware_lifespan_background"
    if any(k in text for k in ["422", "error", "exception", "debug"]):
        return "errors_debugging"
    if any(k in text for k in ["deployment", "docker", "worker", "https", "cloud"]):
        return "deployment_ops"
    if any(k in text for k in ["pydantic", "model", "validation", "schema", "response_model"]):
        return "pydantic_validation"
    if any(k in text for k in ["route", "router", "path", "query", "endpoint", "get ", "post ", "put ", "delete "]):
        return "endpoints_routing"
    return "endpoints_routing"


def deterministic_hash_int(seed: int, value: str) -> int:
    h = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def allocate_category_targets(total: int) -> dict[str, int]:
    raw = {k: CATEGORY_SHARES[k] * total for k in CATEGORY_SHARES}
    base = {k: int(raw[k]) for k in raw}
    remainder = total - sum(base.values())
    ranked = sorted(raw.keys(), key=lambda k: (raw[k] - base[k]), reverse=True)
    for i in range(remainder):
        base[ranked[i % len(ranked)]] += 1
    return base

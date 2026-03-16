from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src.utils.io_utils import read_jsonl, write_jsonl


def _base_page(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    return re.sub(r"-\d+$", "", chunk_id)


def _chunk_suffix(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    m = re.search(r"-(\d+)$", chunk_id)
    return m.group(1) if m else None


def _resolve_gold_doc_id(row: dict, docs_by_id: dict[str, str]) -> str | None:
    source_chunk = str(row.get("source_chunk") or "").strip()
    if source_chunk and source_chunk in docs_by_id:
        return source_chunk
    if source_chunk.startswith("qa-"):
        no_prefix = source_chunk[3:]
        if no_prefix in docs_by_id:
            return no_prefix
    source_page = str(row.get("source_page") or "").strip()
    suffix = _chunk_suffix(source_chunk)
    if source_page and suffix:
        candidate = f"{source_page}-{suffix}"
        if candidate in docs_by_id:
            return candidate
    return None


def _pick_distractors(
    row: dict,
    gold_doc_id: str | None,
    page_to_ids: dict[str, list[str]],
    all_doc_ids: list[str],
    count: int,
    seed: int,
) -> list[str]:
    row_id = str(row.get("id") or "")
    source_page = str(row.get("source_page") or "")
    rng = random.Random(f"{seed}:{row_id}")

    chosen: list[str] = []
    blocked = {gold_doc_id} if gold_doc_id else set()

    page_pool = [cid for cid in page_to_ids.get(source_page, []) if cid not in blocked]
    rng.shuffle(page_pool)
    for cid in page_pool:
        if len(chosen) >= count:
            break
        chosen.append(cid)

    if len(chosen) < count:
        global_pool = [cid for cid in all_doc_ids if cid not in blocked and cid not in set(chosen)]
        rng.shuffle(global_pool)
        need = count - len(chosen)
        chosen.extend(global_pool[:need])

    return chosen[:count]


def _augment_rows(
    rows: list[dict],
    docs_by_id: dict[str, str],
    page_to_ids: dict[str, list[str]],
    seed: int,
) -> list[dict]:
    all_doc_ids = list(docs_by_id.keys())
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        gold_doc_id = _resolve_gold_doc_id(item, docs_by_id)
        distractors = _pick_distractors(
            row=item,
            gold_doc_id=gold_doc_id,
            page_to_ids=page_to_ids,
            all_doc_ids=all_doc_ids,
            count=2,
            seed=seed,
        )

        source_chunk = str(item.get("source_chunk") or "")
        source_chunk_text = item.get("source_chunk_text")
        gold_text = (
            docs_by_id.get(gold_doc_id or "")
            or (source_chunk_text if isinstance(source_chunk_text, str) else None)
            or ""
        )
        gold_id_for_context = gold_doc_id or source_chunk

        context_chunks = [
            {"id": gold_id_for_context, "text": gold_text, "is_gold": True},
        ]
        for cid in distractors:
            context_chunks.append({"id": cid, "text": docs_by_id.get(cid, ""), "is_gold": False})

        item["context_chunks"] = context_chunks
        item["context_chunk_ids"] = [c["id"] for c in context_chunks]
        item["context_size"] = len(context_chunks)
        out.append(item)
    return out


def _process_pair(
    json_path: Path,
    jsonl_path: Path,
    docs_by_id: dict[str, str],
    page_to_ids: dict[str, list[str]],
    seed: int,
) -> tuple[int, int]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    updated = _augment_rows(rows, docs_by_id, page_to_ids, seed=seed)
    json_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(jsonl_path, updated)
    return len(rows), len(updated)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add context triplets (gold + 2 distractors) to main gold/eval/test datasets."
    )
    ap.add_argument("--docs-file", type=Path, default=Path("data/processed/docs.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    docs = read_jsonl(args.docs_file)
    docs_by_id = {str(r["id"]): str(r.get("text", "")) for r in docs if r.get("id")}
    page_to_ids: dict[str, list[str]] = {}
    for doc_id in docs_by_id:
        page = _base_page(doc_id)
        if not page:
            continue
        page_to_ids.setdefault(page, []).append(doc_id)

    pairs = [
        ("qa_gold_main.json", "qa_gold_main.jsonl"),
        ("qa_eval_main.json", "qa_eval_main.jsonl"),
        ("qa_test_main.json", "qa_test_main.jsonl"),
    ]
    for json_name, jsonl_name in pairs:
        json_path = args.processed_dir / json_name
        jsonl_path = args.processed_dir / jsonl_name
        before, after = _process_pair(
            json_path=json_path,
            jsonl_path=jsonl_path,
            docs_by_id=docs_by_id,
            page_to_ids=page_to_ids,
            seed=args.seed,
        )
        print(f"{json_path.name}: {before} rows updated ({after})")


if __name__ == "__main__":
    main()

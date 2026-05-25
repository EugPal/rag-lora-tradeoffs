from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging

GITHUB_TREE_API = "https://api.github.com/repos/kubernetes/website/git/trees/main?recursive=1"
DOCS_PREFIX = "content/en/docs/"
SITE_PREFIX = "https://kubernetes.io/docs/"


def fetch_repo_tree(api_url: str) -> dict:
    req = Request(
        api_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def keep_tree_path(path: str) -> bool:
    if not path.startswith(DOCS_PREFIX):
        return False
    if not path.endswith(".md"):
        return False
    name = path.rsplit("/", 1)[-1]
    if name.startswith("_") and name != "_index.md":
        return False
    if name.lower() == "readme.md":
        return False
    if path.startswith("content/en/docs/contribute/blog/"):
        return False
    if path.startswith("content/en/docs/test/"):
        return False
    if re.match(r"^content/en/docs/reference/command-line-tools-reference/feature-gates/[^/]+\.md$", path):
        return False
    if re.match(r"^content/en/docs/reference/glossary/[^/]+\.md$", path):
        return False
    return True


def docs_path_to_url(tree_path: str) -> str:
    rel = tree_path[len(DOCS_PREFIX) :]
    if rel == "_index.md":
        return SITE_PREFIX
    if rel.endswith("/_index.md"):
        rel = rel[: -len("/_index.md")]
        return SITE_PREFIX + rel.strip("/") + "/"
    rel = rel[: -len(".md")]
    return SITE_PREFIX + rel.strip("/") + "/"


def top_section_from_url(url: str) -> str:
    suffix = url[len(SITE_PREFIX) :].strip("/")
    if not suffix:
        return "root"
    return suffix.split("/", 1)[0]


def slug_from_url(url: str) -> str:
    suffix = url[len(SITE_PREFIX) :].strip("/")
    if not suffix:
        return "docs_home"
    suffix = suffix.replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", suffix).strip("_").lower()


def build_stats(urls: list[str]) -> dict:
    by_section: dict[str, int] = {}
    for url in urls:
        label = top_section_from_url(url)
        by_section[label] = by_section.get(label, 0) + 1
    return {
        "total_urls": len(urls),
        "by_section": dict(sorted(by_section.items())),
        "sample_ids": [slug_from_url(url) for url in urls[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build English Kubernetes docs URL list from the official website repo.")
    parser.add_argument("--tree-api", type=str, default=GITHUB_TREE_API)
    parser.add_argument("--out-file", type=Path, default=Path("data/raw/kubernetes_urls.txt"))
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("data/processed/fresh_start/kubernetes/kubernetes_url_list_stats.json"),
    )
    parser.add_argument("--max-urls", type=int, default=0, help="Optional limit for smoke tests (0 = all).")
    args = parser.parse_args()

    logger = setup_logging("build_kubernetes_url_list")
    payload = fetch_repo_tree(args.tree_api)
    if payload.get("truncated"):
        raise RuntimeError("GitHub tree API returned a truncated payload; refusing to build a partial URL list.")

    tree = payload.get("tree", [])
    urls = [docs_path_to_url(item["path"]) for item in tree if item.get("type") == "blob" and keep_tree_path(item["path"])]

    dedup: list[str] = []
    seen = set()
    for url in sorted(urls):
        if url in seen:
            continue
        seen.add(url)
        dedup.append(url)
    urls = dedup

    if args.max_urls > 0:
        urls = urls[: args.max_urls]

    ensure_dir(args.out_file.parent)
    args.out_file.write_text("\n".join(urls) + "\n", encoding="utf-8")

    stats = build_stats(urls)
    ensure_dir(args.stats_out.parent)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("Saved %d Kubernetes docs URLs to %s", len(urls), args.out_file)
    logger.info("Saved URL list stats to %s", args.stats_out)


if __name__ == "__main__":
    main()

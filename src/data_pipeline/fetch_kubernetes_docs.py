from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm

from src.utils.io_utils import ensure_dir, write_jsonl
from src.utils.logging_utils import setup_logging


def load_urls(urls_file: Path) -> list[str]:
    return [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def filename_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/") or "docs"
    path = path.replace("/", "__")
    path = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path).strip("_")
    return path.lower() + ".html"


def page_id_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path == "docs":
        return "kubernetes_docs_home"
    if path.startswith("docs/"):
        path = path[len("docs/") :]
    path = path.replace("/", "_")
    path = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path).strip("_")
    return path.lower() or "kubernetes_docs_home"


def section_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path == "docs":
        return "root"
    if path.startswith("docs/"):
        path = path[len("docs/") :]
    if not path:
        return "root"
    return path.split("/", 1)[0]


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=60) as response:
            return response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        if exc.code in (301, 302, 307, 308):
            redirect_to = exc.headers.get("Location")
            if not redirect_to:
                raise
            req2 = Request(redirect_to, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req2, timeout=60) as response2:
                return response2.read().decode("utf-8", errors="ignore")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kubernetes docs HTML pages from kubernetes.io.")
    parser.add_argument("--urls-file", type=Path, default=Path("data/raw/kubernetes_urls.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/kubernetes_html"))
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/raw/kubernetes_html_manifest.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=0, help="Optional limit for smoke tests (0 = all).")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download pages even if the target HTML file already exists.",
    )
    args = parser.parse_args()

    logger = setup_logging("fetch_kubernetes_docs")
    urls = load_urls(args.urls_file)
    if args.max_pages > 0:
        urls = urls[: args.max_pages]

    out_dir = ensure_dir(args.out_dir)
    manifest_rows: list[dict] = []

    pbar = tqdm(urls, desc="fetch_kubernetes_docs", unit="page", disable=not sys.stderr.isatty())
    for url in pbar:
        filename = filename_from_url(url)
        target = out_dir / filename
        try:
            if target.exists() and not args.overwrite:
                html = target.read_text(encoding="utf-8", errors="ignore")
            else:
                html = fetch_html(url)
                target.write_text(html, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            time.sleep(args.sleep)
            continue
        manifest_rows.append(
            {
                "id": page_id_from_url(url),
                "url": url,
                "path": urlparse(url).path,
                "section": section_from_url(url),
                "html_file": filename,
            }
        )
        time.sleep(args.sleep)

    write_jsonl(args.manifest_out, manifest_rows)
    logger.info("Saved %d Kubernetes HTML pages to %s", len(manifest_rows), out_dir)
    logger.info("Saved HTML manifest to %s", args.manifest_out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


DEFAULT_URLS = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
]


def slugify_url(url: str) -> str:
    path = urlparse(url).path.strip("/") or "index"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)
    return slug.lower() + ".html"


def fetch(url: str, out_dir: Path, logger) -> Path:
    filename = slugify_url(url)
    target = out_dir / filename
    with urlopen(url) as response:
        html = response.read().decode("utf-8", errors="ignore")
    target.write_text(html, encoding="utf-8")
    logger.info("Saved %s", target)
    return target


def load_urls(urls_file: Path | None) -> list[str]:
    if urls_file and urls_file.exists():
        urls = [line.strip() for line in urls_file.read_text().splitlines() if line.strip()]
        return urls
    return DEFAULT_URLS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FastAPI docs HTML pages.")
    parser.add_argument("--urls-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/fastapi_html"))
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    logger = setup_logging("fetch_fastapi_docs")
    out_dir = ensure_dir(args.out_dir)
    urls = load_urls(args.urls_file)[: args.max_pages]
    pbar = tqdm(urls, desc="fetch_fastapi_docs", unit="page", disable=not sys.stderr.isatty())
    for url in pbar:
        fetch(url, out_dir, logger)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

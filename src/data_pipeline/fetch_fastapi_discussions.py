from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.request import Request, urlopen

from src.utils.io_utils import write_jsonl
from src.utils.logging_utils import setup_logging


BASE_URL = "https://github.com/fastapi/fastapi/discussions/categories/questions"
QUERY = "category%3AQuestions+is%3Aanswered"


def fetch_html(page: int) -> str:
    url = f"{BASE_URL}?discussions_q={QUERY}&page={page}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_page(page_html: str) -> list[dict]:
    out: list[dict] = []
    seen_numbers: set[int] = set()
    for href, raw_title in re.findall(
        r'<a[^>]+href="(/fastapi/fastapi/discussions/\d+)"[^>]*>(.*?)</a>',
        page_html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        m = re.search(r"/fastapi/fastapi/discussions/(\d+)", href)
        if not m:
            continue
        number = int(m.group(1))
        if number in seen_numbers:
            continue
        title = re.sub(r"<[^>]+>", " ", raw_title)
        title = html.unescape(" ".join(title.split()))
        # Skip vote counters / bare numeric anchors.
        if not title or title.isdigit():
            continue
        seen_numbers.add(number)
        out.append(
            {
                "discussion_number": number,
                "title": title,
                "url": f"https://github.com/fastapi/fastapi/discussions/{number}",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch answered FastAPI Discussions questions.")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--out-file", type=Path, default=Path("data/raw/github/fastapi_questions_answered.jsonl"))
    args = ap.parse_args()

    logger = setup_logging("fetch_fastapi_discussions")
    rows: list[dict] = []
    seen_numbers: set[int] = set()

    for page in range(1, args.max_pages + 1):
        html = fetch_html(page)
        page_rows = parse_page(html)
        page_new = 0
        for row in page_rows:
            n = row["discussion_number"]
            if n in seen_numbers:
                continue
            seen_numbers.add(n)
            rows.append(row)
            page_new += 1
        logger.info("Page %d: found=%d new=%d", page, len(page_rows), page_new)
        if page_new == 0:
            break

    rows = sorted(rows, key=lambda r: r["discussion_number"], reverse=True)
    write_jsonl(args.out_file, rows)
    logger.info("Saved %d discussions to %s", len(rows), args.out_file)


if __name__ == "__main__":
    main()

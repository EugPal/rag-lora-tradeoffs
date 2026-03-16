from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import setup_logging


SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
CORE_SECTIONS = ("/tutorial/", "/reference/", "/advanced/", "/deployment/", "/how-to/")
SKIP_EXTENSIONS = (".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".xml", ".txt", ".pdf", ".zip")
EXCLUDED_PATH_PREFIXES = ("/release-notes",)


def load_sitemap_urls(sitemap_url: str) -> list[str]:
    req = Request(sitemap_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        data = response.read()
    root = ET.fromstring(data)
    urls: list[str] = []
    for elem in root.findall(".//sm:url/sm:loc", SITEMAP_NS):
        if elem.text:
            urls.append(elem.text.strip())
    return urls


def classify_path(path: str) -> str:
    clean = path.strip("/")
    if not clean:
        return "root"
    first = clean.split("/", 1)[0]
    if first in {"tutorial", "reference", "advanced", "deployment", "how-to"}:
        return first
    if first in {"docs", "release-notes", "learn"}:
        return first
    if len(first) in (2, 5) and "-" in first:
        # e.g. zh-hant, pt-br (locale-like prefixes)
        return "locale_prefixed"
    if len(first) == 2:
        # e.g. en/es/ru (locale-like prefixes)
        return "locale_prefixed"
    if first.startswith("v") and first[1:].replace(".", "").isdigit():
        return "versioned"
    return "other"


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean_path = parsed.path or "/"
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", "", ""))


def load_urls_from_html_dir(html_dir: Path, base_url: str = "https://fastapi.tiangolo.com") -> list[str]:
    if not html_dir.exists():
        return []
    urls: list[str] = []
    for html_file in sorted(html_dir.glob("*.html")):
        raw = html_file.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if not href or href.startswith(("mailto:", "javascript:")):
                continue
            abs_url = urljoin(base_url, href)
            parsed = urlparse(abs_url)
            if parsed.netloc != "fastapi.tiangolo.com":
                continue
            if parsed.path.lower().endswith(SKIP_EXTENSIONS):
                continue
            urls.append(canonicalize_url(abs_url))
    return urls


def expand_localized_urls(base_urls: list[str], locales: list[str]) -> list[str]:
    expanded: list[str] = []
    for url in base_urls:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path == "/":
            continue
        # Skip URLs already locale-prefixed.
        first = path.strip("/").split("/", 1)[0]
        if len(first) in (2, 5):
            continue
        for locale in locales:
            locale = locale.strip().lower()
            if not locale:
                continue
            localized_path = f"/{locale}{path}"
            expanded.append(
                urlunparse((parsed.scheme, parsed.netloc, localized_path, "", "", ""))
            )
    return expanded


def keep_url(url: str, scope: str) -> bool:
    if "/_llm-test/" in url:
        return False
    parsed = urlparse(url)
    if any(parsed.path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if scope == "all":
        return True
    if scope == "core":
        return url.rstrip("/") == "https://fastapi.tiangolo.com" or any(s in url for s in CORE_SECTIONS)
    if scope == "core-plus":
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if not path:
            return True
        if any(s in url for s in CORE_SECTIONS):
            return True
        # Allow locale/version/docs-like prefixes when present in sitemap.
        first = path.strip("/").split("/", 1)[0]
        if first in {"docs", "release-notes", "learn"}:
            return True
        if len(first) in (2, 5) and ("-" in first or len(first) == 2):
            return True
        if first.startswith("v") and first[1:].replace(".", "").isdigit():
            return True
        return False
    raise ValueError(f"Unknown scope: {scope}")


def build_stats(urls: list[str]) -> dict:
    by_prefix: dict[str, int] = {}
    for url in urls:
        path = urlparse(url).path
        label = classify_path(path)
        by_prefix[label] = by_prefix.get(label, 0) + 1
    return {"total_urls": len(urls), "by_prefix": dict(sorted(by_prefix.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FastAPI URL list from sitemap.")
    parser.add_argument("--sitemap-url", type=str, default=SITEMAP_URL)
    parser.add_argument("--out-file", type=Path, default=Path("data/raw/fastapi_urls.txt"))
    parser.add_argument(
        "--scope",
        type=str,
        choices=["core", "core-plus", "all"],
        default="all",
        help="URL selection scope: core, core-plus (core + locale/version/docs-like paths), or all.",
    )
    parser.add_argument("--stats-out", type=Path, default=Path("data/processed/url_list_stats.json"))
    parser.add_argument(
        "--discover-from-html-dir",
        type=Path,
        default=None,
        help="Optional local HTML dir to discover extra internal FastAPI links.",
    )
    parser.add_argument(
        "--locales",
        type=str,
        default="",
        help="Optional comma-separated locale prefixes to expand, e.g. 'es,fr,ru,pt,ja'.",
    )
    parser.add_argument("--max-urls", type=int, default=0, help="Optional max number of URLs (0 = all).")
    args = parser.parse_args()

    logger = setup_logging("build_fastapi_url_list")
    urls = load_sitemap_urls(args.sitemap_url)
    if args.discover_from_html_dir:
        discovered = load_urls_from_html_dir(args.discover_from_html_dir)
        logger.info(
            "Discovered %d internal links from %s",
            len(discovered),
            args.discover_from_html_dir,
        )
        urls.extend(discovered)
    locale_list = [p.strip() for p in args.locales.split(",") if p.strip()]
    if locale_list:
        localized = expand_localized_urls(urls, locale_list)
        logger.info("Expanded %d localized URLs for locales=%s", len(localized), locale_list)
        urls.extend(localized)
    urls = [u for u in urls if keep_url(u, scope=args.scope)]
    urls = [canonicalize_url(u) for u in urls]

    # stable de-dup preserve order
    dedup: list[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    urls = dedup

    if args.max_urls > 0:
        urls = urls[: args.max_urls]

    ensure_dir(args.out_file.parent)
    args.out_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
    stats = build_stats(urls)
    ensure_dir(args.stats_out.parent)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d URLs to %s (scope=%s)", len(urls), args.out_file, args.scope)
    logger.info("Saved URL stats to %s", args.stats_out)


if __name__ == "__main__":
    main()

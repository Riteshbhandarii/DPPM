"""Utility helpers for the DPPM crawler."""

import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urljoin, unquote

from bs4 import BeautifulSoup


FALLBACK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _fetch_with_http_fallback(url, timeout=60):
    req = Request(url, headers=FALLBACK_HEADERS)
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            html = resp.read().decode("utf-8", errors="replace")
            print(f"[fetch-fallback] {status or 200} {url}")
            return html
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[fetch-fallback] {exc.code} {url}")
        return body
    except URLError as exc:
        print(f"[fetch-fallback] error {url}: {exc}")
        return None


def fetch_page(page, url, delay_seconds):
    """
    Fetches and renders a JS-heavy page using a shared Playwright page.
    """
    response = None
    html_content = None
    try:
        # "networkidle" is brittle on pages with chat/analytics widgets that keep
        # requests open; DOMContentLoaded is enough because the target pages are SSR.
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("body", timeout=10000)
        html_content = page.content()
        if response is not None:
            try:
                print(f"[fetch] {response.status} {getattr(page, 'url', url)}")
            except Exception:
                pass
    except Exception as exc:
        # If navigation times out after partial render, try to salvage the current DOM.
        # This avoids losing usable pages when third-party widgets keep loading.
        try:
            html_content = page.content()
            if html_content and "<html" in html_content.lower():
                print(f"Warning loading {url}: {exc} (using partial page content)")
            else:
                print(f"Error loading {url}: {exc}")
        except Exception:
            print(f"Error loading {url}: {exc}")

    # Playwright may get 403 (headless/automation fingerprint) even when plain HTTP
    # with browser-like headers works from the same host/IP.
    status = None
    try:
        status = response.status if response is not None else None
    except Exception:
        status = None

    if not html_content or status == 403 or "403 Forbidden" in html_content[:500]:
        fallback_html = _fetch_with_http_fallback(url)
        if fallback_html and "<html" in fallback_html.lower():
            html_content = fallback_html

    if not html_content:
        return None

    time.sleep(delay_seconds)
    return BeautifulSoup(html_content, "lxml")


def debug_dump_page(page, soup, label):
    """
    Print a compact debug summary and save the current HTML for inspection.
    """
    try:
        title_tag = soup.find("title") if soup else None
        title = title_tag.get_text(strip=True) if title_tag else "(no title)"
        link_count = len(soup.find_all("a", href=True)) if soup else 0
        text_preview = (soup.get_text(" ", strip=True)[:300] if soup else "").strip()
        current_url = getattr(page, "url", "")

        print(f"[debug] {label}")
        if current_url:
            print(f"[debug] current_url: {current_url}")
        print(f"[debug] title: {title}")
        print(f"[debug] link_count: {link_count}")
        if text_preview:
            print(f"[debug] text_preview: {text_preview}")

        crawler_root = Path(__file__).resolve().parents[1]
        debug_dir = crawler_root / "crawler_datasets" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^a-zA-Z0-9._-]+", "_", label).strip("_")[:80] or "page"
        html_path = debug_dir / f"{safe_label}.html"
        html = str(soup) if soup is not None else page.content()
        html_path.write_text(html, encoding="utf-8")
        print(f"[debug] html_saved: {html_path}")
    except Exception as exc:
        print(f"[debug] failed to collect debug info: {exc}")


def clean_part_name(name, brand, model):
    """
    Removes redundant brand, model, and year info from part names.
    """
    if not name:
        return "Unknown Part"

    name = re.sub(rf"{re.escape(brand)}", "", name, flags=re.IGNORECASE)

    base_model = model.split(",")[0].split("-")[0]
    name = re.sub(rf"{re.escape(base_model)}", "", name, flags=re.IGNORECASE)

    name = re.sub(r"\s*\(\d{4}.*?(?:\)|$)", "", name)
    name = re.sub(r"\(\d{4}\s*-\s*\d{4}\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    if name.lower().endswith(" to"):
        name = name[:-3].rstrip()

    return name or "Unknown Part"


def extract_product_id(url):
    match = re.search(r"ID-(\d+)", url)
    return match.group(1) if match else None


def get_product_links_from_listing(soup):
    product_dict = {}

    for link in soup.find_all("a", href=re.compile(r"ID-\d+")):
        url = urljoin("https://www.varaosahaku.fi/", link.get("href", ""))
        product_id = extract_product_id(url)

        if product_id and product_id not in product_dict:
            product_dict[product_id] = url

    return product_dict


def parse_price(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return None


def dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


def normalize_url_for_match(url):
    return unquote(url).lower()

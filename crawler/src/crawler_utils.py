"""Utility helpers for the DPPM crawler."""

import re
import time
from urllib.parse import urljoin, unquote

from bs4 import BeautifulSoup


def fetch_page(page, url, delay_seconds):
    """
    Fetches and renders a JS-heavy page using a shared Playwright page.
    """
    try:
        # "networkidle" is brittle on pages with chat/analytics widgets that keep
        # requests open; DOMContentLoaded is enough because the target pages are SSR.
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("body", timeout=10000)
        html_content = page.content()
    except Exception as exc:
        # If navigation times out after partial render, try to salvage the current DOM.
        # This avoids losing usable pages when third-party widgets keep loading.
        try:
            html_content = page.content()
            if html_content and "<html" in html_content.lower():
                print(f"Warning loading {url}: {exc} (using partial page content)")
            else:
                print(f"Error loading {url}: {exc}")
                return None
        except Exception:
            print(f"Error loading {url}: {exc}")
            return None

    time.sleep(delay_seconds)
    return BeautifulSoup(html_content, "lxml")


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

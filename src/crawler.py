"""
DPPM Car Parts Data Collector
Scrapes used car parts pricing data from Varaosahaku.fi for thesis research.

Collects: product_id, part_name, price, quality_grade, year, oem_number,
engine_code, mileage, brand, model, category, subcategory, scrape_date.

Author: Ritesh Bhandari (ritesh.bhandari@edu.turkuamk.fi)
Institution: Turku University of Applied Sciences
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, unquote

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# Configuration
USER_AGENT = "ThesisScraper/1.0 (ritesh.bhandari@edu.turkuamk.fi; academic research)"
DELAY_SECONDS = 1.0
OUTPUT_CSV = None
KEEP_CATEGORIES = [
    "Brakes",
    "Engine",
    "Airbag",
    "Gear box / Drive axle / Middle axle",
    "Fuel",
    "Electric / Transmitter / Databox / Sensor",
    "Vehicle exterior / Suspension",
]
MAX_PARTS_PER_SUBCATEGORY = 60  # Limit parts per subcategory for reasonable sample size

FINAL_COLUMNS = [
    "product_id",
    "part_name",
    "price",
    "quality_grade",
    "year",
    "oem_number",
    "engine_code",
    "mileage",
    "brand",
    "model",
    "category",
    "subcategory",
    "scrape_date",
]


def fetch_page(page, url):
    """
    Fetches and renders JavaScript-heavy page using a shared Playwright page.
    Waits for network to be idle before extracting HTML content.
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
        html_content = page.content()
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(html_content, "lxml")


def clean_part_name(name, brand, model):
    """
    Removes redundant brand, model, and year information from part names.
    Returns cleaned part name or 'Unknown Part' if cleaning fails.
    """
    if not name:
        return "Unknown Part"

    name = re.sub(rf"{re.escape(brand)}", "", name, flags=re.IGNORECASE)

    base_model = model.split(",")[0].split("-")[0]
    name = re.sub(rf"{re.escape(base_model)}", "", name, flags=re.IGNORECASE)

    # Remove year ranges
    name = re.sub(r"\s*\(\d{4}.*?(?:\)|$)", "", name)
    name = re.sub(r"\(\d{4}\s*-\s*\d{4}\)", "", name)

    # Clean up extra whitespace
    name = re.sub(r"\s+", " ", name)

    name = name.strip()

    # Remove trailing "to" (often leftover from EN translation)
    if name.lower().endswith(" to"):
        name = name[:-3].rstrip()

    return name or "Unknown Part"


def extract_product_id(url):
    """
    Extracts unique product ID from URL.
    Returns product ID string or None if not found.
    """
    match = re.search(r"ID-(\d+)", url)
    return match.group(1) if match else None


def get_product_links_from_listing(soup):
    """
    Extracts all unique product URLs from a listing page.
    Identifies product links by their ID pattern in the URL.
    """
    product_dict = {}

    for link in soup.find_all("a", href=re.compile(r"ID-\d+")):
        url = urljoin("https://www.varaosahaku.fi/", link.get("href", ""))
        product_id = extract_product_id(url)

        if product_id and product_id not in product_dict:
            product_dict[product_id] = url

    return product_dict


def parse_price(value):
    """
    Converts extracted price to float safely.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    try:
        # JSON-LD price often numeric/string already; just normalize commas
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return None


def dedupe_preserve_order(items):
    """
    Deduplicate list while preserving order.
    """
    return list(dict.fromkeys(items))


def _normalize_url_for_match(url):
    """
    Lowercase + URL-decode for reliable substring checks.
    """
    return unquote(url).lower()


def find_category_links(main_page, base_url, base_domain, brand, model):
    """
    Find category links on the brand/model landing page.
    Uses multiple heuristics to handle site changes.
    """
    all_links = main_page.find_all("a", href=True)
    category_links = []

    brand_l = brand.lower()
    model_l = model.lower()
    base_url_norm = _normalize_url_for_match(base_url).rstrip("/")

    blacklist = {
        brand,
        "Search unattached part",
        "Registration number search",
    }

    car_parts_candidates = []
    for link in all_links:
        href = link.get("href", "")
        link_text = link.get_text(strip=True)
        if not link_text or link_text in blacklist:
            continue

        full_url = urljoin(base_domain + "/", href)
        full_url_norm = _normalize_url_for_match(full_url).rstrip("/")
        if "/pb/search/car-parts" in full_url_norm:
            car_parts_candidates.append((link_text, href, full_url))

        # Heuristic 1: historical /sNN pattern
        if "/s" in href and href.split("/s")[-1].isdigit():
            category_links.append((link_text, href))
            continue

        # Heuristic 2: brand+model scoped car-parts links (newer structure)
        is_car_parts = "/pb/search/car-parts" in full_url_norm
        has_brand_model = f"/{brand_l}/" in full_url_norm and f"/{model_l}" in full_url_norm
        is_base_url = full_url_norm == base_url_norm

        if is_car_parts and has_brand_model and not is_base_url:
            category_links.append((link_text, href))

    return dedupe_preserve_order(category_links), all_links

def scrape_brand_model(page, brand, model):
    """
    Main scraping function that crawls all categories and subcategories for a specific car brand and model.
    Writes rows incrementally to CSV (crash-safe).
    """

    BASE_URL = "https://www.varaosahaku.fi/en-se"

    base_urls = [
        f"{BASE_URL}/pb/Search/Car-parts/s19/{brand}/{model}",
        f"{BASE_URL}/pb/Search/Car-parts/s1/{brand}/{model}",
    ]

    main_page = None
    base_url = None

    for url in base_urls:
        print(f"Trying URL: {url}")
        main_page = fetch_page(page, url)
        if main_page and len(main_page.find_all("a", href=True)) > 10:
            base_url = url
            print(f"✓ Successfully loaded: {url}\n")
            break

    if not main_page or not base_url:
        print("ERROR: Could not load any base URL")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    all_parts_data = []
    csv_exists = Path(OUTPUT_CSV).exists()
    scraped_product_ids = set()

    def append_row(part_data):
        nonlocal csv_exists
        df_row = pd.DataFrame([part_data]).reindex(columns=FINAL_COLUMNS)
        df_row.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not csv_exists,
            index=False
        )
        csv_exists = True

    # Find category links
    category_links, all_links = find_category_links(
        main_page,
        base_url,
        BASE_URL,
        brand,
        model,
    )

    if KEEP_CATEGORIES:
        keep_normalized = {c.strip().lower() for c in KEEP_CATEGORIES}
        category_links = [
            (name, href)
            for (name, href) in category_links
            if name.strip().lower() in keep_normalized
        ]

    if not category_links:
        print("\nERROR: No category links found!")
        print("Available categories on page:")
        for link in all_links:
            text = link.get_text(strip=True)
            if text:
                print(f"  - {text}")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    print(f"Found {len(category_links)} categories to scrape\n")

    scrape_date = datetime.now(ZoneInfo("Europe/Helsinki")).date().isoformat()

    for category_name, category_href in category_links:
        category_url = urljoin(BASE_URL + "/", category_href)
        print(f"\n{'-'*40}")
        print(f"Category: {category_name}")
        print(f"{'-'*40}")

        category_page = fetch_page(page, category_url)
        if not category_page:
            print(" ✗ Failed to load category page")
            continue

        direct_products = get_product_links_from_listing(category_page)

        if direct_products:
            print(f"Found {len(direct_products)} products directly on category page")

            parts_scraped = 0
            for product_id, product_url in direct_products.items():
                if product_id in scraped_product_ids:
                    continue
                if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                    break

                scraped_product_ids.add(product_id)

                part_data = scrape_product_page(
                    page,
                    product_url,
                    brand,
                    model,
                    category_name,
                    "Main",
                    product_id,
                    scrape_date,
                )
                if part_data:
                    all_parts_data.append(part_data)
                    append_row(part_data)
                    print(f"  [{len(all_parts_data)}] {part_data['part_name'][:50]}")
                    parts_scraped += 1

        else:
            # Navigate to subcategories
            subcategory_links = []
            for link in category_page.find_all("a", href=True):
                href = link.get("href", "")
                link_text = link.get_text(strip=True)
                full = urljoin(BASE_URL + "/", href)
                full_norm = _normalize_url_for_match(full)
                brand_l = brand.lower()
                model_l = model.lower()

                if (
                    link_text
                    and link_text != category_name
                    and "/pb/search/car-parts" in full_norm
                    and f"/{brand_l}/" in full_norm
                    and f"/{model_l}" in full_norm
                ):
                    subcategory_links.append((link_text, href))

            subcategory_links = dedupe_preserve_order(subcategory_links)

            if not subcategory_links:
                continue

            for subcategory_name, subcategory_href in subcategory_links:
                subcategory_url = urljoin(BASE_URL + "/", subcategory_href)
                page_num = 1
                parts_scraped = 0

                while True:
                    if "?" in subcategory_url:
                        listing_page_url = f"{subcategory_url}&page={page_num}"
                    else:
                        listing_page_url = f"{subcategory_url}?page={page_num}"

                    listing_page = fetch_page(page, listing_page_url)
                    if not listing_page:
                        break

                    product_dict = get_product_links_from_listing(listing_page)
                    if not product_dict:
                        break

                    for product_id, product_url in product_dict.items():
                        if product_id in scraped_product_ids:
                            continue
                        if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                            break

                        scraped_product_ids.add(product_id)

                        part_data = scrape_product_page(
                            page,
                            product_url,
                            brand,
                            model,
                            category_name,
                            subcategory_name,
                            product_id,
                            scrape_date,
                        )
                        if part_data:
                            all_parts_data.append(part_data)
                            append_row(part_data)
                            print(f"  [{len(all_parts_data)}] {part_data['part_name'][:50]}")
                            parts_scraped += 1

                    if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                        break

                    page_num += 1

    df_final = pd.DataFrame(all_parts_data).reindex(columns=FINAL_COLUMNS)
    if not df_final.empty and "product_id" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["product_id"], keep="first")

    return df_final


def scrape_product_page(page, product_url, brand, model, category_name, subcategory_name, product_id, scrape_date):
    """
    Scrapes a single product page and extracts all relevant data.
    """
    soup = fetch_page(page, product_url)
    if not soup:
        return None

    page_text = soup.get_text()

    # Extract part name (prefer H1, then title)
    part_name = None

    h1_tag = soup.find("h1")
    if h1_tag:
        h1_text = h1_tag.get_text(strip=True)
        if h1_text:
            part_name = h1_text

    title_tag = soup.find("title")
    if title_tag and not part_name:
        title_text = title_tag.get_text()
        if "|" in title_text:
            part_name = title_text.split("|")[0].strip()
        else:
            part_name = title_text.split("-")[0].strip()

    part_name = clean_part_name(part_name, brand, model)

    # Extract price from JSON-LD structured data
    json_ld = soup.find("script", type="application/ld+json")
    price = None
    if json_ld:
        try:
            data = json.loads(json_ld.string)
            price = data.get("offers", {}).get("price")
        except Exception:
            price = None
    price = parse_price(price)

    # Extract quality grade (FI + EN)
    quality_patterns = [
        r"Laatu:\s*([A-C][1-3]?)",
        r"Quality(?:\s*grade)?\s*:?\s*([A-C][1-3]?)",
        r"Condition\s*:?\s*([A-C][1-3]?)",
    ]
    quality_match = None
    for pattern in quality_patterns:
        quality_match = re.search(pattern, page_text, re.IGNORECASE)
        if quality_match:
            break
    quality_grade = quality_match.group(1) if quality_match else None

    # Extract year range
    year_match = re.search(r"\((\d{4})\s*-\s*(\d{4})\)", page_text)
    year = f"{year_match.group(1)}-{year_match.group(2)}" if year_match else None

    # Extract OEM part number
    oem_number = None
    oem_patterns = [
        r"(?:alkuperäinen|oem)\s*(?:nro|numero|num)\s*:?\s*([A-Z0-9\-/]{6,20})",
        r"(?:original|oem)\s*(?:no|number|num)\s*:?\s*([A-Z0-9\-/]{6,20})",
        r"oem[:\-]?\s*([A-Z0-9\-/]{6,20})",
        r"pn[:\-]?\s*([A-Z0-9\-/]{6,20})",
        r"([A-Z]{2,4}\d{4,8}[A-Z\-]?)",
    ]
    for pattern in oem_patterns:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            raw_oem = m.group(1)
            cleaned = re.sub(r"[^\w\-/]", "", raw_oem.upper())
            if 6 <= len(cleaned) <= 20:
                oem_number = cleaned
                break

    # Extract engine code
    engine_code = None
    engine_patterns = [
        r"Moottorin\s+koodi\s*:?\s*([A-Z0-9-]+)",
        r"moottorik[oö]{2}di\s*:?\s*([A-Z0-9-]+)",
        r"Moottori\s*:?\s*([A-Z]{1,2}\d{1,2}[A-Z]{0,3})",
        r"Engine\s+code\s*:?\s*([A-Z0-9-]+)",
        r"Engine\s*:?\s*code\s*([A-Z0-9-]+)",
    ]
    for pattern in engine_patterns:
        engine_match = re.search(pattern, page_text, re.IGNORECASE)
        if engine_match:
            raw_engine = engine_match.group(1).strip()
            if (
                3 <= len(raw_engine) <= 8
                and any(c.isalpha() for c in raw_engine)
                and any(c.isdigit() for c in raw_engine)
                and not raw_engine.startswith(("19", "20"))
            ):
                engine_code = raw_engine.replace("-", "").upper()
                break

    # Extract mileage
    mileage = None
    mileage_patterns = [
        r"lukema(?:t)?\s*\(km\)\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})",
        r"lukema(?:t)?\s*([0-9][0-9\s\.,]{0,12})\s*km?",
        r"(\d{1,3}(?:\s|\.)?\d{3})\s*km",
        r"(\d{1,7})\s*km",
        r"matkamittari\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})",
        r"Mileage\s*\(km\)\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})",
        r"Mileage\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})\s*km?",
        r"Odometer\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})\s*km?",
    ]
    for pattern in mileage_patterns:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            digits_only = re.sub(r"[^\d]", "", raw)
            if digits_only:
                mileage = int(digits_only)
                break

    return {
        "product_id": product_id,
        "part_name": part_name,
        "price": price,
        "quality_grade": quality_grade,
        "year": year,
        "oem_number": oem_number,
        "engine_code": engine_code,
        "mileage": mileage,
        "brand": brand,
        "model": model,
        "category": category_name,
        "subcategory": subcategory_name,
        "scrape_date": scrape_date,
    }


if __name__ == "__main__":
    print("-" * 10)
    print("DPPM Car Parts Data Collector")
    print("-" * 10)
    print(f"Max parts per subcategory: {MAX_PARTS_PER_SUBCATEGORY}")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    base_output = f"dppm_{args.brand.lower()}_{args.model.lower().replace(',', '_').replace('-', '_').replace(' ', '_')}"
    scrape_date = datetime.now(ZoneInfo("Europe/Helsinki")).date().isoformat()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "datasets" / "new"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = output_dir / f"{base_output}_{scrape_date}.csv"
    if candidate.exists():
        suffix = 2
        while (output_dir / f"{base_output}_{scrape_date}_v{suffix}.csv").exists():
            suffix += 1
        candidate = output_dir / f"{base_output}_{scrape_date}_v{suffix}.csv"

    OUTPUT_CSV = str(candidate)

    # Playwright started (stability + speed)
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)

        results = scrape_brand_model(page, args.brand, args.model)

        browser.close()

    print(f"\n{'-'*10}")
    print("Scraping Completed!")
    print(f"{'='*10}  ")
    print(f"\nTotal parts scraped: {len(results)}")
    print(f"Output saved to: {OUTPUT_CSV}")

    if len(results) == 0:
        print("\nWARNING: No parts were scraped!")
    else:
        print(f"\n{'-'*10}")
        print("Data Quality Summary:")
        print(f"{'-'*10}")
        print(f"Parts with prices:        {results['price'].notna().sum():>6} / {len(results)}")
        print(f"Parts with OEM numbers:   {results['oem_number'].notna().sum():>6} / {len(results)}")
        print(f"Parts with engine codes:  {results['engine_code'].notna().sum():>6} / {len(results)}")
        print(f"Parts with mileage:       {results['mileage'].notna().sum():>6} / {len(results)}")
        print(f"Parts with quality grade: {results['quality_grade'].notna().sum():>6} / {len(results)}")

        print("\nQuality Grades Distribution:")
        for grade, count in results["quality_grade"].value_counts().items():
            print(f"  {grade}: {count}")

        if results["price"].notna().any():
            print("\nPrice Statistics:")
            print(f"  Min:    €{results['price'].min():.2f}")
            print(f"  Max:    €{results['price'].max():.2f}")
            print(f"  Mean:   €{results['price'].mean():.2f}")
            print(f"  Median: €{results['price'].median():.2f}")

    print(f"\n{'-'*10}")

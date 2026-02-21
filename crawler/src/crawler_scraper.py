"""Scraping orchestration."""

from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd

from .crawler_categories import filter_categories, find_category_links
from .crawler_config import BASE_URL, DELAY_SECONDS, FINAL_COLUMNS, KEEP_CATEGORIES, MAX_PARTS_PER_SUBCATEGORY, TIMEZONE
from .crawler_parser import parse_product_page
from .crawler_utils import fetch_page, get_product_links_from_listing, normalize_url_for_match, dedupe_preserve_order


def scrape_brand_model(page, brand, model, output_csv):
    base_urls = [
        f"{BASE_URL}/pb/Search/Car-parts/s19/{brand}/{model}",
        f"{BASE_URL}/pb/Search/Car-parts/s1/{brand}/{model}",
    ]

    main_page = None
    base_url = None
    base_category_links = []
    base_all_links = []

    for url in base_urls:
        print(f"Trying URL: {url}")
        main_page = fetch_page(page, url, DELAY_SECONDS)
        if not main_page:
            print(" ✗ Failed to load URL")
            continue

        category_links, all_links = find_category_links(
            main_page,
            url,
            BASE_URL,
            brand,
            model,
        )

        if category_links:
            base_url = url
            base_category_links = category_links
            base_all_links = all_links
            print(f"✓ Successfully loaded: {url}\n")
            break

        title_tag = main_page.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        link_count = len(all_links)
        print("WARNING: Base URL loaded but no category links were detected")
        if title:
            print(f"Page title: {title}")
        print(f"Link count: {link_count}")
        current_url = getattr(page, "url", "")
        if current_url and current_url != url:
            print(f"Redirected to: {current_url}")

    if not main_page or not base_url:
        print("ERROR: Could not load any base URL")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    all_parts_data = []
    csv_exists = Path(output_csv).exists()
    scraped_product_ids = set()

    def flush_rows(rows):
        nonlocal csv_exists
        if not rows:
            return
        if not csv_exists:
            csv_exists = Path(output_csv).exists()
        df_rows = pd.DataFrame(rows).reindex(columns=FINAL_COLUMNS)
        df_rows.to_csv(
            output_csv,
            mode="a",
            header=not csv_exists,
            index=False,
        )
        csv_exists = True

    category_links = filter_categories(base_category_links, KEEP_CATEGORIES)
    all_links = base_all_links or main_page.find_all("a", href=True)

    if not category_links:
        print("\nERROR: No category links found!")
        print("Available categories on page:")
        for link in all_links:
            text = link.get_text(strip=True)
            if text:
                print(f"  - {text}")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    print(f"Found {len(category_links)} categories to scrape\n")

    now = datetime.now(ZoneInfo(TIMEZONE))
    scrape_date = now.date().isoformat()
    scrape_timestamp = now.isoformat(timespec="seconds")

    for category_name, category_href in category_links:
        category_url = urljoin(BASE_URL + "/", category_href)
        print(f"\n{'-'*40}")
        print(f"Category: {category_name}")
        print(f"{'-'*40}")

        category_page = fetch_page(page, category_url, DELAY_SECONDS)
        if not category_page:
            print(" ✗ Failed to load category page")
            continue

        direct_products = get_product_links_from_listing(category_page)

        if direct_products:
            print(f"Found {len(direct_products)} products directly on category page")

            parts_scraped = 0
            subcategory_rows = []
            for product_id, product_url in direct_products.items():
                if product_id in scraped_product_ids:
                    continue
                if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                    break

                scraped_product_ids.add(product_id)

                part_data = _scrape_product(
                    page,
                    product_url,
                    brand,
                    model,
                    category_name,
                    "Main",
                    product_id,
                    scrape_date,
                    scrape_timestamp,
                )
                if part_data:
                    all_parts_data.append(part_data)
                    subcategory_rows.append(part_data)
                    print(f"  [{len(all_parts_data)}] {part_data['part_name'][:50]}")
                    parts_scraped += 1
            flush_rows(subcategory_rows)
            subcategory_rows.clear()

        else:
            subcategory_links = []
            for link in category_page.find_all("a", href=True):
                href = link.get("href", "")
                link_text = link.get_text(strip=True)
                full = urljoin(BASE_URL + "/", href)
                full_norm = normalize_url_for_match(full)
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
                subcategory_rows = []

                while True:
                    if "?" in subcategory_url:
                        listing_page_url = f"{subcategory_url}&page={page_num}"
                    else:
                        listing_page_url = f"{subcategory_url}?page={page_num}"

                    listing_page = fetch_page(page, listing_page_url, DELAY_SECONDS)
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

                        part_data = _scrape_product(
                            page,
                            product_url,
                            brand,
                            model,
                            category_name,
                            subcategory_name,
                            product_id,
                            scrape_date,
                            scrape_timestamp,
                        )
                        if part_data:
                            all_parts_data.append(part_data)
                            subcategory_rows.append(part_data)
                            print(f"  [{len(all_parts_data)}] {part_data['part_name'][:50]}")
                            parts_scraped += 1

                    if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                        break

                    page_num += 1
                flush_rows(subcategory_rows)
                subcategory_rows.clear()

    df_final = pd.DataFrame(all_parts_data).reindex(columns=FINAL_COLUMNS)
    if not df_final.empty and "product_id" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["product_id"], keep="first")

    return df_final


def _scrape_product(
    page,
    product_url,
    brand,
    model,
    category_name,
    subcategory_name,
    product_id,
    scrape_date,
    scrape_timestamp,
):
    soup = fetch_page(page, product_url, DELAY_SECONDS)
    if not soup:
        return None

    parsed = parse_product_page(soup, brand, model)

    return {
        "product_id": product_id,
        "part_name": parsed["part_name"],
        "price": parsed["price"],
        "quality_grade": parsed["quality_grade"],
        "year": parsed["year"],
        "oem_number": parsed["oem_number"],
        "engine_code": parsed["engine_code"],
        "mileage": parsed["mileage"],
        "brand": brand,
        "model": model,
        "category": category_name,
        "subcategory": subcategory_name,
        "scrape_date": scrape_date,
        "scrape_timestamp": scrape_timestamp,
    }

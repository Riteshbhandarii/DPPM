"""
DPPM Car Parts Data Collector
Entry point for scraping used car parts pricing data from Varaosahaku.fi.
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright

from .crawler_config import MAX_PARTS_PER_SUBCATEGORY, TIMEZONE, USER_AGENT
from .crawler_scraper import scrape_brand_model


BLOCKED_RESOURCE_TYPES = {"image", "media"}


def _build_output_path(brand, model, scrape_date):
    base_output = f"dppm_{brand.lower()}_{model.lower().replace(',', '_').replace('-', '_').replace(' ', '_')}"
    crawler_root = Path(__file__).resolve().parents[1]
    output_dir = crawler_root / "crawler_datasets" / "new"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = output_dir / f"{base_output}_{scrape_date}.csv"
    if candidate.exists():
        suffix = 2
        while (output_dir / f"{base_output}_{scrape_date}_v{suffix}.csv").exists():
            suffix += 1
        candidate = output_dir / f"{base_output}_{scrape_date}_v{suffix}.csv"

    return str(candidate)


def main():
    print("-" * 30)
    print("DPPM Car Parts Data Collector")
    print("-" * 30)
    print(f"Max parts per subcategory: {MAX_PARTS_PER_SUBCATEGORY}")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    now = datetime.now(ZoneInfo(TIMEZONE))
    scrape_date = now.date().isoformat()
    output_csv = _build_output_path(args.brand, args.model, scrape_date)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in BLOCKED_RESOURCE_TYPES
            else route.continue_(),
        )

        results = scrape_brand_model(page, args.brand, args.model, output_csv)

        browser.close()

    print(f"\n{'-'*30}")
    print("Scraping Completed!")
    print(f"{'-'*30}")
    print(f"\nTotal parts scraped: {len(results)}")
    print(f"Output saved to: {output_csv}")

    if len(results) == 0:
        print("\nWARNING: No parts were scraped!")
    else:
        print(f"\n{'-'*30}")
        print("Data Quality Summary:")
        print(f"{'-'*30}")
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

    print(f"\n{'-'*30}")


if __name__ == "__main__":
    main()

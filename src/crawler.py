"""
DPPM Car Parts Data Collector v1.3 - Fixed Navigation + Sampling
Scrapes used car parts pricing data from Varaosahaku.fi for thesis research.
Collects: part names, prices, quality grades, OEM numbers, engine codes, mileage, and metadata.

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

import pandas as pd
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


# Configuration
USER_AGENT = 'ThesisScraper/1.0 (ritesh.bhandari@edu.turkuamk.fi; academic research)'
DELAY_SECONDS = 1.0
OUTPUT_CSV = None
CATEGORIES_TO_SCRAPE = ['Jarrut', 'Moottori', 'Kori']  # Brakes, Engine, Body
MAX_PARTS_PER_SUBCATEGORY = 30  # Limit parts per subcategory for reasonable sample size


def fetch_page(url):
    """
    Fetches and renders JavaScript-heavy page using Playwright browser automation.
    Waits for network to be idle before extracting HTML content.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        try:
            page.goto(url, wait_until='networkidle', timeout=30000)
            html_content = page.content()
        except Exception as e:
            print(f"Error loading {url}: {e}")
            browser.close()
            return None
        browser.close()
    
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(html_content, 'lxml')


def clean_part_name(name, brand, model):
    """
    Removes redundant brand, model, and year information from part names.
    Returns cleaned part name or 'Unknown Part' if cleaning fails.
    """
    if not name:
        return "Unknown Part"
    
    # Remove brand
    name = re.sub(rf'{brand}', '', name, flags=re.IGNORECASE)
    
    # Remove model - handle complex model names
    base_model = model.split(',')[0].split('-')[0]
    name = re.sub(rf'{base_model}', '', name, flags=re.IGNORECASE)
    
    # Remove year ranges
    name = re.sub(r'\s*\(\d{4}.*?(?:\)|$)', '', name)
    name = re.sub(r'\(\d{4}\s*-\s*\d{4}\)', '', name)
    
    # Clean up extra whitespace
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip() or "Unknown Part"


def extract_product_id(url):
    """
    Extracts unique product ID from URL.
    Returns product ID string or None if not found.
    """
    match = re.search(r'ID-(\d+)', url)
    return match.group(1) if match else None


def get_product_links_from_listing(soup):
    """
    Extracts all unique product URLs from a listing page.
    Identifies product links by their ID pattern in the URL.
    """
    product_dict = {}
    
    for link in soup.find_all('a', href=re.compile(r'ID-\d+')):
        url = urljoin('https://www.varaosahaku.fi/', link.get('href', ''))
        product_id = extract_product_id(url)
        
        if product_id and product_id not in product_dict:
            product_dict[product_id] = url
    
    return product_dict


def scrape_brand_model(brand, model):
    """
    Main scraping function that crawls all categories and subcategories for a specific car brand and model.
    """
    # Try both s1 and s19 URL patterns
    base_urls = [
        f"https://www.varaosahaku.fi/fi-fi/pb/Hae/Autonosat/s19/{brand}/{model}",
        f"https://www.varaosahaku.fi/fi-fi/pb/Hae/Autonosat/s1/{brand}/{model}"
    ]
    
    main_page = None
    base_url = None
    
    # Try each URL pattern until one works
    for url in base_urls:
        print(f"Trying URL: {url}")
        main_page = fetch_page(url)
        if main_page and len(main_page.find_all('a', href=True)) > 10:
            base_url = url
            print(f"✓ Successfully loaded: {url}\n")
            break
    
    if not main_page or not base_url:
        print("ERROR: Could not load any base URL")
        return pd.DataFrame()
    
    all_parts_data = []
    csv_exists = Path(OUTPUT_CSV).exists()

    def append_row(part_data):
        nonlocal csv_exists
        pd.DataFrame([part_data]).to_csv(
            OUTPUT_CSV,
            mode='a',
            header=not csv_exists,
            index=False
        )
        csv_exists = True
    scraped_product_ids = set()
    
    # Get base model name for filtering
    base_model = model.split(',')[0].split('-')[0]
    
    # Find all category links
    all_links = main_page.find_all('a', href=True)
    category_links = []
    
    for link in all_links:
        href = link.get('href', '')
        link_text = link.get_text(strip=True)
        
        # Check if this is a category link (contains model name in URL and is not a navigation link)
        if (base_model in href and 
            link_text and
            link_text not in ['Etusivu', 'Jäsenyritykset', brand, base_model, 'Autonosat', 'Valmistaja']):
            
            # If we have specific categories to scrape, filter them
            if CATEGORIES_TO_SCRAPE:
                if any(cat in link_text for cat in CATEGORIES_TO_SCRAPE):
                    category_links.append((link_text, href))
            else:
                category_links.append((link_text, href))
    
    # Remove duplicates
    category_links = list(set(category_links))
    
    if not category_links:
        print("\nERROR: No category links found!")
        print("Available categories on page:")
        for link in all_links:
            href = link.get('href', '')
            link_text = link.get_text(strip=True)
            if base_model in href and link_text and link_text not in ['Etusivu', 'Jäsenyritykset']:
                print(f"  - {link_text}")
        return pd.DataFrame()
    
    print(f"Found {len(category_links)} categories to scrape\n")

    # Scrape date for this run (Helsinki time)
    scrape_date = datetime.now(ZoneInfo("Europe/Helsinki")).date().isoformat()

    # Navigate through main categories
    for category_name, category_href in category_links:
        category_url = urljoin('https://www.varaosahaku.fi/', category_href)
        print(f"\n{'='*70}")
        print(f"Category: {category_name}")
        print(f"{'='*70}")
        
        category_page = fetch_page(category_url)
        if not category_page:
            print("  ✗ Failed to load category page")
            continue
        
        # Check if there are products directly on this page
        direct_products = get_product_links_from_listing(category_page)
        
        if direct_products:
            # Products are directly on category page
            print(f"  Found {len(direct_products)} products directly on category page")
            
            parts_scraped = 0
            for product_id, product_url in direct_products.items():
                if product_id in scraped_product_ids:
                    continue
                if parts_scraped >= MAX_PARTS_PER_SUBCATEGORY:
                    print(f"  Reached limit of {MAX_PARTS_PER_SUBCATEGORY} parts for this category")
                    break
                    
                scraped_product_ids.add(product_id)
                
                part_data = scrape_product_page(
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
            # Need to navigate to subcategories
            subcategory_links = []
            for link in category_page.find_all('a', href=True):
                href = link.get('href', '')
                link_text = link.get_text(strip=True)
                
                # Filter for subcategory links (contain category URL and model name)
                if (category_url in urljoin('https://www.varaosahaku.fi/', href) and
                    base_model in href and
                    link_text and
                    link_text != category_name and
                    link_text not in ['Etusivu', 'Jäsenyritykset', brand, base_model, 'Kaikki']):
                    subcategory_links.append((link_text, href))
            
            # Remove duplicates
            subcategory_links = list(set(subcategory_links))
            
            if not subcategory_links:
                print(f"  No subcategories found, skipping...")
                continue
            
            print(f"  Found {len(subcategory_links)} subcategories")
            
            # Navigate through subcategories
            for subcategory_name, subcategory_href in subcategory_links:
                subcategory_url = urljoin('https://www.varaosahaku.fi/', subcategory_href)
                print(f"\n  Subcategory: {subcategory_name}")
                
                parts_in_subcategory = 0
                
                # Paginate through all listing pages
                page_num = 1
                while True:
                    if parts_in_subcategory >= MAX_PARTS_PER_SUBCATEGORY:
                        print(f"    Reached limit of {MAX_PARTS_PER_SUBCATEGORY} parts for this subcategory")
                        break
                        
                    if '?' in subcategory_url:
                        listing_page_url = f"{subcategory_url}&page={page_num}"
                    else:
                        listing_page_url = f"{subcategory_url}?page={page_num}"
                    
                    listing_page = fetch_page(listing_page_url)
                    if not listing_page:
                        break
                    
                    product_dict = get_product_links_from_listing(listing_page)
                    
                    if not product_dict:
                        break
                    
                    print(f"    Page {page_num}: Found {len(product_dict)} products")
                    
                    # Scrape each individual product page
                    for product_id, product_url in product_dict.items():
                        if product_id in scraped_product_ids:
                            continue
                        if parts_in_subcategory >= MAX_PARTS_PER_SUBCATEGORY:
                            break
                            
                        scraped_product_ids.add(product_id)
                        
                        part_data = scrape_product_page(
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
                            print(f"    [{len(all_parts_data)}] {part_data['part_name'][:50]}")
                            parts_in_subcategory += 1
                    
                    if parts_in_subcategory >= MAX_PARTS_PER_SUBCATEGORY:
                        break
                    
                    page_num += 1
    
    # Final deduplication based on product_id
    final_df = pd.DataFrame(all_parts_data)
    if not final_df.empty and 'product_id' in final_df.columns:
        final_df = final_df.drop_duplicates(subset=['product_id'], keep='first')
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Removed {len(all_parts_data) - len(final_df)} duplicate entries")
    
    return final_df


def scrape_product_page(product_url, brand, model, category_name, subcategory_name, product_id, scrape_date):
    """
    Scrapes a single product page and extracts all relevant data.
    """
    soup = fetch_page(product_url)
    if not soup:
        return None
    
    page_text = soup.get_text()
    
    # Extract part name from page title
    part_name = None
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text()
        if '|' in title_text:
            part_name = title_text.split('|')[0].strip()
        else:
            part_name = title_text.split('-')[0].strip()
    
    part_name = clean_part_name(part_name, brand, model)
    
    # Extract price from JSON-LD structured data
    json_ld = soup.find('script', type='application/ld+json')
    price = None
    if json_ld:
        try:
            data = json.loads(json_ld.string)
            price = data.get('offers', {}).get('price')
        except Exception:
            pass
    
    # Extract quality grade
    quality_match = re.search(r'Laatu:\s*([A-C][1-3]?)', page_text)
    quality_grade = quality_match.group(1) if quality_match else None
    
    # Extract year range
    year_match = re.search(r'\((\d{4})\s*-\s*(\d{4})\)', page_text)
    year = f"{year_match.group(1)}-{year_match.group(2)}" if year_match else None
    
    # Extract OEM part number
    oem_number = None
    oem_patterns = [
        r'(?:alkuperäinen|oem)\s*(?:nro|numero|num)\s*:?\s*([A-Z0-9\-/]{6,20})',  # "Alkuperäinen nro: 12345-678"
        r'oem[:\-]?\s*([A-Z0-9\-/]{6,20})',                                       # "OEM: 12345-678"
        r'pn[:\-]?\s*([A-Z0-9\-/]{6,20})',                                        # "PN: 123456"
        r'([A-Z]{2,4}\d{4,8}[A-Z\-]?)',                                           # OEM-like: "1NDT12345", "47070-47090"
    ]
    for pattern in oem_patterns:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            raw_oem = m.group(1)
            # Clean and validate
            cleaned = re.sub(r'[^\w\-/]', '', raw_oem.upper())
            if len(cleaned) >= 6 and len(cleaned) <= 20:  # Reasonable OEM length
                oem_number = cleaned
                break
    
    # Extract engine code - IMPROVED PATTERNS
    engine_code = None
    engine_patterns = [
        r'Moottorin\s+koodi\s*:?\s*([A-Z0-9-]+)',           # "Moottorin koodi: 1ZZ-FE"
        r'moottorik[oö]{2}di\s*:?\s*([A-Z0-9-]+)',          # "Moottorikoodi: 1ZZ-FE" (no space)
        r'Moottori\s*:?\s*([A-Z]{1,2}\d{1,2}[A-Z]{0,3})',   # "Moottori: 1ZZ"
        r'Engine\s+code\s*:?\s*([A-Z0-9-]+)',               # English
    ]
    
    for pattern in engine_patterns:
        engine_match = re.search(pattern, page_text, re.IGNORECASE)
        if engine_match:
            raw_engine = engine_match.group(1).strip()
            # Validate: 3-8 chars, has letters AND digits, not a year
            if (3 <= len(raw_engine) <= 8 and 
                any(c.isalpha() for c in raw_engine) and 
                any(c.isdigit() for c in raw_engine) and
                not raw_engine.startswith(('19', '20'))):
                engine_code = raw_engine.replace('-', '').upper()
                break
    
    # Extract mileage
    mileage = None
    mileage_patterns = [
        r"lukema(?:t)?\s*\(km\)\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})",  # "lukemat (km): 123 456"
        r"lukema(?:t)?\s*([0-9][0-9\s\.,]{0,12})\s*km?",              # "lukema 123456 km"
        r"(\d{1,3}(?:\s|\.)?\d{3})\s*km",                              # "123 456 km" or "123.456 km"
        r"(\d{1,7})\s*km",                                             # "1234567 km" (1-7 digits)
        r"matkamittari\s*[:\-]?\s*([0-9][0-9\s\.,]{0,12})",            # "matkamittari: 123456"
    ]

    for pattern in mileage_patterns:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            digits_only = re.sub(r"[^\d]", "", raw)  # Remove spaces/dots/commas
            if digits_only and len(digits_only) >= 1:
                mileage = int(digits_only)
                break

    # Store extracted data
    return {
        'product_id': product_id,
        'part_name': part_name,
        'price': price,
        'quality_grade': quality_grade,
        'year': year,
        'oem_number': oem_number,
        'engine_code': engine_code,
        'mileage': mileage,
        'brand': brand,
        'model': model,
        'category': category_name,
        'subcategory': subcategory_name,
        'scrape_date': scrape_date
    }


if __name__ == '__main__':
    print("="*70)
    print("DPPM Car Parts Data Collector v1.3 - With Sampling")
    print("="*70)
    print(f"Max parts per subcategory: {MAX_PARTS_PER_SUBCATEGORY}")
    print()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--brand', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    base_output = f'dppm_{args.brand.lower()}_{args.model.lower().replace(",", "_").replace("-", "_")}'
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
    results = scrape_brand_model(args.brand, args.model)
    
    # Data quality summary
    print(f"\n{'='*70}")
    print("Scraping Completed!")
    print(f"{'='*70}")
    print(f"\nTotal parts scraped: {len(results)}")
    print(f"Output saved to: {OUTPUT_CSV}")
    
    if len(results) == 0:
        print("\nWARNING: No parts were scraped!")
    else:
        print(f"\n{'='*70}")
        print("Data Quality Summary:")
        print(f"{'='*70}")
        print(f"Parts with prices:        {results['price'].notna().sum():>6} / {len(results)}")
        print(f"Parts with OEM numbers:   {results['oem_number'].notna().sum():>6} / {len(results)}")
        print(f"Parts with engine codes:  {results['engine_code'].notna().sum():>6} / {len(results)}")
        print(f"Parts with mileage:       {results['mileage'].notna().sum():>6} / {len(results)}")
        print(f"Parts with quality grade: {results['quality_grade'].notna().sum():>6} / {len(results)}")
        
        print(f"\nQuality Grades Distribution:")
        for grade, count in results['quality_grade'].value_counts().items():
            print(f"  {grade}: {count}")
        
        print(f"\nPrice Statistics:")
        print(f"  Min:    €{results['price'].min():.2f}")
        print(f"  Max:    €{results['price'].max():.2f}")
        print(f"  Mean:   €{results['price'].mean():.2f}")
        print(f"  Median: €{results['price'].median():.2f}")
    print(f"\n{'='*70}")

"""
DPPM Car Parts Data Collector v1.0
Scrapes used car parts pricing data from Varaosahaku.fi for thesis research.
Collects: part names, prices, quality grades, OEM numbers, engine codes, mileage, and metadata.

Author: Ritesh Bhandari (ritesh.bhandari@edu.turkuamk.fi)
Institution: Turku University of Applied Sciences
"""

import pandas as pd
import time
import json
import re
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


# Configuration
USER_AGENT = 'ThesisScraper/1.0 (ritesh.bhandari@edu.turkuamk.fi; academic research)'
DELAY_SECONDS = 1.0
OUTPUT_CSV = 'dppm_corolla_test.csv'
CATEGORIES_TO_SCRAPE = ['Jarrut', 'Moottori', 'Kori']  # Brakes, Engine, Body


def fetch_page(url):
    """
    Fetches and renders JavaScript-heavy page using Playwright browser automation.
    Waits for network to be idle before extracting HTML content.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until='networkidle')
        html_content = page.content()
        browser.close()
    
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(html_content, 'lxml')


def clean_part_name(name):
    """
    Removes redundant brand, model, and year information from part names.
    Returns cleaned part name or 'Unknown Part' if cleaning fails.
    """
    if not name:
        return "Unknown Part"
    
    name = re.sub(r'TOYOTA\s+COROLLA', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\d{4}\s*-\s*\d{4}\)', '', name)
    name = ' '.join(name.split())
    
    return name.strip() or "Unknown Part"


def get_product_links_from_listing(soup):
    """
    Extracts all unique product URLs from a listing page.
    Identifies product links by their ID pattern in the URL.
    """
    product_urls = set()
    
    for link in soup.find_all('a', href=re.compile(r'ID-\d+')):
        url = urljoin('https://www.varaosahaku.fi/', link.get('href', ''))
        product_urls.add(url)
    
    return list(product_urls)


def scrape_brand_model(brand, model):
    """
    Main scraping function that crawls all categories and subcategories for a specific car brand and model.
    Extracts detailed product information from individual product pages.
    Saves data incrementally to CSV after each product to prevent data loss.
    """
    base_url = f"https://www.varaosahaku.fi/fi-fi/pb/Hae/Autonosat/s19/{brand}/{model}"
    main_page = fetch_page(base_url)
    
    all_parts_data = []
    visited_subcategory_urls = set()
    scraped_product_urls = set()
    
    # Navigate through main categories
    category_pattern = '|'.join([f'/{cat}$' for cat in CATEGORIES_TO_SCRAPE])
    category_links = main_page.find_all('a', href=re.compile(category_pattern))
    
    for category_link in category_links:
        category_name = category_link.get_text(strip=True)
        category_url = urljoin('https://www.varaosahaku.fi/', category_link['href'])
        print(f"\n{category_name}")
        
        category_page = fetch_page(category_url)
        processed_subcategories = set()
        
        # Navigate through subcategories within each main category
        for sub_link in category_page.find_all('a', href=True):
            subcategory_url = urljoin('https://www.varaosahaku.fi/', sub_link['href'])
            subcategory_name = sub_link.get_text(strip=True)
            
            # Filter out navigation links and duplicates
            if (not subcategory_name or
                subcategory_name in ['Etusivu', 'Jäsenyritykset', 'Toyota', 'Corolla', 'Autonosat', 'Valmistaja'] or
                subcategory_name in processed_subcategories or
                subcategory_url in visited_subcategory_urls or
                category_url not in subcategory_url or
                subcategory_url == category_url or
                ('?' in subcategory_url and subcategory_url.split('?')[0] == category_url)):
                continue
            
            visited_subcategory_urls.add(subcategory_url)
            processed_subcategories.add(subcategory_name)
            print(f"  {subcategory_name}")
            
            # Build listing page URL
            if '?' in subcategory_url:
                listing_page_url = f"{subcategory_url}&page=1"
            else:
                listing_page_url = f"{subcategory_url}?page=1"
            
            listing_page = fetch_page(listing_page_url)
            product_urls = get_product_links_from_listing(listing_page)
            
            # Scrape each individual product page
            for product_url in product_urls:
                if product_url in scraped_product_urls:
                    continue
                scraped_product_urls.add(product_url)
                
                soup = fetch_page(product_url)
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
                
                part_name = clean_part_name(part_name)
                print(f"    {part_name[:40]}")
                
                # Extract price from JSON-LD structured data
                json_ld = soup.find('script', type='application/ld+json')
                price = None
                if json_ld:
                    try:
                        data = json.loads(json_ld.string)
                        price = data.get('offers', {}).get('price')
                    except:
                        pass
                
                # Extract quality grade (A1, A2, B1, etc.)
                quality_match = re.search(r'Laatu:\s*([A-C][1-3]?)', page_text)
                quality_grade = quality_match.group(1) if quality_match else None
                
                # Extract year range
                year_match = re.search(r'\((\d{4})\s*-\s*(\d{4})\)', page_text)
                year = f"{year_match.group(1)}-{year_match.group(2)}" if year_match else None
                
                # Extract and clean OEM part number
                oem_number = None
                oem_match = re.search(r'Alkuperäinen\s+nro\s*:?\s*([A-Z0-9-]+)', page_text, re.IGNORECASE)
                if oem_match:
                    raw_oem = oem_match.group(1)
                    oem_number = re.sub(r'[A-Za-z]+$', '', raw_oem).strip('-')
                
                # Extract and clean engine code
                engine_code = None
                engine_match = re.search(r'Moottorin\s+koodi\s*:?\s*([A-Z0-9-]+)', page_text, re.IGNORECASE)
                if engine_match:
                    raw_engine = engine_match.group(1)
                    engine_code = re.sub(r'[A-Za-z]{4,}$', '', raw_engine).strip('-')
                
                # Extract mileage in kilometers
                mileage = None
                for pattern in [r'Lukema[:\s]*(\d{5,7})', r'(\d{5,7})\s*km', r'matkamittarilukema[:\s]*(\d{5,7})']:
                    mileage_match = re.search(pattern, page_text, re.IGNORECASE)
                    if mileage_match:
                        mileage = int(mileage_match.group(1))
                        break
                
                # Store extracted data
                part_data = {
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
                    'subcategory': subcategory_name
                }
                
                all_parts_data.append(part_data)
                pd.DataFrame(all_parts_data).to_csv(OUTPUT_CSV, index=False)
    
    return pd.DataFrame(all_parts_data)


if __name__ == '__main__':
    print("DPPM Data Collector v1.0\n")
    results = scrape_brand_model('Toyota', 'Corolla')
    print(f"\nCompleted: {len(results)} parts scraped")
    print(f"Output saved to: {OUTPUT_CSV}")

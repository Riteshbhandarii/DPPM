"""
Debug version of DPPM Car Parts Data Collector
This version prints detailed information about what's found on the page
"""
import argparse
import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


USER_AGENT = 'ThesisScraper/1.0 (ritesh.bhandari@edu.turkuamk.fi; academic research)'
DELAY_SECONDS = 1.0


def fetch_page(url):
    """Fetch page with Playwright"""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        
        try:
            page.goto(url, wait_until='networkidle', timeout=30000)
            html_content = page.content()
        except Exception as e:
            print(f"Error loading page: {e}")
            browser.close()
            return None
        
        browser.close()
    
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(html_content, 'lxml')


def debug_brand_model(brand, model):
    """Debug function to see what's on the page"""
    
    # Try both URL patterns
    base_urls = [
        f"https://www.varaosahaku.fi/fi-fi/pb/Hae/Autonosat/s1/{brand}/{model}",
        f"https://www.varaosahaku.fi/fi-fi/pb/Hae/Autonosat/s19/{brand}/{model}"
    ]
    
    print("="*70)
    print(f"DEBUG: Testing {brand} {model}")
    print("="*70)
    
    for url in base_urls:
        print(f"\n{'='*70}")
        print(f"Testing URL: {url}")
        print(f"{'='*70}")
        
        soup = fetch_page(url)
        if not soup:
            print("✗ Failed to load page")
            continue
        
        print("✓ Page loaded successfully\n")
        
        # Check page title
        title = soup.find('title')
        if title:
            print(f"Page Title: {title.get_text()}\n")
        
        # Find all links
        all_links = soup.find_all('a', href=True)
        print(f"Total links found: {len(all_links)}\n")
        
        # Categorize links
        print("="*70)
        print("ALL LINKS ON PAGE:")
        print("="*70)
        
        link_data = []
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Skip empty or very long text
            if not text or len(text) > 100:
                continue
            
            # Get link classes
            classes = ' '.join(link.get('class', []))
            
            link_data.append({
                'text': text,
                'href': href[:100],  # Truncate long URLs
                'classes': classes
            })
        
        # Print unique link texts with their hrefs
        seen_texts = set()
        for item in link_data:
            if item['text'] not in seen_texts:
                seen_texts.add(item['text'])
                print(f"Text: '{item['text']}'")
                print(f"  URL: {item['href']}")
                if item['classes']:
                    print(f"  Classes: {item['classes']}")
                print()
        
        # Look for specific patterns
        print("="*70)
        print("POTENTIAL CATEGORY LINKS:")
        print("="*70)
        
        base_model = model.split(',')[0].split('-')[0]
        for item in link_data:
            # Links that contain the model name and aren't navigation
            if (base_model.lower() in item['href'].lower() and 
                item['text'] not in ['Etusivu', 'Jäsenyritykset', brand, base_model, 'Autonosat', 'Valmistaja']):
                print(f"✓ {item['text']}")
                print(f"  {item['href']}")
                print()
        
        # Check for product links
        print("="*70)
        print("PRODUCT LINKS (containing 'ID-'):")
        print("="*70)
        product_count = 0
        for item in link_data:
            if 'ID-' in item['href']:
                product_count += 1
                if product_count <= 5:  # Show first 5
                    print(f"{item['text']}: {item['href']}")
        print(f"\nTotal product links: {product_count}\n")
        
        # If this URL worked, don't try the next one
        if len(all_links) > 10:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brand', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    debug_brand_model(args.brand, args.model)
    
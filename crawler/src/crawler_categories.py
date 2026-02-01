"""Category discovery and filtering."""

from urllib.parse import urljoin

from .crawler_utils import dedupe_preserve_order, normalize_url_for_match


def find_category_links(main_page, base_url, base_domain, brand, model):
    all_links = main_page.find_all("a", href=True)
    category_links = []

    brand_l = brand.lower()
    model_l = model.lower()
    base_url_norm = normalize_url_for_match(base_url).rstrip("/")

    blacklist = {
        brand,
        "Search unattached part",
        "Registration number search",
    }

    for link in all_links:
        href = link.get("href", "")
        link_text = link.get_text(strip=True)
        if not link_text or link_text in blacklist:
            continue

        full_url = urljoin(base_domain + "/", href)
        full_url_norm = normalize_url_for_match(full_url).rstrip("/")

        if "/s" in href and href.split("/s")[-1].isdigit():
            category_links.append((link_text, href))
            continue

        is_car_parts = "/pb/search/car-parts" in full_url_norm
        has_brand_model = f"/{brand_l}/" in full_url_norm and f"/{model_l}" in full_url_norm
        is_base_url = full_url_norm == base_url_norm

        if is_car_parts and has_brand_model and not is_base_url:
            category_links.append((link_text, href))

    return dedupe_preserve_order(category_links), all_links


def filter_categories(category_links, keep_categories):
    if not keep_categories:
        return category_links

    keep_normalized = {c.strip().lower() for c in keep_categories}
    return [
        (name, href)
        for (name, href) in category_links
        if name.strip().lower() in keep_normalized
    ]

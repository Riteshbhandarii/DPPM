"""Parse hand-saved varaosahaku.fi result pages into a listing table.

One saved HTML page = one part on one car, family view, whole census of that cell.
Each rendered <li class="item"> card is one live listing. The ld+json ItemList
carries the price; the card carries mileage, quality code, generation and the
href that supplies category, subcategory, position, fitment years and product_id.
The two are joined on the listing code (V17010) that both sides print.

usage: python scripts/parse_september_listings.py <folder> <out.csv>
"""

import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

CARD_SPLIT = re.compile(r'<li [^>]*class="item ')
TAGS = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
# Cmd+S writes some saves with root-relative hrefs and others with absolute
# ones, so accept both rather than anchoring on a leading slash.
HREF = re.compile(
    r'href="([^"]*?/Search/Car-parts/[^"]*?/ID-(\d+)[^"]*)"'
)
PATH = re.compile(
    r"/Search/Car-parts/[^/]+/(?P<brand>[^/]+)/(?P<gen>[^/]+)/"
    r"(?P<y0>\d{4})_(?P<y1>\d{4})/(?P<category>[^/]+)/(?P<subcategory>[^/]+)/"
    r"(?P<position>[^/]+)/ID-(?P<product_id>\d+)"
)


def path_label(segment):
    """URL path segment -> the label February stored.

    The site writes spaces as '-' and a literal hyphen as '_', so
    'Mass-air_flow-sensor' is February's 'mass air-flow sensor'.
    """
    return html.unescape(segment).replace("-", " ").replace("_", "-").lower()


def text_of(fragment):
    return WS.sub(" ", TAGS.sub(" | ", fragment))


def field(card_text, label):
    m = re.search(re.escape(label) + r"\s*\|[\s|]*:?[\s|]*([^|]+)", card_text)
    return m.group(1).strip() if m else None


def ld_prices(page):
    """listing code -> (price, currency, sku, oem, generation) from the ld+json ItemList."""
    out = {}
    for block in re.findall(
        r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', page, re.S
    ):
        try:
            data = json.loads(block.strip())
        except Exception:
            continue
        if isinstance(data, list):
            items = data
        elif data.get("@type") == "ItemList":
            items = [e.get("item", e) for e in data.get("itemListElement", [])]
        else:
            items = [data]
        for item in items:
            if not isinstance(item, dict) or item.get("@type") != "Product":
                continue
            name = item.get("name") or ""
            code = name.split("|")[0].strip()
            label = name.split("|", 1)[1].strip() if "|" in name else None
            brand = item.get("brand")
            offers = item.get("offers") or {}
            out[code] = {
                "price": offers.get("price"),
                "currency": offers.get("priceCurrency"),
                "sku": item.get("sku"),
                "oem_number": item.get("mpn"),
                "generation": brand.get("name") if isinstance(brand, dict) else brand,
                "seller": ((offers.get("seller") or {}).get("name")),
                "label": label,
            }
    return out


def parse_page(path):
    page = Path(path).read_text(encoding="utf-8", errors="ignore")
    prices = ld_prices(page)
    rows = []
    for card in CARD_SPLIT.split(page)[1:]:
        href = HREF.search(card)
        if not href:
            continue
        parts = PATH.search(href.group(1))
        if not parts:
            continue
        card_text = text_of(card)
        code = re.search(r'title="(V\d+)"', card)
        code = code.group(1) if code else None
        listing = prices.get(code, {})
        mileage = field(card_text, "Mileage (km)")
        quality = re.search(r"Quality code.{0,80}?\(([A-C]\d?)\)", card_text)
        rows.append(
            {
                "source_file": Path(path).name,
                "listing_code": code,
                "product_id": int(parts.group("product_id")),
                "sku": listing.get("sku"),
                "price": listing.get("price"),
                "currency": listing.get("currency"),
                "part_label": listing.get("label"),
                "generation": listing.get("generation"),
                "seller": listing.get("seller"),
                "oem_number": listing.get("oem_number"),
                "category": path_label(parts.group("category")).replace(" & ", " / "),
                "subcategory": path_label(parts.group("subcategory")),
                "position": (
                    None if parts.group("position") == "_"
                    else parts.group("position").replace("-", " ")
                ),
                "year_start": int(parts.group("y0")),
                "year_end": int(parts.group("y1")),
                "mileage": int(re.sub(r"\D", "", mileage)) if mileage and re.sub(r"\D", "", mileage) else None,
                "quality_grade": quality.group(1) if quality else None,
                "model_year": field(card_text, "Model Year"),
                "engine_code": field(card_text, "Engine code"),
            }
        )
    return rows


def main(folder, out_path):
    seen_digests = {}
    rows = []
    for html in sorted(Path(folder).glob("*.html")):
        digest = hashlib.md5(html.read_bytes()).hexdigest()
        if digest in seen_digests:
            print(f"  skip duplicate file: {html.name} == {seen_digests[digest]}")
            continue
        seen_digests[digest] = html.name
        page_rows = parse_page(html)
        print(f"  {len(page_rows):4d}  {html.name}")
        rows += page_rows

    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} listings -> {out_path}")
    print(f"distinct product_id: {len({r['product_id'] for r in rows})}")
    missing = [r for r in rows if r["price"] is None]
    if missing:
        print(f"WARNING: {len(missing)} rows with no price")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

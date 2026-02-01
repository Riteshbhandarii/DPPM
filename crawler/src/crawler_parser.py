"""Parsing helpers for product pages."""

import json
import re

from .crawler_utils import clean_part_name, parse_price


def parse_product_page(soup, brand, model):
    page_text = soup.get_text()

    # Part name
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

    # Price
    json_ld = soup.find("script", type="application/ld+json")
    price = None
    if json_ld:
        try:
            data = json.loads(json_ld.string)
            price = data.get("offers", {}).get("price")
        except Exception:
            price = None
    price = parse_price(price)

    # Quality grade
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

    # Year range
    year_match = re.search(r"\((\d{4})\s*-\s*(\d{4})\)", page_text)
    year = f"{year_match.group(1)}-{year_match.group(2)}" if year_match else None

    # OEM number
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

    # Engine code
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

    # Mileage
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
        "part_name": part_name,
        "price": price,
        "quality_grade": quality_grade,
        "year": year,
        "oem_number": oem_number,
        "engine_code": engine_code,
        "mileage": mileage,
    }

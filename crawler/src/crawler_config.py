"""Shared configuration for the DPPM crawler."""

USER_AGENT = "ThesisScraper/1.0 (ritesh.bhandari@edu.turkuamk.fi; academic research)"
DELAY_SECONDS = 1.0
TIMEZONE = "Europe/Helsinki"
BASE_URL = "https://www.varaosahaku.fi/en-se"

KEEP_CATEGORIES = [
    "Brakes",
    "Engine",
    "Airbag",
    "Gear box / Drive axle / Middle axle",
    "Fuel",
    "Electric / Transmitter / Databox / Sensor",
    "Vehicle exterior / Suspension",
]

MAX_PARTS_PER_SUBCATEGORY = 30

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
    "scrape_timestamp",
]

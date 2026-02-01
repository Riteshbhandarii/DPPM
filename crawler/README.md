# Crawler

This folder contains the web crawler for collecting used car parts data from Varaosahaku.fi.

## Structure

- `__init__.py` — Marks `crawler/` as a Python package.
- `__main__.py` — Package entrypoint for `python -m crawler`.
- `src/` — Crawler source code.
  - `crawler.py` — CLI entrypoint (builds output path, runs scrape, prints summary).
  - `crawler_config.py` — All crawler settings (user agent, limits, categories, columns).
  - `crawler_utils.py` — Shared helpers (fetch page, parse price, clean names, etc.).
  - `crawler_categories.py` — Category discovery + filtering.
  - `crawler_parser.py` — Product-page parsing (price, OEM, mileage, etc.).
  - `crawler_scraper.py` — Main scraping workflow (categories, paging, CSV writing).
- `crawler_datasets/` — Output data for the crawler.
  - `new/` — Latest runs.
  - `old/` — Archived runs.

## How to run

From repo root:

```
python3 -m crawler --brand Toyota --model Corolla
```

Outputs go to `crawler/crawler_datasets/new/`.

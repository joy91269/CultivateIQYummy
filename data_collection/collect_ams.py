"""Download USDA AMS terminal market wholesale prices for the study products (MyMarketNews MARS API).

Requires a free MyMarketNews API key (log in at https://mymarketnews.ams.usda.gov, click your name,
then "Show API key"). Provide it through the environment variable MARS_API_KEY, e.g. in a local `.env` file:
    MARS_API_KEY=xxxxxxxx
The key is never written to the repository.

Usage (from the repository root):
    python data_collection/collect_ams.py

Output:
    data/raw/ams/<market>_<commodity>_<year>.json     raw API responses (one per market x commodity x year)
    data/processed/ams_terminal_prices.csv.gz         tidy table, one row per reported price line
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from common import PROCESSED, RAW, ROOT, USER_AGENT, load_yaml, study_period

OUT = RAW / "ams"
KEEP = ["report_date", "market_location_city", "market_location_state", "slug_name", "commodity", "variety",
        "package", "item_size", "origin", "organic", "low_price", "high_price", "mostly_low_price",
        "mostly_high_price", "market_tone_comments", "offerings_comments"]


def api_key() -> str:
    key = os.environ.get("MARS_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MARS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("MARS_API_KEY is not set (see the docstring of this script).")
    return key


def fetch(session: requests.Session, url: str, q: str, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            r = session.get(url, params={"q": q}, timeout=300)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            err = str(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"AMS request failed for {url} {q}: {err}")


def main() -> None:
    cfg = load_yaml("ams_markets.yaml")
    products = load_yaml("products.yaml")["products"]
    start, end = study_period()
    session = requests.Session()
    session.auth = (api_key(), "")
    session.headers["User-Agent"] = USER_AGENT
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for market, m in cfg["markets"].items():
        for pid, spec in products.items():
            slug = m[spec["ams"]["report_group"]]
            url = f"{cfg['mars_api']}/{slug}/{cfg['section']}"
            for commodity in spec["ams"]["commodities"]:
                for year in range(start.year, end.year if end.month == 1 and end.day == 1 else end.year + 1):
                    path = OUT / f"{market.replace(' ', '_')}_{commodity.replace(', ', '_').replace(' ', '_')}_{year}.json"
                    if not path.exists():
                        # commodity names such as "Lettuce, Iceberg" contain a comma, which the MARS query
                        # language treats as a list separator, so the value must be double-quoted
                        q = f'report_begin_date=01/01/{year}:12/31/{year};commodity="{commodity}"'
                        payload = fetch(session, url, q)
                        path.write_text(json.dumps(payload))
                        print(f"{market:13s} {commodity:18s} {year}: {len(payload.get('results') or [])} rows",
                              flush=True)
                        time.sleep(1.0)
                    rows = json.loads(path.read_text()).get("results") or []
                    if rows:
                        df = pd.DataFrame(rows)[KEEP]
                        df["product"] = pid
                        df["market"] = market
                        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices["report_date"] = pd.to_datetime(prices["report_date"], format="%m/%d/%Y")
    for col in ["low_price", "high_price", "mostly_low_price", "mostly_high_price"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.sort_values(["product", "market", "report_date"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PROCESSED / "ams_terminal_prices.csv.gz", index=False, compression="gzip")
    print(f"wrote {len(prices)} price lines -> data/processed/ams_terminal_prices.csv.gz")


if __name__ == "__main__":
    main()

"""Download weekly US Google search interest for each product (Google Trends) as a second online signal.

For every product the same four search terms are queried (config/products.yaml, `google_trends`): the product name
alone, and the name with "price", "shortage" and "recall". Each term is requested on its own, US only, over the study
period, which Google Trends returns as one value per week (Sunday to Saturday). Google scales every request to
0-100 (100 = the term's busiest week), so values compare weeks within a term, not terms with each other; 0 means too
few searches to report. The analysis only compares each week with the previous weeks of the same term.

Google Trends has no public API; the data come through the pytrends library, which calls the same endpoints as the
website. Google rate-limits these calls, so the script pauses between terms, retries with a growing wait, and keeps
every downloaded term in data/raw/google_trends/ so that a rerun only fetches what is missing. Google samples the
searches behind each request, so a new download can differ slightly from the committed one.

Usage (from the repository root):
    python data_collection/collect_google_trends.py

Output:
    data/raw/google_trends/<term>.csv          one file per search term, as returned (weeks starting Sunday)
    data/processed/google_trends_weekly.csv    product, series, term, week_start (Monday), interest
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
from pytrends.request import TrendReq

from common import PROCESSED, RAW, load_yaml, study_period

OUT = RAW / "google_trends"
PAUSE_S = 8


def fetch(py: TrendReq, term: str, timeframe: str, geo: str, tries: int = 6) -> pd.DataFrame:
    for attempt in range(tries):
        try:
            py.build_payload([term], timeframe=timeframe, geo=geo)
            df = py.interest_over_time()
            if df.empty:
                raise RuntimeError(f"no data returned for '{term}'")
            return df.reset_index().rename(columns={"date": "google_week_start", term: "interest",
                                                    "isPartial": "is_partial"})
        except Exception as e:  # pytrends raises its own errors for HTTP 429 and others
            wait = 60 * (attempt + 1)
            print(f"  {term}: {type(e).__name__}: {e}; retrying in {wait} s")
            time.sleep(wait)
    raise SystemExit(f"could not download '{term}' after {tries} tries")


def main() -> None:
    cfg = load_yaml("products.yaml")
    gt = cfg["google_trends"]
    start, end = study_period()
    last = (end.date() - timedelta(days=1))
    timeframe = f"{start.date().isoformat()} {last.isoformat()}"
    OUT.mkdir(parents=True, exist_ok=True)
    py = TrendReq(hl="en-US", tz=360)

    frames = []
    for product in cfg["products"]:
        for series, template in gt["terms"].items():
            term = template.format(product=product)
            path = OUT / f"{term.replace(' ', '_')}.csv"
            if not path.exists():
                print(f"downloading '{term}' ({gt['geo']}, {timeframe})")
                fetch(py, term, timeframe, gt["geo"]).to_csv(path, index=False)
                time.sleep(PAUSE_S)
            df = pd.read_csv(path, parse_dates=["google_week_start"])
            df.insert(0, "term", term)
            df.insert(0, "series", series)
            df.insert(0, "product", product)
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # Google weeks run Sunday to Saturday; the Reddit and price series use ISO weeks starting on Monday.
    # Each Google week is assigned to the ISO week that shares six of its seven days.
    out["week_start"] = out["google_week_start"] + pd.Timedelta(days=1)
    out = out[(out["week_start"] >= pd.Timestamp(start.date())) & (out["week_start"] < pd.Timestamp(end.date()))]
    out = out[["product", "series", "term", "week_start", "google_week_start", "interest", "is_partial"]]
    out.to_csv(PROCESSED / "google_trends_weekly.csv", index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(out)} rows for {out['term'].nunique()} terms to {PROCESSED / 'google_trends_weekly.csv'}")
    print(f"accessed {date.today().isoformat()}")


if __name__ == "__main__":
    main()

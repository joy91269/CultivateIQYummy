"""Download USDA AMS daily movement (shipment) reports for the study products and build weekly national shipments.

Shipments are the pounds of each commodity shipped from US production districts plus imports through ports of
entry, as reported by every Market News office (config/ams_markets.yaml, `movement_reports`). Together with the
terminal market prices they show whether a price change came with less product reaching the market (supply shock)
or with the same or more (demand shift).

Requires the same free MyMarketNews API key as collect_ams.py (MARS_API_KEY in the environment or in `.env`).

Usage (from the repository root):
    python data_collection/collect_ams_movement.py
Downloads can be split over parallel processes and the tables built afterwards from the cached responses:
    python data_collection/collect_ams_movement.py --offices "McAllen TX" "Nogales AZ" --download-only

Output:
    data/raw/ams_movement/<office>_<commodity>_<year>.json   raw API responses
    data/processed/ams_movement_daily.csv.gz                 one row per reported shipment line
    data/processed/ams_shipments_weekly.csv                  pounds per product and ISO week, domestic vs imports
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from collect_ams import api_key, fetch
from common import PROCESSED, RAW, USER_AGENT, load_yaml, study_period

OUT = RAW / "ams_movement"
KEEP = ["report_date", "slug_name", "trans_Mode", "commodity", "district", "origin", "import/Export", "asw_Flag",
        "asw_date", "season", "1 lb units"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offices", nargs="*", default=None, help="only these offices (names in config)")
    ap.add_argument("--download-only", action="store_true", help="download and cache, do not build the tables")
    args = ap.parse_args()
    cfg = load_yaml("ams_markets.yaml")
    products = load_yaml("products.yaml")["products"]
    start, end = study_period()
    years = range(start.year, end.year if end.month == 1 and end.day == 1 else end.year + 1)
    session = requests.Session()
    session.auth = (api_key(), "")
    session.headers["User-Agent"] = USER_AGENT
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for office, slug in cfg["movement_reports"].items():
        if args.offices and office not in args.offices:
            continue
        url = f"{cfg['mars_api']}/{slug}/{cfg['section']}"
        for pid, spec in products.items():
            for commodity in spec["ams"]["commodities"]:
                for year in years:
                    path = OUT / f"{office.replace(' ', '_')}_{commodity.replace(', ', '_').replace(' ', '_')}_{year}.json"
                    if not path.exists():
                        q = f'report_begin_date=01/01/{year}:12/31/{year};commodity="{commodity}"'
                        payload = fetch(session, url, q)
                        path.write_text(json.dumps(payload))
                        print(f"{office:17s} {commodity:18s} {year}: {len(payload.get('results') or [])} rows",
                              flush=True)
                        time.sleep(1.0)
                    rows = json.loads(path.read_text()).get("results") or []
                    if rows:
                        df = pd.DataFrame(rows).reindex(columns=KEEP)
                        df["product"] = pid
                        df["office"] = office
                        frames.append(df)
    if args.download_only:
        return
    mv = pd.concat(frames, ignore_index=True)
    mv["pounds"] = pd.to_numeric(mv.pop("1 lb units"), errors="coerce")
    mv["report_date"] = pd.to_datetime(mv["report_date"], format="%m/%d/%Y")
    # late additions ("asw_Flag" set) belong to the earlier shipping date given in asw_date
    late = pd.to_datetime(mv["asw_date"], format="%m/%d/%Y", errors="coerce")
    mv["ship_date"] = late.where(mv["asw_Flag"].notna() & late.notna(), mv["report_date"])
    # the same line can appear in two offices' reports (e.g. rail shipments); keep it once
    key = ["ship_date", "trans_Mode", "commodity", "district", "origin", "import/Export", "asw_Flag", "pounds"]
    n0 = len(mv)
    mv = mv.drop_duplicates(key).sort_values(["product", "ship_date"]).reset_index(drop=True)
    mv["import"] = mv["import/Export"].astype(str).str.startswith("I")
    mv = mv[(mv["ship_date"] >= start) & (mv["ship_date"] < end)]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    mv.drop(columns=["import/Export"]).to_csv(PROCESSED / "ams_movement_daily.csv.gz", index=False,
                                             compression="gzip")
    mv["week_start"] = mv["ship_date"] - pd.to_timedelta(mv["ship_date"].dt.weekday, unit="D")
    wk = (mv.pivot_table(index=["product", "week_start"], columns="import", values="pounds", aggfunc="sum",
                         fill_value=0)
          .rename(columns={False: "domestic_lb", True: "import_lb"}).reset_index())
    for c in ["domestic_lb", "import_lb"]:
        if c not in wk:
            wk[c] = 0.0
    wk["total_lb"] = wk["domestic_lb"] + wk["import_lb"]
    wk.to_csv(PROCESSED / "ams_shipments_weekly.csv", index=False)
    print(f"{n0} lines, {len(mv)} after removing duplicates -> ams_movement_daily.csv.gz; "
          f"{len(wk)} product-weeks -> ams_shipments_weekly.csv")


if __name__ == "__main__":
    main()

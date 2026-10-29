"""Three more free online signals, 2019-01-01 to 2026-09-30, for the forecast check:
GDELT US news volume and tone, openFDA food recalls, and Wikipedia pageviews.

    python forecast_check/collect_extra.py

Raw responses are cached in forecast_check/data/raw_extra/ (not committed); the tidy daily table is written to
forecast_check/data/extra_daily.csv. GDELT allows about one request every 5 s; when it answers HTTP 429 the script
waits and retries, so a full run can take a few hours.
"""
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data"
RAW = OUT / "raw_extra"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "CultivateIQ-student-data-challenge/1.0 (academic research; contact via GitHub joy91269)"}
S = requests.Session()
S.headers.update(UA)
START, END = "20190101", "20260930"

PROD = {"avocado": "(avocado OR avocados OR guacamole)", "lettuce": "(lettuce OR romaine)",
        "onion": "(onion OR onions)", "cucumber": "(cucumber OR cucumbers)"}
TOPIC = {"all": "", "shortage": "(shortage OR shortages OR scarce)", "recall": "(recall OR recalled OR outbreak)",
         "price": "(price OR prices OR expensive)"}
FDA = {"avocado": "avocado", "lettuce": "lettuce+romaine", "onion": "onion", "cucumber": "cucumber"}
WIKI = {"avocado": "Avocado", "lettuce": "Lettuce", "onion": "Onion", "cucumber": "Cucumber"}


def get(url: str, path: Path, pause: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    for t in range(12):
        r = S.get(url, timeout=120)
        if r.status_code == 200:
            try:
                js = r.json()
            except ValueError:
                js = None
            if js is not None:
                path.write_text(json.dumps(js))
                time.sleep(pause)
                return js
        time.sleep(60 * (t + 1) if r.status_code == 429 else pause * (t + 2))
    raise RuntimeError(f"failed {url} ({r.status_code}: {r.text[:200]})")


rows = []
# GDELT DOC 2.0: one call per product x topic x measure over the whole period (daily resolution)
for p, pq in PROD.items():
    for tname, tq in TOPIC.items():
        q = f"{pq} {tq} sourcecountry:US sourcelang:english".replace("  ", " ")
        for mode in ["TimelineVolRaw", "TimelineTone"]:
            url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
                {"query": q, "mode": mode, "format": "json",
                 "startdatetime": START + "000000", "enddatetime": END + "235959"})
            js = get(url, RAW / f"gdeltfull_{p}_{tname}_{mode}.json", 15)
            res = js.get("query_details", {}).get("date_resolution")
            for tl in js.get("timeline", []):
                for d in tl.get("data", []):
                    rows.append({"source": "gdelt", "product": p, "topic": tname, "resolution": res, "measure": mode,
                                 "date": d["date"][:8], "value": d["value"], "norm": d.get("norm")})
        print("gdelt", p, tname, flush=True)

# openFDA food enforcement reports (recalls) that name the product
for p, term in FDA.items():
    for field in ["product_description", "reason_for_recall"]:
        url = (f"https://api.fda.gov/food/enforcement.json?search={field}:({term})"
               f"+AND+recall_initiation_date:[{START}+TO+{END}]&count=recall_initiation_date")
        js = get(url, RAW / f"fda_{p}_{field}.json", 1)
        for d in js.get("results", []):
            rows.append({"source": "fda", "product": p, "topic": field, "measure": "recalls",
                         "date": d["time"], "value": d["count"]})
    print("fda", p, flush=True)

# Wikipedia pageviews of the English article (users only, no bots)
for p, art in WIKI.items():
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}"
           f"/daily/{START}/{END}")
    js = get(url, RAW / f"wiki_{p}.json", 1)
    for d in js.get("items", []):
        rows.append({"source": "wikipedia", "product": p, "topic": "article", "measure": "views",
                     "date": d["timestamp"][:8], "value": d["views"]})
    print("wiki", p, flush=True)

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df.to_csv(OUT / "extra_daily.csv", index=False)
print(df.groupby(["source", "measure"]).size())

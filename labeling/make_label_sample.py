"""Draw the stratified sample that is labelled by hand (price / supply / demand / other).

Default strata: product x record type (post / comment) x (keyword candidate vs. no market keyword). Two thirds of
each cell come from observations whose text contains at least one market-event keyword, one third from the rest.

`--quota` instead draws a fixed number per keyword group and product, which enriches the rare classes:
    price  = text contains a price keyword
    supply = supply-shortage, weather or food-safety keyword (and no price keyword)
    demand = demand or promotion keyword only
    none   = no market keyword
At most one observation per thread is drawn so that labelled examples are as independent as possible.
`--append` adds a batch to an existing labeling/label_sample.csv without re-drawing earlier rows or threads.

Batches used in this project (see the `batch` column):
    python labeling/make_label_sample.py --record-type post --per-product 80
    python labeling/make_label_sample.py --record-type comment --quota price:15,supply:15,demand:5,none:5 --near --append
    python labeling/make_label_sample.py --record-type post --quota price:12,supply:12 --append --seed 7
    python labeling/make_label_sample.py --record-type post --subreddit news --per-product 12 --append
    python labeling/make_label_sample.py --record-type comment --quota price:8,supply:8,demand:16,none:4 --near \
        --append --seed 11 --tag 2
    python labeling/make_label_sample.py --record-type post --quota demand:14,supply:6,price:6 --append --seed 13 --tag 2
The last two batches (tag 2) were added after the first model comparison showed too few demand examples.
    python labeling/make_label_sample.py --record-type any --random 40 --append --seed 21
    python labeling/make_label_sample.py --record-type comment --quota supply:10,demand:15 --near --append --seed 23 --tag 3
    python labeling/make_label_sample.py --record-type post --quota supply:8,demand:8 --append --seed 29 --tag 3
The random batch (40 per product, no keyword strata) measures how common each label is in the whole dataset; the
other two (tag 3) add rare-class examples after a learning curve showed scores still rising with more labels.
Batch 6 ("silver" labels: AI-labelled, used for training only, see labeling/codebook.md):
    python labeling/make_label_sample.py --record-type any --random 100 --append --seed 31 --tag 6
    python labeling/make_label_sample.py --record-type any --random 57 --append --seed 43 --tag 6
    python labeling/make_label_sample.py --record-type comment --quota price:40,supply:60,demand:60,none:20 --near \
        --append --seed 37 --tag 6
    python labeling/make_label_sample.py --record-type post --quota price:40,supply:60,demand:60,none:20 --append \
        --seed 41 --tag 6
Output: labeling/label_sample.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_collection"))
OUT = ROOT / "labeling/label_sample.csv"
COLS = ["obs_id", "batch", "product", "record_type", "subreddit", "date", "thread_id", "event_types", "text", "url"]


def near_group(text: str, product: str, window: int = 80) -> str:
    """Keyword group whose keyword occurs within `window` characters of a mention of the product."""
    from build_keywords import EventTagger
    from common import load_yaml
    global _PAT
    if "_PAT" not in globals():
        tagger = EventTagger()
        prods = load_yaml("products.yaml")["products"]
        _PAT = ({k: re.compile(v["regex"], re.I) for k, v in prods.items()}, tagger.patterns)
    prod_re, ev_pats = _PAT
    text = str(text)
    spans = [m.span() for m in prod_re[product].finditer(text)]
    groups = {"price": ["price_change"], "supply": ["supply_shortage", "weather_event", "food_safety_recall"],
              "demand": ["demand_change", "marketing_promotion"]}
    for grp, types in groups.items():
        for t in types:
            for m in ev_pats[t].finditer(text):
                if any(abs(m.start() - a) <= window or abs(m.start() - b) <= window for a, b in spans):
                    return grp
    return "none" if not any(p.search(text) for p in ev_pats.values()) else "far"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-type", choices=["post", "comment", "any"], required=True)
    ap.add_argument("--random", type=int, default=0,
                    help="simple random sample of this many observations per product (no keyword strata): "
                         "measures how common each label really is")
    ap.add_argument("--per-product", type=int, default=0)
    ap.add_argument("--quota", default="", help="e.g. price:15,supply:15,demand:5,none:5 (per product)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--subreddit", default=None, help="restrict the batch to one subreddit (e.g. news)")
    ap.add_argument("--tag", default="", help="suffix for the batch name, e.g. 2 -> posts_enriched2")
    ap.add_argument("--near", action="store_true",
                    help="with --quota: a price/supply/demand keyword must occur within 80 characters of the product")
    args = ap.parse_args()
    ev = pd.read_csv(ROOT / "data/processed/event_dataset.csv", low_memory=False)
    if args.record_type != "any":
        ev = ev[ev["record_type"] == args.record_type]
    ev = ev.copy()
    if args.subreddit:
        ev = ev[ev["subreddit"] == args.subreddit]
    old = pd.read_csv(OUT) if args.append and OUT.exists() else pd.DataFrame(columns=COLS)
    ev = ev[~ev["thread_id"].astype(str).isin(old["thread_id"].astype(str))]
    ev["candidate"] = ev["event_types"] != "none"
    ev = ev.sample(frac=1, random_state=args.seed).drop_duplicates("thread_id")   # one per thread
    et = ev["event_types"]
    ev["kw_group"] = "none"
    ev.loc[et.str.contains("demand_change|marketing_promotion"), "kw_group"] = "demand"
    ev.loc[et.str.contains("supply_shortage|weather_event|food_safety_recall"), "kw_group"] = "supply"
    ev.loc[et.str.contains("price_change"), "kw_group"] = "price"
    if args.near:
        ev["kw_group"] = [near_group(t, p) for t, p in zip(ev["text"], ev["product"])]
    parts = []
    for _, g in ev.groupby("product"):
        if args.random:
            parts.append(g.head(args.random))     # g is already shuffled: a simple random sample
        elif args.quota:
            for item in args.quota.split(","):
                grp, n = item.split(":")
                parts.append(g[g["kw_group"] == grp].head(int(n)))
        else:
            n_cand = round(args.per_product * 2 / 3)
            cand, rest = g[g["candidate"]], g[~g["candidate"]]
            parts.append(cand.head(n_cand))
            parts.append(rest.head(args.per_product - min(n_cand, len(cand))))
    batch = pd.concat(parts).sample(frac=1, random_state=args.seed)
    batch["batch"] = ("random" if args.random else f"{args.record_type}s" + ("_enriched" if args.quota else "")
                      + (f"_{args.subreddit}" if args.subreddit else "")) + args.tag
    print(batch.groupby(["product", "kw_group"]).size().to_string())
    sample = pd.concat([old, batch[COLS]], ignore_index=True)   # keeps columns added by later steps
    sample.to_csv(OUT, index=False)
    print(f"{len(batch)} new rows, {len(sample)} total -> labeling/label_sample.csv")


if __name__ == "__main__":
    main()

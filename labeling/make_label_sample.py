"""Draw the stratified sample that is labelled by hand (price / supply / demand / other).

Strata: product x (keyword candidate vs. no market keyword). Two thirds of each product's quota comes from
observations whose text contains at least one market-event keyword (so that the rarer price/supply/demand
classes are represented), one third from the rest. At most one observation per thread is drawn so that the
labelled examples are as independent as possible.

Usage (from the repository root):
    python labeling/make_label_sample.py --per-product 150
Output: labeling/label_sample.csv (columns to fill: label, notes)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-product", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    ev = pd.read_csv(ROOT / "data/processed/event_dataset.csv", low_memory=False)
    ev["candidate"] = ev["event_types"] != "none"
    ev = ev.sample(frac=1, random_state=args.seed).drop_duplicates("thread_id")   # one per thread
    parts = []
    for prod, g in ev.groupby("product"):
        n_cand = round(args.per_product * 2 / 3)
        cand, rest = g[g["candidate"]], g[~g["candidate"]]
        parts.append(cand.head(n_cand))
        parts.append(rest.head(args.per_product - min(n_cand, len(cand))))
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample["label"] = ""
    sample["notes"] = ""
    cols = ["obs_id", "product", "record_type", "subreddit", "date", "thread_id", "event_types", "text", "url",
            "label", "notes"]
    sample[cols].to_csv(ROOT / "labeling/label_sample.csv", index=False)
    print(sample.groupby(["product", "candidate"]).size().to_string())
    print(f"wrote {len(sample)} rows -> labeling/label_sample.csv")


if __name__ == "__main__":
    main()

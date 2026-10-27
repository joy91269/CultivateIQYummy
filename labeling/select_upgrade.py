"""Choose silver texts for a person to check, so that they become gold labels (review round 4).

Silver texts carry the label two AI annotators agreed on, and are used only for training. A person confirms or
corrects the agreed label on the review page; a silver text with a human decision becomes gold
(labeling/collect_reviews.py). The 40 silver texts the person already checked in the silver spot check become gold
the same way.

Selection (seed 47): every unreviewed silver text whose agreed label is price, supply or demand (the rare classes,
where gold is thinnest), plus a simple random sample of agreed "other" texts, 300 texts in all.

Adds or updates the column `upgrade` (True for texts picked to become gold) in labeling/label_sample.csv.
How a picked text becomes gold is decided in labeling/collect_reviews.py.

Usage: python labeling/select_upgrade.py [--n 300]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 47


def main(n: int) -> None:
    path = ROOT / "labeling/label_sample.csv"
    s = pd.read_csv(path)
    rev = json.loads((ROOT / "labeling/reviews.json").read_text())
    silver = s["tier"].eq("silver")
    agree = s["label_ai1"].eq(s["label_ai2"])
    reviewed = s["obs_id"].isin(rev)
    pool = s[silver & agree & ~reviewed]
    signal = pool[pool["label_ai1"].isin(["price", "supply", "demand"])]
    other = pool[pool["label_ai1"].eq("other")].sample(n=max(0, n - len(signal)), random_state=SEED)
    picks = set(signal["obs_id"]) | set(other["obs_id"])
    if "upgrade" not in s:
        s["upgrade"] = False
    s["upgrade"] = s["upgrade"].fillna(False).astype(bool) | s["obs_id"].isin(picks) | (silver & reviewed)
    s.to_csv(path, index=False)
    print(f"{len(picks)} silver texts picked for review ({len(signal)} price/supply/demand, {len(other)} other); "
          f"{int((silver & reviewed).sum())} already-checked silver texts also marked to become gold")
    print(s[s["obs_id"].isin(picks)].groupby(["product", "label_ai1"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args().n)

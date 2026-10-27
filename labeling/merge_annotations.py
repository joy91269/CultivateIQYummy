"""Attach the two independent AI annotations to the sample and pick the rows a person must review.

Input
    labeling/label_sample.csv
    labeling/annotations/annotator1_*.jsonl   {"obs_id", "label", "reason"} per line
    labeling/annotations/annotator2_*.jsonl
Output (written back to labeling/label_sample.csv)
    label_ai1, reason_ai1, label_ai2, reason_ai2, needs_review
needs_review = every disagreement + a random check of agreements. Rows are reviewed in rounds: rows that already
have a needs_review flag keep it (they may already be reviewed); only rows added since get a new flag.
    round 1 (first 577 rows): up to 10 random agreements per agreed label
    round 2 (batch 4, 217 rows): up to 3 random agreements per agreed label

    round 4 (batch 6, 1,883 rows, --silver): "silver" labels for training only. Disagreements are not reviewed
             but dropped; a person checks a random sample of 10 agreed labels per label to estimate their accuracy.

Usage: python labeling/merge_annotations.py [--check-per-label N] [--silver]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "labeling/annotations"
LABELS = ["price", "supply", "demand", "other"]


def read(prefix: str) -> pd.DataFrame:
    rows = []
    for p in sorted(ANN.glob(f"{prefix}_*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates("obs_id", keep="last")
    bad = ~df["label"].isin(LABELS)
    if bad.any():
        raise SystemExit(f"{prefix}: {bad.sum()} invalid labels")
    return df.set_index("obs_id")


def main(check_per_label: int = 10, seed: int = 2026, silver: bool = False) -> None:
    s = pd.read_csv(ROOT / "labeling/label_sample.csv")
    if "tier" not in s:
        s["tier"] = pd.NA
    prev = s["needs_review"] if "needs_review" in s else pd.Series(pd.NA, index=s.index)
    new = prev.isna()
    if "review_round" not in s:
        s["review_round"] = pd.NA
    s.loc[new, "review_round"] = int(pd.to_numeric(s["review_round"], errors="coerce").max(skipna=True) + 1) \
        if s["review_round"].notna().any() else 1
    a1, a2 = read("annotator1"), read("annotator2")
    s["label_ai1"], s["reason_ai1"] = s["obs_id"].map(a1["label"]), s["obs_id"].map(a1["reason"])
    s["label_ai2"], s["reason_ai2"] = s["obs_id"].map(a2["label"]), s["obs_id"].map(a2["reason"])
    missing = s["label_ai1"].isna() | s["label_ai2"].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum()} rows lack an annotation")
    agree = s["label_ai1"] == s["label_ai2"]
    check = (s[agree & new].groupby("label_ai1", group_keys=False)
             .apply(lambda g: g.sample(min(check_per_label, len(g)), random_state=seed)).index)
    s["needs_review"] = prev.astype("boolean")
    # gold: every disagreement is decided by a person; silver: disagreements are dropped, only the check is reviewed
    s.loc[new, "needs_review"] = False if silver else ~agree[new]
    s.loc[check, "needs_review"] = True
    s.loc[new, "tier"] = "silver" if silver else "gold"
    s["tier"] = s["tier"].fillna("gold")
    s["needs_review"] = s["needs_review"].astype(bool)
    s.to_csv(ROOT / "labeling/label_sample.csv", index=False)
    print(f"{len(s)} rows; agreement {agree.mean():.1%}; new rows: {new.sum()}, agreement {agree[new].mean():.1%}, "
          f"{(~agree & new).sum()} disagreements + {len(check)} random agreements to review")
    print(pd.crosstab(s["label_ai1"], s["label_ai2"], rownames=["annotator 1"], colnames=["annotator 2"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-per-label", type=int, default=10)
    ap.add_argument("--silver", action="store_true", help="new rows are silver labels (training only)")
    a = ap.parse_args()
    main(a.check_per_label, silver=a.silver)

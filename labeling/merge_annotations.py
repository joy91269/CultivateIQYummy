"""Attach the two independent AI annotations to the sample and pick the rows a person must review.

Input
    labeling/label_sample.csv
    labeling/annotations/annotator1_*.jsonl   {"obs_id", "label", "reason"} per line
    labeling/annotations/annotator2_*.jsonl
Output (written back to labeling/label_sample.csv)
    label_ai1, reason_ai1, label_ai2, reason_ai2, needs_review
needs_review = every disagreement + a random check of agreements (up to 10 per agreed label).

Usage: python labeling/merge_annotations.py
"""
from __future__ import annotations

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


def main(check_per_label: int = 10, seed: int = 2026) -> None:
    s = pd.read_csv(ROOT / "labeling/label_sample.csv")
    a1, a2 = read("annotator1"), read("annotator2")
    s["label_ai1"], s["reason_ai1"] = s["obs_id"].map(a1["label"]), s["obs_id"].map(a1["reason"])
    s["label_ai2"], s["reason_ai2"] = s["obs_id"].map(a2["label"]), s["obs_id"].map(a2["reason"])
    missing = s["label_ai1"].isna() | s["label_ai2"].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum()} rows lack an annotation")
    agree = s["label_ai1"] == s["label_ai2"]
    check = (s[agree].groupby("label_ai1", group_keys=False)
             .apply(lambda g: g.sample(min(check_per_label, len(g)), random_state=seed)).index)
    s["needs_review"] = ~agree
    s.loc[check, "needs_review"] = True
    s.to_csv(ROOT / "labeling/label_sample.csv", index=False)
    print(f"{len(s)} rows; agreement {agree.mean():.1%}; {(~agree).sum()} disagreements + {len(check)} random "
          f"agreements = {s['needs_review'].sum()} rows to review")
    print(pd.crosstab(s["label_ai1"], s["label_ai2"], rownames=["annotator 1"], colnames=["annotator 2"]))


if __name__ == "__main__":
    main()

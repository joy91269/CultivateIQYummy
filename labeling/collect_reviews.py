"""Merge the two AI annotations and the human review into the final label file.

Input
    labeling/label_sample.csv  sample with label_ai1/reason_ai1 (annotator 1), label_ai2/reason_ai2 (annotator 2,
                               labelled independently without seeing annotator 1) and needs_review
    labeling/reviews.json      export of the review page's store: {"<obs_id>": {"label": ..., "note": ..., "recheck": ...}}
    labeling/reviews_first_pass.json  the same export before the reviewer's second look at demand vs other
    labeling/adjudication.json  blind decisions on texts where Codex and ChatGPT agreed with each other, not with us
Output
    labeling/labels.csv        final `label` and where it came from (`label_source`):
                               "human review"            a person decided (all disagreements + random agreements)
                               "AI agreement"            both AI annotators agreed and the row was not drawn for review
                               "AI annotator 1 (unreviewed disagreement)"  only if a flagged row was not reviewed

Usage: python labeling/collect_reviews.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["price", "supply", "demand", "other"]


def main() -> None:
    s = pd.read_csv(ROOT / "labeling/label_sample.csv")
    path = ROOT / "labeling/reviews.json"
    rev = json.loads(path.read_text()) if path.exists() else {}
    s["ai_agree"] = s["label_ai1"] == s["label_ai2"]
    s["reviewed"] = s["obs_id"].isin(rev)
    s["human_label"] = s["obs_id"].map(lambda o: rev.get(o, {}).get("label"))
    s["note"] = s["obs_id"].map(lambda o: rev.get(o, {}).get("note", ""))
    first_path = ROOT / "labeling/reviews_first_pass.json"
    first = json.loads(first_path.read_text()) if first_path.exists() else {}
    s["rechecked"] = s["obs_id"].map(lambda o: bool(rev.get(o, {}).get("recheck")))
    s["first_pass_label"] = s["obs_id"].map(lambda o: first.get(o, {}).get("label"))
    s["label"] = s["human_label"].where(s["reviewed"], s["label_ai1"])
    s["label_source"] = "AI agreement"
    s.loc[s["reviewed"], "label_source"] = "human review"
    s.loc[~s["reviewed"] & ~s["ai_agree"], "label_source"] = "AI annotator 1 (unreviewed disagreement)"
    # blind adjudication of texts where both annotators from the other model family (Codex, ChatGPT) agreed with each
    # other but not with our label: the person's decision replaces the label (labeling/adjudication.json)
    adj_path = ROOT / "labeling/adjudication.json"
    adj = json.loads(adj_path.read_text()) if adj_path.exists() else {}
    s["adjudicated"] = s["obs_id"].isin(adj)
    s["label_before_adjudication"] = s["label"].where(s["adjudicated"])
    s.loc[s["adjudicated"], "label"] = s.loc[s["adjudicated"], "obs_id"].map(lambda o: adj[o]["label"])
    s.loc[s["adjudicated"], "label_source"] = "human adjudication (cross-family disagreement)"
    if "tier" not in s:
        s["tier"] = "gold"
    # silver rows where the AI annotators disagree are not used at all
    drop = (s["tier"] == "silver") & ~s["reviewed"] & ~s["ai_agree"]
    s.loc[drop, "label"] = pd.NA
    s.loc[drop, "label_source"] = "dropped (silver, AI annotators disagree)"
    if (~s["label"].isin(LABELS) & ~drop).any():
        raise SystemExit("unknown label found")
    cols = ["obs_id", "batch", "tier", "product", "record_type", "subreddit", "date", "thread_id", "url", "label_ai1",
            "reason_ai1", "label_ai2", "reason_ai2", "ai_agree", "review_round", "needs_review", "reviewed", "first_pass_label", "rechecked", "human_label",
            "adjudicated", "label_before_adjudication", "label",
            "label_source", "note"]
    s[cols].to_csv(ROOT / "labeling/labels.csv", index=False)
    print(f"{len(s)} rows; AI annotators agree on {s['ai_agree'].mean():.1%}; {s['reviewed'].sum()} reviewed by a person")
    print(s["label_source"].value_counts().to_string())
    print(pd.crosstab(s["product"], s["label"]).reindex(columns=LABELS))


if __name__ == "__main__":
    main()

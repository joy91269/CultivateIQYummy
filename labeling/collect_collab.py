"""Agreement between the second human annotator (collaborator) and the first reviewer.

Inputs
    labeling/collab.json        export of the collaborator page's store: {"<person>_<obs_id>": {"obs_id", "label", "by"}}
    labeling/collab_sample.csv  the 80 texts shown (set: adjudicated / silver check / earlier review)
    labeling/labels.csv, labeling/adjudication.json   the first reviewer's decisions
Output
    outputs/tables/human_agreement.csv   percent agreement and Cohen's kappa with the first reviewer, per set, for
                                         the second annotator and, for comparison, the four AI annotators

Usage: python labeling/collect_collab.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    collab = pd.DataFrame(json.loads((ROOT / "labeling/collab.json").read_text()).values())
    sample = pd.read_csv(ROOT / "labeling/collab_sample.csv")
    lab = pd.read_csv(ROOT / "labeling/labels.csv").set_index("obs_id")
    adj = json.loads((ROOT / "labeling/adjudication.json").read_text())
    first = {o: (adj[o]["label"] if o in adj else lab.loc[o, "human_label"]) for o in sample["obs_id"]}
    rows = []
    for person, g in collab.groupby("by"):
        g = g.merge(sample, on="obs_id")
        g["reviewer"] = g["obs_id"].map(first)
        for name, part in [("all", g)] + list(g.groupby("set")):
            rows.append({"annotator": person, "set": name, "n": len(part),
                         "agreement": round((part["label"] == part["reviewer"]).mean(), 4),
                         "cohen_kappa": round(cohen_kappa_score(part["label"], part["reviewer"]), 4)})
        print(f"{person}: {len(g)} texts")
        print(pd.crosstab(g["reviewer"], g["label"], rownames=["first reviewer"], colnames=["second annotator"]))
    # The four AI annotators against the first reviewer on the same texts (the OpenAI annotators labelled gold only)
    ai = {"claude_opus": lab["label_ai1"].to_dict(), "claude_sonnet": lab["label_ai2"].to_dict()}
    for name, f in [("codex", "annotator3_codex.jsonl"), ("chatgpt", "annotator4_chatgpt.jsonl")]:
        lines = (ROOT / "labeling/annotations" / f).read_text().splitlines()
        ai[name] = {json.loads(x)["obs_id"]: json.loads(x)["label"] for x in lines if x.strip()}
    s = sample.assign(reviewer=sample["obs_id"].map(first))
    for name, labels in ai.items():
        for set_name, part in [("all", s)] + list(s.groupby("set")):
            part = part.assign(ai=part["obs_id"].map(labels)).dropna(subset=["ai"])
            if len(part):
                rows.append({"annotator": name, "set": set_name, "n": len(part),
                             "agreement": round((part["ai"] == part["reviewer"]).mean(), 4),
                             "cohen_kappa": round(cohen_kappa_score(part["ai"], part["reviewer"]), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs/tables/human_agreement.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

"""Agreement of the independent annotators from other model families with the two Claude annotators and the final labels.

Inputs
    labeling/labels.csv                         final labels (gold tier: rounds 1-3)
    labeling/codex_task/output_*.jsonl          annotator 3: OpenAI Codex, labelled blind with labeling/codex_task/TASK.md
    labeling/chatgpt_task/output_*.jsonl        annotator 4 (optional): ChatGPT, same instructions
Outputs
    outputs/tables/annotator_agreement.csv      pairwise Cohen's kappa and percent agreement, Fleiss' kappa
    labeling/annotations/annotator3_codex.jsonl (and annotator4_chatgpt.jsonl) copies in the annotations folder

Usage: python labeling/third_annotator.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["price", "supply", "demand", "other"]
EXTERNAL = {"codex": ("labeling/codex_task", "annotator3_codex"), "chatgpt": ("labeling/chatgpt_task", "annotator4_chatgpt")}


def read_outputs(folder: Path) -> pd.Series:
    rows = []
    for f in sorted(folder.glob("output_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        return pd.Series(dtype=object)
    df = pd.DataFrame(rows).drop_duplicates("obs_id", keep="last")
    df["label"] = df["label"].str.strip().str.lower()
    bad = ~df["label"].isin(LABELS)
    if bad.any():
        raise SystemExit(f"{folder}: {bad.sum()} invalid labels")
    return df.set_index("obs_id")["label"]


def fleiss_kappa(table: np.ndarray) -> float:
    """table: items x categories, counts of raters choosing each category (same number of raters per item)."""
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    p_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e)


def main() -> None:
    lab = pd.read_csv(ROOT / "labeling/labels.csv")
    if "tier" in lab:
        lab = lab[lab["tier"].fillna("gold") == "gold"]
    raters = {"claude_opus": lab.set_index("obs_id")["label_ai1"], "claude_sonnet": lab.set_index("obs_id")["label_ai2"]}
    for name, (folder, stem) in EXTERNAL.items():
        lbl = read_outputs(ROOT / folder)
        if len(lbl):
            raters[name] = lbl
            src = ROOT / folder
            out = ROOT / "labeling/annotations" / f"{stem}.jsonl"
            with open(out, "w") as f:
                for p in sorted(src.glob("output_*.jsonl")):
                    f.write(p.read_text())
            print(f"{name}: {len(lbl)} labels -> {out.relative_to(ROOT)}")
    final = lab.set_index("obs_id")["label"]
    ids = final.index
    for k, v in raters.items():
        ids = ids.intersection(v.index)
    print(f"{len(ids)} gold texts labelled by all {len(raters)} annotators")
    rows = []
    for a, b in itertools.combinations(list(raters) + ["final"], 2):
        x = (raters[a] if a != "final" else final).reindex(ids)
        y = (raters[b] if b != "final" else final).reindex(ids)
        rows.append({"pair": f"{a} vs {b}", "agreement": round((x == y).mean(), 4),
                     "cohen_kappa": round(cohen_kappa_score(x, y), 4), "n": len(ids)})
    table = np.stack([np.stack([(raters[r].reindex(ids) == l).to_numpy() for l in LABELS], 1) for r in raters]).sum(0)
    rows.append({"pair": f"Fleiss kappa, {len(raters)} AI annotators", "agreement": np.nan,
                 "cohen_kappa": round(fleiss_kappa(table.astype(float)), 4), "n": len(ids)})
    res = pd.DataFrame(rows)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "outputs/tables/annotator_agreement.csv", index=False)
    print(res.to_string(index=False))
    for name in [r for r in raters if r in EXTERNAL]:
        dis = final.reindex(ids) != raters[name].reindex(ids)
        print(f"\n{name} disagrees with the final label on {dis.sum()} texts:")
        print(pd.crosstab(final.reindex(ids)[dis], raters[name].reindex(ids)[dis], rownames=["final"], colnames=[name]))


if __name__ == "__main__":
    main()

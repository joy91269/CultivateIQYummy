"""Train, evaluate and apply the market-signal classifiers; score sentiment (Task 2, compute-heavy part).

Usage (from the repository root):
    python -m analysis.run_models evaluate   # thread-grouped 5-fold CV of all four models on the labelled sample
    python -m analysis.run_models predict    # train the selected model on all labels, classify every observation
    python -m analysis.run_models sentiment  # sentiment score for every observation
    python -m analysis.run_models all

Outputs
    outputs/tables/model_cv_metrics.csv          one row per model (accuracy, macro-F1 with 95% CI, per-class F1)
    outputs/tables/model_cv_predictions.csv      out-of-fold predictions of every model for every labelled row
    data/processed/classified_observations.csv.gz  obs_id, predicted label, class probabilities, sentiment
Runs on CPU; the slowest step (sentiment for every observation) takes about an hour on 2 cores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.classify import (LABELS, embed, embed_lr_fit, finetune_fit, finetune_predict, grouped_folds,
                               keyword_predict, metrics, model_input, sentiment_scores, set_threads,
                               zeroshot_predict)

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/processed/event_dataset.csv"
LABELS_CSV = ROOT / "labeling/labels.csv"
TABLES = ROOT / "outputs/tables"
CLASSIFIED = ROOT / "data/processed/classified_observations.csv.gz"
CACHE = ROOT / ".cache"
FINAL_MODEL_FILE = ROOT / "outputs/tables/selected_model.json"


def load_labelled() -> pd.DataFrame:
    lab = pd.read_csv(LABELS_CSV)
    lab = lab[lab["label"].isin(LABELS)].copy()
    ev = pd.read_csv(EVENTS, usecols=["obs_id", "thread_id", "event_types", "text", "product"], low_memory=False)
    lab = lab.drop(columns=[c for c in ["thread_id", "event_types", "text", "product"] if c in lab]).merge(
        ev, on="obs_id", how="inner")
    lab["input"] = [model_input(p, t) for p, t in zip(lab["product"], lab["text"])]
    return lab.reset_index(drop=True)


def evaluate() -> None:
    lab = load_labelled()
    y, groups = lab["label"].to_numpy(), lab["thread_id"].astype(str).to_numpy()
    folds = grouped_folds(groups, y)
    preds = pd.DataFrame({"obs_id": lab["obs_id"], "label": y, "thread_id": groups})

    preds["keyword"] = keyword_predict(lab["event_types"])
    print("zero-shot ...", flush=True)
    preds["zeroshot"] = zeroshot_predict(lab["input"].tolist())

    print("embeddings + logistic regression ...", flush=True)
    X = embed(lab["input"].tolist())
    oof = np.empty(len(lab), dtype=object)
    for tr, te in folds:
        oof[te] = embed_lr_fit(X[tr], y[tr]).predict(X[te])
    preds["embed_lr"] = oof

    print("fine-tuned DistilRoBERTa ...", flush=True)
    oof = np.empty(len(lab), dtype=object)
    for k, (tr, te) in enumerate(folds):
        tok, model = finetune_fit(lab["input"].iloc[tr].tolist(), y[tr].tolist(), str(CACHE / f"ft_fold{k}"))
        oof[te], _ = finetune_predict(tok, model, lab["input"].iloc[te].tolist())
        print(f"  fold {k + 1}/{len(folds)} done", flush=True)
    preds["finetune"] = oof

    rows = []
    for m in ["keyword", "zeroshot", "embed_lr", "finetune"]:
        r = metrics(y, preds[m].to_numpy(), groups)
        r["model"] = m
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "model_cv_metrics.csv")
    preds.to_csv(TABLES / "model_cv_predictions.csv", index=False)
    best = res["macro_f1"].idxmax()
    FINAL_MODEL_FILE.write_text(json.dumps({"selected_model": best, "selection_metric": "macro_f1 (grouped 5-fold CV)"}))
    print(res.round(3).to_string())
    print("selected:", best)


def predict() -> None:
    lab = load_labelled()
    best = json.loads(FINAL_MODEL_FILE.read_text())["selected_model"]
    ev = pd.read_csv(EVENTS, usecols=["obs_id", "product", "text", "event_types"], low_memory=False)
    inputs = [model_input(p, t) for p, t in zip(ev["product"], ev["text"])]
    if best == "finetune":
        tok, model = finetune_fit(lab["input"].tolist(), lab["label"].tolist(), str(CACHE / "ft_final"))
        pred, prob = finetune_predict(tok, model, inputs)
    elif best == "embed_lr":
        clf = embed_lr_fit(embed(lab["input"].tolist()), lab["label"].to_numpy())
        prob = clf.predict_proba(embed(inputs))
        prob = prob[:, [list(clf.classes_).index(l) for l in LABELS]]
        pred = np.array(LABELS)[prob.argmax(1)]
    elif best == "zeroshot":
        pred, prob = zeroshot_predict(inputs), np.full((len(inputs), len(LABELS)), np.nan)
    else:
        pred, prob = keyword_predict(ev["event_types"]), np.full((len(inputs), len(LABELS)), np.nan)
    out = pd.DataFrame({"obs_id": ev["obs_id"], "pred_label": pred, "model": best})
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = prob[:, i].round(4)
    # rows of the labelled sample keep their human label for the analysis
    out = out.merge(lab[["obs_id", "label"]].rename(columns={"label": "human_label"}), on="obs_id", how="left")
    if CLASSIFIED.exists():   # keep an existing sentiment column
        old = pd.read_csv(CLASSIFIED)
        if "sentiment" in old:
            out = out.merge(old[["obs_id", "sentiment"]], on="obs_id", how="left")
    out.to_csv(CLASSIFIED, index=False, compression="gzip")
    print(out["pred_label"].value_counts().to_string())


def sentiment(chunk: int = 4000) -> None:
    """Sentiment per Reddit record (texts shared by several products are scored once). Resumable: finished
    chunks are cached in .cache/sentiment_scores.csv."""
    ev = pd.read_csv(EVENTS, usecols=["obs_id", "record_id", "text"], low_memory=False)
    rec = ev.drop_duplicates("record_id")[["record_id", "text"]].reset_index(drop=True)
    cache = CACHE / "sentiment_scores.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    scores = dict(pd.read_csv(cache).itertuples(index=False)) if cache.exists() else {}
    todo = rec[~rec["record_id"].isin(scores)]
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        scores.update(zip(part["record_id"], sentiment_scores(part["text"].tolist()).round(4)))
        pd.DataFrame({"record_id": list(scores), "sentiment": list(scores.values())}).to_csv(cache, index=False)
        print(f"sentiment: {min(i + chunk, len(todo))}/{len(todo)} records", flush=True)
    s = ev[["obs_id", "record_id"]].assign(sentiment=ev["record_id"].map(scores))
    out = pd.read_csv(CLASSIFIED) if CLASSIFIED.exists() else pd.DataFrame({"obs_id": ev["obs_id"]})
    out = out.drop(columns=["sentiment"], errors="ignore").merge(s[["obs_id", "sentiment"]], on="obs_id", how="right")
    out.to_csv(CLASSIFIED, index=False, compression="gzip")
    print(out["sentiment"].describe().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["evaluate", "predict", "sentiment", "all"])
    a = ap.parse_args()
    set_threads()
    if a.step in ("evaluate", "all"):
        evaluate()
    if a.step in ("predict", "all"):
        predict()
    if a.step in ("sentiment", "all"):
        sentiment()

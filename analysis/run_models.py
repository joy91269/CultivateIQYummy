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

from analysis.classify import (EMBED_MODELS, LABELS, embed, prior_adjust, embed_lr_fit, finetune_fit, finetune_predict, grouped_folds,
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
    # texts come from the sample file itself, so the labelled set does not depend on later rebuilds of the dataset
    # (two labelled comments were posted after 2024 and are not in the final event dataset)
    smp = pd.read_csv(ROOT / "labeling/label_sample.csv", usecols=["obs_id", "thread_id", "event_types", "text", "product"])
    lab = lab.drop(columns=[c for c in ["thread_id", "event_types", "text", "product"] if c in lab]).merge(
        smp, on="obs_id", how="inner")
    lab["input"] = [model_input(p, t) for p, t in zip(lab["product"], lab["text"])]
    return lab.reset_index(drop=True)


def cached_zeroshot(lab: pd.DataFrame) -> np.ndarray:
    """Zero-shot labels do not depend on the training labels, so they are cached by obs_id."""
    path = CACHE / "zeroshot_predictions.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    have = dict(pd.read_csv(path).itertuples(index=False)) if path.exists() else {}
    todo = lab[~lab["obs_id"].isin(have)]
    if len(todo):
        have.update(zip(todo["obs_id"], zeroshot_predict(todo["input"].tolist())))
        pd.DataFrame({"obs_id": list(have), "zeroshot": list(have.values())}).to_csv(path, index=False)
    return lab["obs_id"].map(have).to_numpy()


def cached_embeddings(lab: pd.DataFrame, key: str = "embed_lr") -> np.ndarray:
    """Sentence embeddings of the labelled texts for one of EMBED_MODELS, cached by obs_id."""
    stem = "embeddings" if key == "embed_lr" else f"embeddings_{key}"
    path, ids_path = CACHE / f"{stem}.npy", CACHE / f"{stem}_ids.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    if path.exists():
        X, ids = np.load(path), pd.read_csv(ids_path)["obs_id"].tolist()
    else:
        X, ids = None, []
    pos = {o: i for i, o in enumerate(ids)}
    todo = lab[~lab["obs_id"].isin(pos)]
    if len(todo):
        new = embed(todo["input"].tolist(), model_name=EMBED_MODELS[key])
        X = new if X is None else np.vstack([X, new])
        ids = ids + todo["obs_id"].tolist()
        np.save(path, X)
        pd.DataFrame({"obs_id": ids}).to_csv(ids_path, index=False)
        pos = {o: i for i, o in enumerate(ids)}
    return X[[pos[o] for o in lab["obs_id"]]]


def evaluate() -> None:
    lab = load_labelled()
    y, groups = lab["label"].to_numpy(), lab["thread_id"].astype(str).to_numpy()
    folds = grouped_folds(groups, y)
    preds = pd.DataFrame({"obs_id": lab["obs_id"], "label": y, "thread_id": groups})

    preds["keyword"] = keyword_predict(lab["event_types"])
    print("zero-shot ...", flush=True)
    preds["zeroshot"] = cached_zeroshot(lab)

    for key, name in EMBED_MODELS.items():
        print(f"embeddings ({name}) + logistic regression ...", flush=True)
        X = cached_embeddings(lab, key)
        oof = np.empty(len(lab), dtype=object)
        prob = np.zeros((len(lab), len(LABELS)))
        chosen = []
        for tr, te in folds:
            clf = embed_lr_fit(X[tr], y[tr], groups[tr])
            oof[te] = clf.predict(X[te])
            prob[te] = clf.predict_proba(X[te])[:, [list(clf.classes_).index(l) for l in LABELS]]
            chosen.append(clf.chosen_C_)
        preds[key] = oof
        for i, l in enumerate(LABELS):
            preds[f"{key}_p_{l}"] = prob[:, i].round(4)
        print(f"  C chosen in the inner CV of each fold: {chosen}", flush=True)

    print("fine-tuned DistilRoBERTa ...", flush=True)
    # each fold takes ~25 min on a 2-core CPU, so finished folds are cached; the folder name records the label set
    # (number of texts and of each label) so that new labels never reuse old folds
    sig = f"n{len(y)}_" + "_".join(f"{l}{(y == l).sum()}" for l in LABELS)
    fold_dir = CACHE / f"ft_cv_{sig}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    oof = np.empty(len(lab), dtype=object)
    for k, (tr, te) in enumerate(folds):
        path = fold_dir / f"fold{k}.csv"
        if path.exists():
            done = pd.read_csv(path)
            if set(done["obs_id"]) == set(lab["obs_id"].iloc[te]):
                oof[te] = lab["obs_id"].iloc[te].map(dict(zip(done["obs_id"], done["pred"]))).to_numpy()
                print(f"  fold {k + 1}/{len(folds)} cached", flush=True)
                continue
        tok, model = finetune_fit(lab["input"].iloc[tr].tolist(), y[tr].tolist())
        oof[te], _ = finetune_predict(tok, model, lab["input"].iloc[te].tolist())
        pd.DataFrame({"obs_id": lab["obs_id"].iloc[te], "pred": oof[te]}).to_csv(path, index=False)
        print(f"  fold {k + 1}/{len(folds)} done", flush=True)
    preds["finetune"] = oof

    rows = []
    for m in ["keyword", "zeroshot", *EMBED_MODELS, "finetune"]:
        r = metrics(y, preds[m].to_numpy(), groups)
        r["model"] = m
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    # calibration of the class probabilities: summed out-of-fold probabilities vs. true counts
    true_n = pd.Series(y).value_counts()
    for key in EMBED_MODELS:
        for l in LABELS:
            res.loc[key, f"expected_{l}_vs_true"] = f"{preds[f'{key}_p_{l}'].sum():.0f} / {true_n.get(l, 0)}"
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "model_cv_metrics.csv")
    preds.to_csv(TABLES / "model_cv_predictions.csv", index=False)
    best = res["macro_f1"].idxmax()
    FINAL_MODEL_FILE.write_text(json.dumps({"selected_model": best, "selection_metric": "macro_f1 (grouped 5-fold CV)"}))
    print(res.round(3).to_string())
    print("selected:", best)


def predict_in_chunks(ids: list[str], inputs: list[str], score, cache_path, chunk: int = 5000) -> np.ndarray:
    """Class probabilities for every input, computed chunk by chunk and cached so an interrupted run resumes.
    Delete the cache file after changing the labels or the model."""
    have = pd.read_csv(cache_path).set_index("obs_id") if cache_path.exists() else pd.DataFrame(columns=LABELS)
    todo = [i for i, o in enumerate(ids) if o not in have.index]
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        p = score([inputs[i] for i in idx])
        part = pd.DataFrame(p, columns=LABELS, index=pd.Index([ids[i] for i in idx], name="obs_id"))
        part.round(5).to_csv(cache_path, mode="a", header=not cache_path.exists())
        have = pd.concat([have, part])
        print(f"predict: {min(start + chunk, len(todo))}/{len(todo)}", flush=True)
    return have.loc[ids, LABELS].to_numpy(dtype=float)


def embed_all(ids: list[str], inputs: list[str], key: str, chunk: int = 5000) -> np.ndarray:
    """Embeddings of every observation, computed in chunks and cached (.cache/embed_all_<key>/), so an interrupted
    run resumes and a new classifier on the same embeddings needs no re-embedding."""
    d = CACHE / f"embed_all_{key}"
    d.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, start in enumerate(range(0, len(ids), chunk)):
        f, fid = d / f"chunk{k:03d}.npy", d / f"chunk{k:03d}_ids.csv"
        if f.exists() and pd.read_csv(fid)["obs_id"].tolist() == ids[start:start + chunk]:
            parts.append(np.load(f))
            continue
        X = embed(inputs[start:start + chunk], model_name=EMBED_MODELS[key])
        np.save(f, X)
        pd.DataFrame({"obs_id": ids[start:start + chunk]}).to_csv(fid, index=False)
        parts.append(X)
        print(f"embedded {min(start + chunk, len(ids))}/{len(ids)}", flush=True)
    return np.vstack(parts)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def predict() -> None:
    lab = load_labelled()
    best = json.loads(FINAL_MODEL_FILE.read_text())["selected_model"]
    ev = pd.read_csv(EVENTS, usecols=["obs_id", "product", "text", "event_types"], low_memory=False)
    inputs = [model_input(p, t) for p, t in zip(ev["product"], ev["text"])]
    if best == "finetune" or best in EMBED_MODELS:
        if best == "finetune":
            model_dir = CACHE / "ft_final"
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if (model_dir / "config.json").exists():
                tok, model = AutoTokenizer.from_pretrained(model_dir), \
                    AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
            else:
                tok, model = finetune_fit(lab["input"].tolist(), lab["label"].tolist())
                tok.save_pretrained(model_dir)
                model.save_pretrained(model_dir)
            score = lambda xs: finetune_predict(tok, model, xs)[1]
        if best == "finetune":
            prob = predict_in_chunks(ev["obs_id"].tolist(), inputs, score, CACHE / "predict_probs_finetune.csv")
        else:
            clf = embed_lr_fit(cached_embeddings(lab, best), lab["label"].to_numpy(),
                               lab["thread_id"].astype(str).to_numpy())
            print(f"final model: C = {clf.chosen_C_}", flush=True)
            Xall = embed_all(ev["obs_id"].tolist(), inputs, best)
            prob = clf.predict_proba(Xall)[:, [list(clf.classes_).index(l) for l in LABELS]]
        pred = np.array(LABELS)[prob.argmax(1)]
    elif best == "zeroshot":
        pred, prob = zeroshot_predict(inputs), np.full((len(inputs), len(LABELS)), np.nan)
    else:
        pred, prob = keyword_predict(ev["event_types"]), np.full((len(inputs), len(LABELS)), np.nan)
    out = pd.DataFrame({"obs_id": ev["obs_id"], "product": ev["product"], "pred_label": pred, "model": best})
    for i, l in enumerate(LABELS):
        out[f"p_{l}_raw"] = prob[:, i].round(4)
    if not np.isnan(prob).all():
        # the labelled sample is enriched with market talk, so the classifier expects more price/supply/demand
        # than the full dataset holds: re-estimate each product's label shares by EM and rescale the probabilities
        rows = []
        adj = np.zeros_like(prob)
        for prod in sorted(ev["product"].unique()):
            m = (ev["product"] == prod).to_numpy()
            tr = lab.loc[lab["product"] == prod, "label"].value_counts(normalize=True).reindex(LABELS).fillna(0)
            adj[m], pi = prior_adjust(prob[m], tr.to_numpy().clip(1e-3))
            rnd = lab[(lab["product"] == prod) & (lab["batch"] == "random")]
            for i, l in enumerate(LABELS):
                k, n = int((rnd["label"] == l).sum()), len(rnd)
                lo, hi = wilson(k, n)
                rows.append({"product": prod, "label": l, "share_in_labelled_sample": round(tr[l], 4),
                             "mean_probability_before_adjustment": round(prob[m, i].mean(), 4),
                             "estimated_share_after_adjustment": round(pi[i], 4),
                             "random_sample_share": round(k / n, 4) if n else np.nan,
                             "random_sample_ci95": f"{lo:.3f}-{hi:.3f}" if n else "", "random_sample_n": n})
        pd.DataFrame(rows).to_csv(TABLES / "label_prevalence.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        prob = adj
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = prob[:, i].round(4)
    out["pred_label"] = np.array(LABELS)[prob.argmax(1)] if not np.isnan(prob).all() else out["pred_label"]
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

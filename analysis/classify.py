"""Market-signal classifiers (price / supply / demand / other) and sentiment scoring.

Models compared (all evaluated with thread-grouped cross-validation on the manually labelled sample, so that a
post and the comments under it never sit on both sides of a train/test split):
  1. keyword   - the Task 1 keyword dictionary mapped to the four classes (no training)
  2. zeroshot  - facebook/bart-large-mnli zero-shot NLI classifier (no training)
  3. embed_lr* - frozen sentence embeddings + logistic regression, with three embedding models (EMBED_MODELS):
                 all-mpnet-base-v2 (2021), bge-base-en-v1.5 (2023) and gte-modernbert-base (2025)
  4. finetune  - distilroberta-base fine-tuned end to end on the labelled sample

Each text is shown to the models together with the product it is about ("Product: onion. ..."), because one
comment can mention several products and the label is product specific.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

LABELS = ["price", "supply", "demand", "other"]
KEYWORD_TO_LABEL = {
    "price_change": "price",
    "supply_shortage": "supply", "weather_event": "supply", "food_safety_recall": "supply",
    "demand_change": "demand", "marketing_promotion": "demand",
}
SEED = 13


def model_input(product: str, text: str, max_chars: int = 1200) -> str:
    text = " ".join(str(text).split())
    return f"Product: {product}. {text[:max_chars]}"


# ------------------------------------------------------------------------------------------- 1. keywords
def keyword_predict(event_types: pd.Series) -> np.ndarray:
    out = []
    for et in event_types.fillna("none"):
        labels = {KEYWORD_TO_LABEL[t] for t in et.split("|") if t in KEYWORD_TO_LABEL}
        out.append(next((l for l in ["price", "supply", "demand"] if l in labels), "other"))
    return np.array(out)


# ------------------------------------------------------------------------------------------- 2. zero-shot
ZS_HYPOTHESES = {
    "price": "the price, cost or affordability of the product",
    "supply": "the supply, availability, shortage, harvest, recall or quality problems of the product",
    "demand": "people wanting, buying, craving or trending interest in the product",
    "other": "something unrelated to the market for the product",
}


def zeroshot_predict(inputs: list[str], batch_size: int = 8) -> np.ndarray:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    names = list(ZS_HYPOTHESES.values())
    back = {v: k for k, v in ZS_HYPOTHESES.items()}
    res = clf(inputs, candidate_labels=names, hypothesis_template="This text is about {}.",
              batch_size=batch_size, truncation=True)
    return np.array([back[r["labels"][0]] for r in res])


# ------------------------------------------------------------------------------------------- 3. embeddings
EMBED_MODELS = {"embed_lr": "sentence-transformers/all-mpnet-base-v2",      # 110 M parameters, 2021
                "embed_lr_bge": "BAAI/bge-base-en-v1.5",                    # 110 M, 2023
                "embed_lr_gte": "Alibaba-NLP/gte-modernbert-base"}          # 149 M, ModernBERT encoder, 2025
EMBED_MODEL = EMBED_MODELS["embed_lr"]
_EMBEDDERS: dict = {}     # loaded embedding models, reused across calls


def embed(inputs: list[str], batch_size: int = 32, model_name: str = EMBED_MODEL) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    # some checkpoints are stored in float16, which is slow and less precise on CPU: always run in float32
    if model_name not in _EMBEDDERS:
        _EMBEDDERS[model_name] = SentenceTransformer(model_name, device="cpu", model_kwargs={"dtype": torch.float32})
    model = _EMBEDDERS[model_name]
    return model.encode(inputs, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


C_GRID = (0.5, 2.0, 8.0, 32.0)


def embed_lr_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Logistic regression on frozen embeddings. The regularisation strength C is chosen by an inner thread-grouped
    3-fold CV on the training data only (macro-F1), so the outer CV score is not tuned on its own test folds.
    No class weights: weighting the rare classes up raises their recall but makes the predicted probabilities
    overstate them; unweighted probabilities are calibrated, so summing them gives unbiased weekly counts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y, groups))
    best_c, best_f = C_GRID[0], -1.0
    for c in C_GRID:
        pred = np.empty(len(y), dtype=object)
        for tr, te in inner:
            pred[te] = LogisticRegression(C=c, max_iter=3000, random_state=SEED).fit(X[tr], y[tr]).predict(X[te])
        f = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
        if f > best_f:
            best_c, best_f = c, f
    clf = LogisticRegression(C=best_c, max_iter=3000, random_state=SEED).fit(X, y)
    clf.chosen_C_ = best_c
    return clf


# ------------------------------------------------------------------------------------------- 4. fine-tune
FT_MODEL = "distilroberta-base"


def finetune_fit(train_inputs: list[str], train_labels: list[str], out_dir: str | None = None, epochs: int = 4,
                 batch_size: int = 16, lr: float = 4e-5, max_length: int = 192):
    """Fine-tune DistilRoBERTa with a plain PyTorch loop: AdamW, linear warm-up/decay, class-weighted loss.
    (`out_dir` is unused and kept for interface compatibility.)"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FT_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    y = np.array([LABELS.index(l) for l in train_labels])
    counts = np.bincount(y, minlength=len(LABELS))
    weights = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(y) / batch_size))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    model.train()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / batch_size))):
            enc = tok([train_inputs[i] for i in b], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt")
            logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y[b]), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model


def predict_proba_sorted(tok, model, inputs: list[str], max_length: int, batch_size: int = 32) -> np.ndarray:
    """Softmax probabilities, batching texts of similar length together (much less padding on CPU)."""
    import torch
    model.eval()
    order = np.argsort([len(t) for t in inputs])
    out = None
    with torch.inference_mode():
        for i in range(0, len(inputs), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([inputs[j] for j in idx], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt")
            p = torch.softmax(model(**enc).logits, dim=-1).numpy()
            if out is None:
                out = np.zeros((len(inputs), p.shape[1]))
            out[idx] = p
    return out


def finetune_predict(tok, model, inputs: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    p = predict_proba_sorted(tok, model, inputs, max_length=192, batch_size=batch_size)
    return np.array(LABELS)[p.argmax(1)], p


# ------------------------------------------------------------------------------------------- evaluation
def grouped_folds(groups: np.ndarray, labels: np.ndarray, n_splits: int = 5):
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(groups, labels, groups))


def metrics(y_true, y_pred, groups=None, n_boot: int = 1000) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
           "cohen_kappa": cohen_kappa_score(y_true, y_pred)}
    for l, f in zip(LABELS, f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)):
        out[f"f1_{l}"] = f
    if groups is not None:   # cluster bootstrap over threads for a 95% CI of macro-F1
        rng = np.random.default_rng(SEED)
        g = np.asarray(groups)
        uniq = np.unique(g)
        idx_by_g = {k: np.where(g == k)[0] for k in uniq}
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by_g[k] for k in rng.choice(uniq, len(uniq), replace=True)])
            boots.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0))
        out["macro_f1_ci_low"], out["macro_f1_ci_high"] = np.percentile(boots, [2.5, 97.5])
    return out


# ------------------------------------------------------------------------------------------- sentiment
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_SENT = None


def sentiment_scores(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Returns P(positive) - P(negative) in [-1, 1] for each text."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    global _SENT
    if _SENT is None:   # load once per process
        _SENT = (AutoTokenizer.from_pretrained(SENT_MODEL),
                 AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).eval())
    tok, model = _SENT
    id2 = {i: l.lower() for i, l in model.config.id2label.items()}
    pos = [k for k, v in id2.items() if v.startswith("pos")][0]
    neg = [k for k, v in id2.items() if v.startswith("neg")][0]
    p = predict_proba_sorted(tok, model, [" ".join(str(t).split())[:600] for t in texts], max_length=128,
                             batch_size=batch_size)
    return p[:, pos] - p[:, neg]


def set_threads(n: int | None = None) -> None:
    import torch
    torch.set_num_threads(n or os.cpu_count() or 1)


# ------------------------------------------------------------------------------------------- prior shift
def prior_adjust(prob: np.ndarray, train_prior: np.ndarray, n_iter: int = 200, tol: float = 1e-8):
    """Adjust class probabilities to the label mix of a new population (Saerens, Latinne & Decaestecker 2002).

    The classifier learned the label mix of the labelled sample, which was enriched with market keywords, so it
    expects more price/supply/demand texts than the full dataset contains. EM re-estimates the population label
    shares `pi` from the unlabelled texts and rescales each text's probabilities by pi / train_prior. Assumes that
    within each label the texts look alike in the sample and in the population (label shift).
    Returns (adjusted probabilities, estimated population shares)."""
    train_prior = np.asarray(train_prior, dtype=float)
    pi = train_prior.copy()
    for _ in range(n_iter):
        adj = prob * (pi / train_prior)
        adj /= adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    adj = prob * (pi / train_prior)
    adj /= adj.sum(1, keepdims=True)
    return adj, pi

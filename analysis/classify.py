"""Market-signal classifiers (price / supply / demand / other) and sentiment scoring.

Models compared (all evaluated with thread-grouped cross-validation on the manually labelled sample, so that a
post and the comments under it never sit on both sides of a train/test split):
  1. keyword   - the Task 1 keyword dictionary mapped to the four classes (no training)
  2. zeroshot  - facebook/bart-large-mnli zero-shot NLI classifier (no training)
  3. embed_lr  - frozen sentence embeddings (sentence-transformers/all-mpnet-base-v2) + logistic regression
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
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def embed(inputs: list[str], batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return model.encode(inputs, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


def embed_lr_fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", random_state=SEED).fit(X, y)


# ------------------------------------------------------------------------------------------- 4. fine-tune
FT_MODEL = "distilroberta-base"


def finetune_fit(train_inputs: list[str], train_labels: list[str], out_dir: str, epochs: int = 4):
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FT_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    enc = tok(train_inputs, truncation=True, max_length=192)
    y = [LABELS.index(l) for l in train_labels]
    counts = np.bincount(y, minlength=len(LABELS))
    weights = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)

    class DS(Dataset):
        def __len__(self):
            return len(y)

        def __getitem__(self, i):
            item = {k: enc[k][i] for k in enc}
            item["labels"] = y[i]
            return item

    class WeightedTrainer(Trainer):   # class-weighted loss for the imbalanced classes
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=16,
                             learning_rate=4e-5, weight_decay=0.01, warmup_ratio=0.1, logging_steps=20,
                             save_strategy="no", report_to=[], seed=SEED, use_cpu=True)
    from transformers import DataCollatorWithPadding
    trainer = WeightedTrainer(model=model, args=args, train_dataset=DS(),
                              data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    return tok, model


def finetune_predict(tok, model, inputs: list[str], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            enc = tok(inputs[i:i + batch_size], truncation=True, max_length=192, padding=True, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    p = np.vstack(probs)
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


def sentiment_scores(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Returns P(positive) - P(negative) in [-1, 1] for each text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).eval()
    id2 = {i: l.lower() for i, l in model.config.id2label.items()}
    pos, neg = [k for k, v in id2.items() if v.startswith("pos")][0], [k for k, v in id2.items() if v.startswith("neg")][0]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok([" ".join(str(t).split())[:600] for t in texts[i:i + batch_size]], truncation=True,
                      max_length=128, padding=True, return_tensors="pt")
            p = torch.softmax(model(**enc).logits, dim=-1).numpy()
            out.append(p[:, pos] - p[:, neg])
    return np.concatenate(out)


def set_threads(n: int | None = None) -> None:
    import torch
    torch.set_num_threads(n or os.cpu_count() or 1)

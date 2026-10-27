"""Write notebooks/task2_signals_and_prices.ipynb from the cells below (keeps the notebook diff-friendly).

    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/task2_signals_and_prices.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("md", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Task 2 — Classify market signals and compare them with USDA AMS prices

**Products:** avocados, lettuce (iceberg/romaine), onions, cucumbers · **Period:** 2022-01-01 to 2024-12-31
**Inputs:** `data/processed/event_dataset.csv` (Task 1), `labeling/labels.csv` (hand-reviewed sample),
`data/processed/ams_terminal_prices.csv.gz` (USDA AMS terminal markets).

Steps in this notebook
1. Labelled sample and agreement between the AI first pass and the human review
2. Classifier comparison (thread-grouped 5-fold cross-validation) and the selected model
3. Every observation classified as price / supply / demand / other, with a sentiment score
4. Weekly volume, sentiment and signal counts per product
5. Unusual weeks (trailing 8-week baseline, |z| ≥ 2) and the events behind them
6. USDA AMS weekly price indices
7. Timing: does the online signal lead, coincide with, or lag the price move?
8. Price impact relative to the pre-event baseline
9. Supply shock or demand shift?

The classifier and sentiment step is slow on a CPU (about 3–4 hours on 2 cores), so by default the notebook reads
the saved outputs of `python -m analysis.run_models all`. Set `RECOMPUTE_MODELS = True` to rerun it here.
""")

code(r"""
import os, sys, json, subprocess, warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "analysis").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis import prices as P, signals as S
from analysis.classify import LABELS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 110)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     # report figures are drawn at print size (6.5 in wide) with 8 pt text
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"]})
for _f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf"]:
    if Path(_f).exists():
        matplotlib.font_manager.fontManager.addfont(_f)

FIG, TAB = ROOT / "outputs/figures", ROOT / "outputs/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
START, END = "2022-01-01", "2025-01-01"
PRODUCTS = ["avocado", "lettuce", "onion", "cucumber"]
COLORS = {"price": "#b4690e", "supply": "#1f6fb2", "demand": "#9a3d8f", "other": "#9aa3a0"}
RECOMPUTE_MODELS = False

if RECOMPUTE_MODELS:
    subprocess.run([sys.executable, "-m", "analysis.run_models", "all"], check=True)

events_ref = pd.read_csv("config/events_reference.csv", parse_dates=["start_date", "end_date"])
events_ref[["event_id", "product", "event_name", "start_date", "end_date", "hypothesis"]]
""")

md(r"""
## 1. Labelled sample

The sample was drawn by `labeling/make_label_sample.py`, at most one text per Reddit thread: a stratified batch of
posts, keyword-enriched batches of posts and comments so that the rare price, supply and demand classes are
represented, and simple random batches that show how common each label really is. Labels come in two tiers:

* **gold**, used for training and for every accuracy figure, in two parts:
  * 1,118 texts from the first batches: two AI annotators (Claude Opus and Claude Sonnet, separate sessions)
    labelled every row with `labeling/codebook.md`; a team member decided every disagreement and a random sample of
    agreements, then took a second look at every reviewed row where demand was one of the choices. Two further
    annotators from another model family (OpenAI Codex and ChatGPT) labelled the same rows blind; where both
    disagreed with our label, a team member decided again on a blind A/B page.
  * 340 texts moved up from silver (`labeling/select_upgrade.py`): every silver text whose agreed label was price,
    supply or demand, 100 random agreed *other* texts, and the 40 silver texts already checked. A team member
    confirmed or corrected 202 of them; ChatGPT labelled all 340 blind (`labeling/crosscheck2/`). Where ChatGPT's
    label differed from ours (27 texts), a team member decided on the blind A/B page; the rest, which no person
    reviewed, have the label both Claude annotators and ChatGPT agree on.
* **silver**: the other texts of the silver batch where the two Claude annotators agreed (rows where they disagreed
  are dropped). After the move all remaining silver labels are *other*. Silver labels are only added to the
  training folds.

`label` is the human decision where a person decided the row and the agreed AI label elsewhere; the full procedure is
at the end of the codebook.
""")

code(r"""
from sklearn.metrics import cohen_kappa_score
lab = pd.read_csv("labeling/labels.csv")
lab = lab[lab["label"].isin(LABELS)]
gold, silver = lab[lab["tier"] == "gold"], lab[lab["tier"] == "silver"]
moved = gold["upgrade"].fillna(False).astype(bool)
first, gold_up = gold[~moved], gold[moved]
print(f"{len(gold)} gold ({len(first)} from the first batches, {len(gold_up)} moved up from silver) and "
      f"{len(silver)} silver labels "
      f"({(lab.record_type == 'post').sum()} posts, {(lab.record_type == 'comment').sum()} comments)")
display(pd.crosstab([lab["tier"], lab["product"]], lab["label"], margins=True).reindex(columns=LABELS + ["All"]))
display(pd.crosstab(lab["batch"], lab["tier"]))
print(f"First-batch gold, Claude annotators 1 and 2: agree on {first['ai_agree'].mean():.1%} of rows, Cohen's kappa "
      f"{cohen_kappa_score(first['label_ai1'], first['label_ai2']):.2f}")
display(pd.crosstab(first["label_ai1"], first["label_ai2"], rownames=["annotator 1"], colnames=["annotator 2"]))
print(gold["label_source"].value_counts().to_string())
check = first[first["reviewed"] & first["ai_agree"]]
print(f"Random check of {len(check)} gold rows where the Claude annotators agreed: the reviewer kept the AI label "
      f"for {(check['human_label'] == check['label_ai1']).mean():.1%}.")
dis = first[first["reviewed"] & ~first["ai_agree"]]
print(f"Disagreements decided by the reviewer: {len(dis)}; reviewer sided with annotator 1 in "
      f"{(dis['human_label'] == dis['label_ai1']).sum()}, annotator 2 in "
      f"{(dis['human_label'] == dis['label_ai2']).sum()}, neither in "
      f"{((dis['human_label'] != dis['label_ai1']) & (dis['human_label'] != dis['label_ai2'])).sum()}.")
re2 = gold[gold["rechecked"]]
print(f"Second look at demand vs. other: {len(re2)} rows, {(re2['first_pass_label'] != re2['human_label']).sum()} changed:")
display(pd.crosstab(re2["first_pass_label"], re2["human_label"], rownames=["first look"], colnames=["second look"]))
""")

code(r"""
# Four annotators from two model families on the gold rows, and the blind A/B adjudication
display(pd.read_csv(TAB / "annotator_agreement.csv"))
adj = first[first["adjudicated"]]
print(f"Adjudicated (both OpenAI annotators agree with each other, not with our label): {len(adj)} rows; "
      f"our label kept in {(adj['label'] == adj['label_before_adjudication']).sum()}, OpenAI label taken in "
      f"{(adj['label'] != adj['label_before_adjudication']).sum()}.")
display(pd.crosstab(adj["label_before_adjudication"], adj["label"], rownames=["before"], colnames=["after"]))
# texts moved up from silver: the person's review, ChatGPT's blind label and the blind decisions
rev_up = gold_up[gold_up["reviewed"]]
print(f"Moved up from silver: {len(gold_up)} texts; the person reviewed {len(rev_up)} and changed the agreed AI label "
      f"in {(rev_up['human_label'] != rev_up['label_ai1']).sum()}:")
display(pd.crosstab(rev_up["label_ai1"], rev_up["human_label"], rownames=["agreed AI label"], colnames=["person"]))
adj2 = gold_up[gold_up["adjudicated"]]
print(f"ChatGPT differed from our label on {len(adj2)} of them; blind decision kept ours in "
      f"{(adj2['label'] == adj2['label_before_adjudication']).sum()}, took ChatGPT's in "
      f"{(adj2['label'] == adj2['label_chatgpt_round2']).sum()}, chose a third label in "
      f"{((adj2['label'] != adj2['label_before_adjudication']) & (adj2['label'] != adj2['label_chatgpt_round2'])).sum()}.")
if (TAB / "human_agreement.csv").exists():
    print("Second human annotator, blind, on a sample of rows:")
    display(pd.read_csv(TAB / "human_agreement.csv"))
""")

md(r"""
## 2. Which classifier?

All models were evaluated with **thread-grouped stratified 5-fold cross-validation**: a post and the comments under
it never fall on both sides of a train/test split, so near-duplicate context cannot leak into the test fold.
The 95% interval for macro-F1 comes from a cluster bootstrap over threads. For the embedding models the
regularisation strength of the logistic regression is chosen inside each training fold (inner 3-fold CV), so the
reported scores are not tuned on the test folds. Every score is computed on the **gold** labels only; the `+ silver`
variants add the silver labels to each training fold (never to a test fold). The model with the highest macro-F1 is
selected; macro-F1 weights the four labels equally, which matters because most gold texts are *other*.
""")

code(r"""
cv = pd.read_csv(TAB / "model_cv_metrics.csv", index_col="model")
names = {"keyword": "Keyword dictionary (Task 1 rules)", "zeroshot": "Zero-shot BART-large-MNLI",
         "embed_lr": "Embeddings (all-mpnet-base-v2) + logistic regression",
         "embed_lr_bge": "Embeddings (bge-base-en-v1.5) + logistic regression",
         "embed_lr_gte": "Embeddings (gte-modernbert-base) + logistic regression",
         "finetune": "Fine-tuned DistilRoBERTa"}
names |= {f"{k}_silver": f"{v} + silver" for k, v in list(names.items()) if k.startswith("embed")}
show = cv.rename(index=names)[["accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "cohen_kappa",
                              "f1_price", "f1_supply", "f1_demand", "f1_other"]].round(3)
display(show)
calib = [c for c in cv.columns if c.startswith("expected_")]
print("Calibration: summed out-of-fold probabilities vs. true counts (embedding models)")
display(cv.loc[cv[calib].notna().all(axis=1), calib].rename(index=names))
selected = json.load(open(TAB / "selected_model.json"))["selected_model"]
print("Selected model:", names[selected])

oof = pd.read_csv(TAB / "model_cv_predictions.csv")
cm = pd.crosstab(oof["label"], oof[selected], rownames=["human label"], colnames=["predicted"]).reindex(
    index=LABELS, columns=LABELS, fill_value=0)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm.values / cm.values.sum(1, keepdims=True), cmap="Greens", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=9)
ax.set_xticks(range(4), LABELS); ax.set_yticks(range(4), LABELS)
ax.set_xlabel("predicted"); ax.set_ylabel("human label")
ax.set_title(f"{names[selected]}: out-of-fold confusion matrix", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=200); plt.show()
rnd = oof.merge(lab[["obs_id", "batch"]], on="obs_id")
rnd = rnd[rnd["batch"] == "random"]
print(f"On the {len(rnd)} texts drawn by simple random sampling (closest to the real mix of texts), the selected "
      f"model is right for {(rnd['label'] == rnd[selected]).mean():.1%} (out of fold).")
print("Accuracy by product (selected model, out of fold):")
display(oof.merge(lab[["obs_id", "product"]], on="obs_id").assign(ok=lambda d: d["label"] == d[selected])
        .groupby("product")["ok"].mean().round(3))
""")

md(r"""
## 3. Classify every observation

The selected model, retrained on the whole labelled sample, gives every observation of the event dataset a
probability for each label; `pred_label` is the most likely one. Rows of the labelled sample keep their human label.

**Counting signals.** A weekly count of hard labels would be biased: a classifier that misses 40% of demand texts
undercounts demand every week, and one tuned for recall overcounts it. Our logistic regression is fitted without
class weights, so its probabilities are calibrated: in cross-validation the summed probabilities match the true
class counts (section 2, `expected_*_vs_true`). Weekly counts are therefore **expected counts**, the sum of the
class probabilities over that week's texts (probabilistic classify-and-count).

**Checking the label shares against random samples.** Calibration in cross-validation holds for texts like the
labelled ones, but most labelled texts were enriched with market keywords. We therefore compare, per product, the
mean predicted probability over the whole dataset with the share in the simple random samples (batches `random` and
`random6`, about 190 texts per product, labels from the AI annotators and the reviewer). We also tried an EM
re-estimation of the label shares (Saerens et al., 2002; `analysis.classify.prior_adjust`); it moved the estimates
slightly further from the random samples (mean absolute difference 1.9 vs 1.7 percentage points), because the
training labels already contain about 770 randomly drawn texts. The unadjusted probabilities are used.

Sentiment is `P(positive) − P(negative)` from `cardiffnlp/twitter-roberta-base-sentiment-latest`.
""")

code(r"""
ev = pd.read_csv("data/processed/event_dataset.csv", low_memory=False, parse_dates=["week_start", "date"])
cls = pd.read_csv("data/processed/classified_observations.csv.gz")
obs = ev.merge(cls[["obs_id", "pred_label", "human_label", "sentiment"] + [f"p_{l}" for l in LABELS]],
               on="obs_id", how="left")
has_human = obs["human_label"].isin(LABELS)          # labelled sample: the human label replaces the prediction
obs.loc[has_human, "pred_label"] = obs.loc[has_human, "human_label"]
for l in LABELS:
    obs.loc[has_human, f"p_{l}"] = (obs.loc[has_human, "human_label"] == l).astype(float)
print(f"{has_human.sum()} observations carry their human label")
print(f"{len(obs):,} observations, {obs['record_id'].nunique():,} Reddit records, {obs['thread_id'].nunique():,} threads")
prev = pd.read_csv(TAB / "label_prevalence.csv")
display(prev.pivot(index="product", columns="label", values=["mean_probability_before_adjustment",
                   "estimated_share_after_adjustment", "random_sample_share"]).round(3))
dist = pd.crosstab(obs["product"], obs["pred_label"]).reindex(columns=LABELS)
expected = obs.groupby("product")[[f"p_{l}" for l in LABELS]].sum().round(0).astype(int)
expected.columns = LABELS
dist_share = expected.div(expected.sum(1), axis=0).round(3)
display(pd.concat({"most likely label": dist, "expected count": expected, "expected share": dist_share}, axis=1))
dist.to_csv(TAB / "predicted_label_counts.csv")
""")

md(r"""
## 4. Weekly market signals

Per product and ISO week: number of observations (`volume`), product posts per 10,000 posts in the listed
subreddits (`posts_per_10k`, removes growth of the subreddits themselves), mean sentiment, and the number and share of
price / supply / demand observations. Grey bands are the documented events in `config/events_reference.csv`.
""")

code(r"""
post_counts = pd.read_csv("data/processed/weekly_post_counts.csv", parse_dates=["week_start"])
sig = S.weekly_signals(obs, post_counts, START, END)
sig = S.add_anomalies(sig)
sig.to_csv(TAB / "weekly_signals.csv", index=False)

def shade(ax, prod):
    prev, level = None, 0
    for e in events_ref[events_ref["product"] == prod].sort_values("start_date").itertuples():
        ax.axvspan(e.start_date, e.end_date, color="0.9", zorder=0, lw=0)
        level = level + 1 if prev is not None and (e.start_date - prev).days < 45 else 0   # stagger close labels
        ax.text(e.start_date, 1 - 0.13 * level, f" {e.event_id}", transform=ax.get_xaxis_transform(), va="top",
                fontsize=6.5, color="0.35")
        prev = e.start_date

fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.9), sharex=True)
for ax, prod in zip(axes.flat, PRODUCTS):
    g = sig[sig["product"] == prod].set_index("week_start")
    for lab_ in ["price", "supply", "demand"]:
        ax.plot(g.index, g[f"n_{lab_}"], color=COLORS[lab_], lw=0.9, label=lab_)
    ax.set_title(prod, loc="left", fontweight="bold")
    shade(ax, prod)
for ax in axes[:, 0]:
    ax.set_ylabel("texts per week")
for ax in axes[-1]:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "weekly_signals.png", dpi=300); plt.show()

fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
for ax, prod in zip(axes, PRODUCTS):
    g = sig[sig["product"] == prod].set_index("week_start")
    ax.bar(g.index, g["posts_per_10k"], width=6, color="#2f7d4f", alpha=0.8)
    ax2 = ax.twinx(); ax2.plot(g.index, g["sentiment"].rolling(4, min_periods=2).mean(), color="#333", lw=1)
    ax2.axhline(0, color="0.6", lw=0.6); ax2.set_ylim(-0.6, 0.6)
    ax.set_ylabel(f"{prod}\nposts per 10k"); ax2.set_ylabel("sentiment\n(4-wk mean)")
    shade(ax, prod)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Normalised discussion volume (bars) and sentiment (line)", y=0.995)
fig.tight_layout(); fig.savefig(FIG / "volume_sentiment.png", dpi=200); plt.show()
""")

md(r"""
## 5. Unusual weeks and the events behind them

A week is unusual for a series when it lies at least 2 standard deviations above (*increase*) or below
(*decrease*) the mean of the previous 8 weeks; otherwise *no change*. Series: log volume, mean sentiment and the
log counts of price, supply and demand observations. For each unusual week we list the words most over-represented
in that week relative to the product's whole corpus (log-odds with an informative Dirichlet prior) and the most
up-voted texts, and we match the week to a documented event within ±2 weeks.
""")

code(r"""
import re
from collections import Counter

STOP = set(("the a an and or but if of to in on for with at by from is are was were be been it its this that these "
            "those i you he she we they my your our their me him her them as so not no do does did have has had just "
            "like get got can will would could should about what when where who how all any more most some than then "
            "there here out up down over very too also really one two im dont thats ive youre theyre its it's i'm "
            "don't i've you're they're amp https http com www reddit").split())

def tokens(text):
    return [w for w in re.findall(r"[a-z][a-z']+", str(text).lower()) if w not in STOP and len(w) > 2]

def distinctive_words(week_texts, all_counts, k=8):
    wc = Counter(w for t in week_texts for w in tokens(t))
    n_w, n_a = sum(wc.values()), sum(all_counts.values())
    if n_w == 0:
        return ""
    a0, scores = 0.01 * n_a, {}
    for w, c in wc.items():
        if c < 3:
            continue
        a = all_counts.get(w, 0) * 0.01 + 0.01
        lw = np.log((c + a) / (n_w + a0 - c - a))
        la = np.log((all_counts.get(w, 0) + a) / (n_a + a0 - all_counts.get(w, 0) - a))
        scores[w] = (lw - la) / np.sqrt(1 / (c + a) + 1 / (all_counts.get(w, 0) + a))
    return ", ".join(sorted(scores, key=scores.get, reverse=True)[:k])

zcols = {"volume": "z_volume", "price": "z_n_price", "supply": "z_n_supply", "demand": "z_n_demand",
         "sentiment": "z_sentiment"}
rows = []
for prod in PRODUCTS:
    g = sig[sig["product"] == prod]
    corpus = Counter(w for t in obs.loc[obs["product"] == prod, "text"] for w in tokens(t))
    for r in g.itertuples():
        hits = {k: getattr(r, c) for k, c in zcols.items() if pd.notna(getattr(r, c)) and abs(getattr(r, c)) >= 2}
        if not hits or r.volume < 5:
            continue
        wk = obs[(obs["product"] == prod) & (obs["week_start"] == r.week_start)]
        top = wk.sort_values("score", ascending=False).head(2)
        ev_match = events_ref[(events_ref["product"] == prod)
                              & (events_ref["start_date"] - pd.Timedelta(weeks=2) <= r.week_start)
                              & (events_ref["end_date"] + pd.Timedelta(weeks=2) >= r.week_start)]
        rows.append({"product": prod, "week_start": r.week_start.date(), "volume": r.volume,
                     "unusual_series": ", ".join(f"{k} {'increase' if v > 0 else 'decrease'} (z={v:.1f})" for k, v in hits.items()),
                     "max_abs_z": max(abs(v) for v in hits.values()),
                     "distinctive_words": distinctive_words(wk["text"], corpus),
                     "documented_event": "; ".join(ev_match["event_id"] + " " + ev_match["event_name"]),
                     "top_text": " | ".join(top["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 140)),
                     "top_url": " ".join(top["url"])})
unusual = pd.DataFrame(rows).sort_values(["product", "week_start"])
unusual.to_csv(TAB / "unusual_weeks.csv", index=False)
print(f"{len(unusual)} unusual product-weeks; {(unusual['documented_event'] != '').sum()} fall within 2 weeks of a documented event")
display(unusual.sort_values("max_abs_z", ascending=False).head(25)[
    ["product", "week_start", "volume", "unusual_series", "distinctive_words", "documented_event"]])
""")

md(r"""
**Which spikes point to real events?** Many weeks are unusual in some series (five series, two directions), so a
single unusual week means little. About 12% of product-weeks lie within two weeks of a documented event; if a kind
of spike carried no information, about 12% of its weeks would too. For each series and threshold we count the
weeks at or above it (|z| for sentiment), how many lie near a documented event, and test that share against the
base rate (one-sided binomial test).
""")

code(r"""
from scipy.stats import binomtest
sw = sig[(sig["week_start"] >= "2022-01-03") & (sig["week_start"] <= "2024-12-23")].copy()   # whole weeks only
def near_event(prod, w):
    m = events_ref[(events_ref["product"] == prod) & (events_ref["start_date"] - pd.Timedelta(weeks=2) <= w)
                   & (events_ref["end_date"] + pd.Timedelta(weeks=2) >= w)]
    return ", ".join(m["event_id"])
sw["near_event"] = [near_event(p, w) for p, w in zip(sw["product"], sw["week_start"])]
base = (sw["near_event"] != "").mean()
rows = []
for name, col in zcols.items():
    for thr in (2, 3, 4):
        x = sw[col].abs() if name == "sentiment" else sw[col]
        hit = sw[x >= thr]
        k = int((hit["near_event"] != "").sum())
        found = sorted({e for v in hit["near_event"] if v for e in v.split(", ")})
        rows.append({"series": name, "z_at_least": thr, "weeks": len(hit), "near_documented_event": k,
                     "share": round(k / len(hit), 2) if len(hit) else np.nan,
                     "p_vs_base_rate": round(binomtest(k, len(hit), base, alternative="greater").pvalue, 4)
                     if len(hit) else np.nan,
                     "events_found": ", ".join(found)})
spikes = pd.DataFrame(rows)
spikes.to_csv(TAB / "spike_event_enrichment.csv", index=False)
print(f"{len(sw)} product-weeks; {base:.1%} lie within 2 weeks of a documented event (the base rate)")
display(spikes)
""")

md(r"""
## 6. USDA AMS terminal market prices

`analysis/prices.py` turns ~457,000 reported price lines into a fixed-composition weekly price index per product
(two-way fixed effects: item series × week; an item series is one market × commodity × variety × package × size ×
origin × organic combination). Index mean over 2022–2024 ≈ 100.
""")

code(r"""
prices = P.load_prices("data/processed/ams_terminal_prices.csv.gz")
widx = S.price_anomalies(P.weekly_indices(prices))
widx.to_csv(TAB / "price_index_weekly.csv", index=False)
nat = widx[widx["market"] == "ALL"]
print(prices.groupby("product").agg(price_lines=("price", "size"), item_series=("series", "nunique"),
                                    markets=("market", "nunique")))

fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
for ax, prod in zip(axes, PRODUCTS):
    g = nat[nat["product"] == prod]
    ax.plot(g["week_start"], g["index"], color="#18211d", lw=1.3, zorder=3)
    for mkt, gm in widx[(widx["product"] == prod) & (widx["market"] != "ALL")].groupby("market"):
        ax.plot(gm["week_start"], gm["index"], color="0.68", lw=0.5, zorder=1)
    ax.set_ylabel(f"{prod}\nprice index")
    shade(ax, prod)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("USDA AMS terminal market price index (black: all markets pooled; grey: single markets)", y=0.995)
fig.tight_layout(); fig.savefig(FIG / "price_indices.png", dpi=200); plt.show()
""")

md(r"""
**Shipments.** To tell supply from demand we also need the *quantity* reaching the market. AMS movement reports
give the pounds of each commodity shipped from US production districts and imported through ports of entry
(`data_collection/collect_ams_movement.py`). We sum them per product and week.
""")

code(r"""
ship = pd.read_csv("data/processed/ams_shipments_weekly.csv", parse_dates=["week_start"])
ship = ship[(ship["week_start"] >= "2022-01-03") & (ship["week_start"] <= "2024-12-23")]   # whole weeks only
ship["log_lb"] = np.log(ship["total_lb"].where(ship["total_lb"] > 0))
ship_log = {p: g.set_index("week_start")["log_lb"] for p, g in ship.groupby("product")}
print(ship.groupby("product")[["domestic_lb", "import_lb"]].sum().div(1e9).round(2).rename(
    columns=lambda c: c.replace("_lb", " (billion lb, 2022-2024)")))

fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
for ax, prod in zip(axes, PRODUCTS):
    g = ship[ship["product"] == prod]
    ax.stackplot(g["week_start"], g["domestic_lb"] / 1e6, g["import_lb"] / 1e6, colors=["#7fa98f", "#d2b07a"],
                 labels=["US shipments", "imports"], lw=0)
    ax.set_ylabel(f"{prod}\nmillion lb / week")
    shade(ax, prod)
fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.965))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Weekly shipments reaching US markets (USDA AMS movement reports)", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "shipments.png", dpi=200); plt.show()

# Report figure: price index (left) and shipments (right) side by side, one row per product
fig, axes = plt.subplots(4, 2, figsize=(6.5, 5.0), sharex=True)
for (axp, axs), prod in zip(axes, PRODUCTS):
    g = nat[nat["product"] == prod]
    for mkt, gm in widx[(widx["product"] == prod) & (widx["market"] != "ALL")].groupby("market"):
        axp.plot(gm["week_start"], gm["index"], color="0.75", lw=0.35, zorder=1)
    axp.plot(g["week_start"], g["index"], color="#18211d", lw=0.9, zorder=3)
    gs = ship[ship["product"] == prod]
    axs.stackplot(gs["week_start"], gs["domestic_lb"] / 1e6, gs["import_lb"] / 1e6, colors=["#7fa98f", "#d2b07a"],
                  labels=["US shipments", "imports"], lw=0)
    axp.set_ylabel(prod, fontweight="bold")
    shade(axp, prod); shade(axs, prod)
axes[0, 0].set_title("(a) Wholesale price index", loc="left", fontweight="bold")
axes[0, 1].set_title("(b) Shipments, million lb per week", loc="left", fontweight="bold")
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower right", ncol=2, frameon=False)
for ax in axes[-1]:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(h_pad=0.6, rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "prices_and_shipments.png", dpi=300); plt.show()
""")

md(r"""
## 7. Timing: does the online signal lead or lag the price?

**Series-level.** For each product we correlate a weekly signal (its deviation from the previous 8 weeks, z-score)
with the price index deviation (also an 8-week trailing z-score) shifted by *k* weeks:
corr(signal<sub>t</sub>, price<sub>t+k</sub>). *k > 0* means the online signal **leads** the price, *k < 0* that
it **lags**. The grey band in the figure is ±1.96/√n, the 5% band for one pre-chosen lag between independent series.
Our series are autocorrelated and we pick the best of 17 lags, so that band is too generous. The test we use is a
rotation test (`analysis.signals.best_lag_pvalue`): the signal is shifted in time against the price by every offset of
at least 13 weeks, which keeps each series' autocorrelation and seasonality but breaks any real timing link, and the
p-value is the share of shifts whose best |r| over all lags is at least as large as the observed one. It is
approximate: in simulations with unrelated series of the same autocorrelation as ours (lag-1 about 0.2 for the
signals, 0.8 for the price), it gives p < 0.05 for about 8% of pairs instead of 5%; the simple band flags about 58%.

**Event-level.** For each documented event we find, within a window from 4 weeks before its start to 8 weeks
after its end, the first week the signal crosses z ≥ 2 and the first week the price index crosses |z| ≥ 2, and
report the difference in days (positive = signal first).
""")

code(r"""
signal_cols = {"volume": "z_volume", "price posts": "z_n_price", "supply posts": "z_n_supply",
               "demand posts": "z_n_demand", "sentiment": "z_sentiment"}
xc_rows, xc_curves = [], {}
for prod in PRODUCTS:
    g = sig[sig["product"] == prod].set_index("week_start")
    pz = nat[nat["product"] == prod].set_index("week_start")["z_price"].reindex(g.index)
    for name, col in signal_cols.items():
        xc = S.cross_correlation(g[col], pz, max_lag=8)
        xc_curves[(prod, name)] = xc
        if xc["r"].isna().all():
            continue
        best = xc.loc[xc["r"].abs().idxmax()]
        r0 = xc.loc[xc["lag_weeks"] == 0, "r"].item()
        pval = S.best_lag_pvalue(g[col], pz, max_lag=8)
        xc_rows.append({"product": prod, "signal": name, "r_same_week": round(r0, 3),
                        "best_lag_weeks": int(best["lag_weeks"]), "r_at_best_lag": round(best["r"], 3),
                        "outside_simple_band": abs(best["r"]) > best["ci95"], "p_rotation": round(pval, 3),
                        "reading": "signal leads price" if best["lag_weeks"] > 0 else
                                   "signal lags price" if best["lag_weeks"] < 0 else "same week"})
xcorr = pd.DataFrame(xc_rows)
xcorr.to_csv(TAB / "lead_lag_crosscorrelation.csv", index=False)
display(xcorr)

fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.2), sharey=True)
for ax, prod in zip(axes, PRODUCTS):
    for name, colr in [("volume", "#2f7d4f"), ("supply posts", COLORS["supply"]), ("price posts", COLORS["price"]),
                       ("demand posts", COLORS["demand"])]:
        xc = xc_curves[(prod, name)]
        ax.plot(xc["lag_weeks"], xc["r"], color=colr, lw=0.9, label=name)
    ax.fill_between(xc["lag_weeks"], -xc["ci95"], xc["ci95"], color="0.9", zorder=0)
    ax.axvline(0, color="0.5", lw=0.5); ax.set_title(prod, fontweight="bold"); ax.set_xlabel("lag (weeks)")
axes[0].set_ylabel("corr(signal$_t$, price$_{t+k}$)")
fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(FIG / "lead_lag.png", dpi=300); plt.show()
""")

code(r"""
timing = []
for e in events_ref.itertuples():
    g = sig[sig["product"] == e.product].set_index("week_start")
    pz = nat[nat["product"] == e.product].set_index("week_start")["z_price"]
    w0, w1 = e.start_date - pd.Timedelta(weeks=4), e.end_date + pd.Timedelta(weeks=8)
    p_first = S.first_crossing(pz, w0, w1, 2.0, "abs")
    row = {"event_id": e.event_id, "product": e.product, "event": e.event_name, "event_start": e.start_date.date(),
           "first_price_move_week": p_first.date() if pd.notna(p_first) else None,
           "price_direction": ("up" if pz.get(p_first, 0) > 0 else "down") if pd.notna(p_first) else None}
    for name, col in [("volume", "z_volume"), ("supply", "z_n_supply"), ("price", "z_n_price"), ("demand", "z_n_demand")]:
        s_first = S.first_crossing(g[col], w0, w1, 2.0, "up")
        row[f"first_{name}_signal_week"] = s_first.date() if pd.notna(s_first) else None
        row[f"{name}_lead_days"] = (p_first - s_first).days if pd.notna(p_first) and pd.notna(s_first) else None
    timing.append(row)
timing = pd.DataFrame(timing)
timing.to_csv(TAB / "event_timing.csv", index=False)
display(timing)
""")

md(r"""
## 8. Price impact relative to the pre-event baseline

**Response window.** Produce takes days to weeks to move from field or border to a terminal market, and stock in
transit is sold first, so prices and recorded shipments respond with a lag. For every event we therefore measure
over a *response window*: the documented event dates plus `RESPONSE_WEEKS` = 2 weeks after the end. The same rule
is used for all events, for the placebo windows and for shipments.

Change of the pooled price index during the response window relative to its mean over the 8 weeks before the event
(mean and peak change), and the same calendar-window change in the other study years as a seasonality control
(`seasonally_adjusted_pct` = event-year change minus the average other-year change, in log points). Other-year
windows that overlap another documented event of the product are not used as comparisons.
Per-market ranges show how uniform the move was across the 10 terminal markets.

`overlaps_with` lists other documented events of the same product whose dates fall inside the 8-week baseline or the
response window. Their effects cannot be separated: in 2022 the Wendy's romaine outbreak (E8) runs into the start of
the INSV lettuce shortage (E2), and in 2024 the cucumber Salmonella outbreak (E3) runs into the viral salad trend (E4).
""")

code(r"""
RESPONSE_WEEKS = 2
events_ref["response_end"] = events_ref["end_date"] + pd.Timedelta(weeks=RESPONSE_WEEKS)
def event_windows(prod):   # documented response windows of a product (kept out of seasonal and placebo baselines)
    return [(r.start_date, r.response_end) for r in events_ref.itertuples() if r.product == prod]

def overlapping(e, pre_weeks=8):   # other events of the product dated inside this event's baseline or response window
    lo, hi = e.start_date - pd.Timedelta(weeks=pre_weeks), e.response_end
    return ", ".join(r.event_id for r in events_ref.itertuples() if r.product == e.product
                     and r.event_id != e.event_id and r.start_date <= hi and r.end_date >= lo)

impact = []
for e in events_ref.itertuples():
    li = nat[nat["product"] == e.product].set_index("week_start")["log_index"]
    r = S.price_impact(li, e.start_date, e.response_end, exclude=event_windows(e.product))
    mk = []
    for mkt, gm in widx[(widx["product"] == e.product) & (widx["market"] != "ALL")].groupby("market"):
        mk.append(S.price_impact(gm.set_index("week_start")["log_index"], e.start_date, e.response_end,
                                 exclude=event_windows(e.product))["mean_change_pct"])
    mk = [m for m in mk if pd.notna(m)]
    impact.append({"event_id": e.event_id, "product": e.product, "event": e.event_name,
                   **{k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()},
                   "market_range_pct": f"{min(mk):.0f} to {max(mk):.0f}" if mk else "",
                   "overlaps_with": overlapping(e)})
impact = pd.DataFrame(impact)
impact.to_csv(TAB / "event_price_impact.csv", index=False)
display(impact)
""")

md(r"""
**In dollars.** The index is relative. For a concrete reading, the most often reported conventional item of each
product (for example a carton of 24 film-wrapped iceberg heads) in the 10 markets: median price per package in the
8 weeks before the event, during the response window, and the highest weekly median. These figures are not
seasonally adjusted.
""")

code(r"""
conv = prices[prices["organic"].astype(str).str.upper() != "Y"].copy()
conv["report_date"] = pd.to_datetime(conv["report_date"])
item_cols = ["commodity", "variety", "package", "item_size"]
dollars = []
for e in events_ref.itertuples():
    g = conv[conv["product"] == e.product]
    top = g.groupby(item_cols, dropna=False).size().idxmax()
    g = g[(g[item_cols].fillna("") == pd.Series(top, index=item_cols).fillna("")).all(axis=1)]
    pre = g[(g["report_date"] >= e.start_date - pd.Timedelta(weeks=8)) & (g["report_date"] < e.start_date)]
    dur = g[(g["report_date"] >= e.start_date) & (g["report_date"] <= e.response_end)]
    item = " ".join(str(v) for v in top if pd.notna(v) and str(v) not in ("", "NA"))
    dollars.append({"event_id": e.event_id, "item": item, "median_before_usd": pre["price"].median(),
                    "median_during_usd": dur["price"].median(),
                    "highest_weekly_median_usd": dur.groupby("week_start")["price"].median().max(),
                    "price_lines_before_during": f"{len(pre)} / {len(dur)}"})
dollars = pd.DataFrame(dollars).round(2)
dollars.to_csv(TAB / "event_price_examples_usd.csv", index=False)
display(dollars)
""")

md(r"""
## 9. Supply shock or demand shift?

Economics tells supply shocks from demand shifts by how price and quantity move together: a **supply shock** raises
the price while less product reaches the market; a **demand shift** moves price and quantity in the same direction.
We first read each event from the market data alone, then ask whether the online talk tells the same story.

**Market reading.** For the price index and for weekly national shipments we take the seasonally adjusted change
over the response window (section 8: mean during the window vs. the 8 weeks before, minus the same calendar change
in the other years). To judge whether a change is large we compare it with the same statistic for every other window
of the same length in 2022–2024 that does not overlap a documented event of the product (a placebo distribution),
and report its percentile.

* Price: percentile ≥ 90 = unusual rise, ≤ 10 = unusual fall.
* Shipments: percentile ≤ 25 = fell, ≥ 75 = rose, otherwise flat.

| Price | Shipments | Market reading |
|---|---|---|
| unusual rise | fell | **supply shock** |
| unusual rise | flat or rose | **demand increase** |
| unusual fall | fell or flat | **demand decrease** |
| unusual fall | rose | **supply increase** |
| no unusual move | any | no clear price effect |

**Online reading.** Expected counts of *supply* and *demand* texts (section 3) during the event dates
(online talk reacts at once, so no response lag) vs. the 8 weeks before, as a rate ratio with a 95% Poisson interval. A kind of talk "rose" when its interval is above 1 and it
rose more than background (*other*) talk about the product. Supply talk rising (at least as much as demand talk) →
*supply*; demand talk rising more → *demand*. Note that by our codebook, reports of recalls and products pulled from
shelves count as supply talk; people avoiding a product count as demand talk.
""")

code(r"""
def counts(prod, s0, s1):
    g = obs[(obs["product"] == prod) & (obs["week_start"] >= s0) & (obs["week_start"] <= s1)]
    return {l: float(g[f"p_{l}"].sum()) for l in LABELS} | {"n": len(g), "sentiment": g["sentiment"].mean()}

def wk(d):   # Monday of the week containing d
    d = pd.Timestamp(d); return d - pd.Timedelta(days=d.weekday())

fmt = lambda t: f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f})"
verdict = []
for e in events_ref.itertuples():
    s0, s1 = wk(e.start_date), wk(e.end_date)
    w_dur, w_pre = (s1 - s0).days // 7 + 1, 8
    others = event_windows(e.product)
    # market reading (response window)
    li = nat[nat["product"] == e.product].set_index("week_start")["log_index"]
    sa = "seasonally_adjusted_pct"
    p_pct = S.placebo_percentile(li, e.start_date, e.response_end, exclude=others, stat=sa)["percentile"]
    q = S.price_impact(ship_log[e.product], e.start_date, e.response_end, exclude=others)
    q_pct = S.placebo_percentile(ship_log[e.product], e.start_date, e.response_end, exclude=others, stat=sa)["percentile"]
    p_up, p_down = p_pct >= 90, p_pct <= 10
    q_dir = "fell" if q_pct <= 25 else "rose" if q_pct >= 75 else "flat"
    market = ("supply shock" if p_up and q_dir == "fell" else "demand increase" if p_up else
              "supply increase" if p_down and q_dir == "rose" else "demand decrease" if p_down else
              "no clear price effect")
    # online reading
    pre = counts(e.product, s0 - pd.Timedelta(weeks=w_pre), s0 - pd.Timedelta(days=1))
    dur = counts(e.product, s0, s1)
    rr = {l: S.rate_ratio(dur[l], w_dur, pre[l], w_pre) for l in LABELS}
    rose = {l: rr[l][1] > 1 and rr[l][0] > rr["other"][0] for l in ["supply", "demand"]}
    online = ("supply" if rose["supply"] and (not rose["demand"] or rr["supply"][0] >= rr["demand"][0]) else
              "demand" if rose["demand"] else "no rise in supply or demand talk")
    agree = ("yes" if (market.startswith("supply") and online == "supply")
                      or (market.startswith("demand") and online == "demand") else
             "no" if market != "no clear price effect" and online in ("supply", "demand") else "-")
    imp = impact.set_index("event_id").loc[e.event_id]
    verdict.append({"event_id": e.event_id, "product": e.product, "event": e.event_name,
                    "price_change_pct": imp["mean_change_pct"], "price_seasonally_adjusted_pct": imp[sa],
                    "price_percentile": round(p_pct), "shipments_change_pct": round(q["mean_change_pct"], 1),
                    "shipments_seasonally_adjusted_pct": round(q[sa], 1), "shipments_percentile": round(q_pct),
                    "shipments": q_dir,
                    "market_reading": market,
                    "supply_talk_rate_ratio": fmt(rr["supply"]), "demand_talk_rate_ratio": fmt(rr["demand"]),
                    "price_talk_rate_ratio": fmt(rr["price"]), "other_talk_rate_ratio": fmt(rr["other"]),
                    "supply_talk_before_during": f"{pre['supply']:.0f} / {dur['supply']:.0f}",
                    "demand_talk_before_during": f"{pre['demand']:.0f} / {dur['demand']:.0f}",
                    "weeks_before_during": f"{w_pre} / {w_dur}",
                    "online_reading": online, "readings_agree": agree, "expected": e.hypothesis,
                    "overlaps_with": overlapping(e)})
verdict = pd.DataFrame(verdict)
verdict.to_csv(TAB / "event_supply_vs_demand.csv", index=False)
display(verdict)
""")

md(r"""
## 10. Unusual weeks without a documented event

Large online anomalies that are not near a documented event, with whether the price index also moved
(|z| ≥ 2) within ±4 weeks. These are candidate market events found by the data alone.
""")

code(r"""
cand = unusual[(unusual["documented_event"] == "") & (unusual["volume"] >= 10)].copy()
def price_move_near(prod, wk):
    pz = nat[nat["product"] == prod].set_index("week_start")["z_price"]
    w = pd.Timestamp(wk)
    near = pz[(pz.index >= w - pd.Timedelta(weeks=4)) & (pz.index <= w + pd.Timedelta(weeks=4))]
    big = near[near.abs() >= 2]
    return f"{big.index[0].date()} ({'up' if big.iloc[0] > 0 else 'down'})" if len(big) else ""
cand["price_move_within_4wk"] = [price_move_near(p, w) for p, w in zip(cand["product"], cand["week_start"])]
cand = cand.sort_values("max_abs_z", ascending=False).head(15)
cand.to_csv(TAB / "undocumented_unusual_weeks.csv", index=False)
display(cand[["product", "week_start", "volume", "unusual_series", "distinctive_words", "price_move_within_4wk", "top_text"]])
""")

nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s) for kind, s in CELLS]
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "task2_signals_and_prices.ipynb"
nbf.write(nb, out)
print("wrote", out)

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis import prices as P, signals as S
from analysis.classify import LABELS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 110)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

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

The sample was drawn by `labeling/make_label_sample.py`: a stratified batch of posts (product × keyword stratum, one
text per thread), then keyword-enriched batches of posts and comments so that the rare price and supply classes are
represented. Two AI annotators labelled every row independently with `labeling/codebook.md` (annotator 2 never saw
annotator 1's labels). A team member then decided every row where they disagreed plus a random sample of rows where
they agreed. `label` is the human decision where a person reviewed the row and the agreed AI label elsewhere.
""")

code(r"""
from sklearn.metrics import cohen_kappa_score
lab = pd.read_csv("labeling/labels.csv")
lab = lab[lab["label"].isin(LABELS)]
print(f"{len(lab)} labelled texts ({(lab.record_type == 'post').sum()} posts, {(lab.record_type == 'comment').sum()} comments)")
display(pd.crosstab([lab["batch"], lab["product"]], lab["label"], margins=True).reindex(columns=LABELS + ["All"]))
print(f"AI annotator agreement: {lab['ai_agree'].mean():.1%} of rows, Cohen's kappa "
      f"{cohen_kappa_score(lab['label_ai1'], lab['label_ai2']):.2f}")
display(pd.crosstab(lab["label_ai1"], lab["label_ai2"], rownames=["annotator 1"], colnames=["annotator 2"]))
print(lab["label_source"].value_counts().to_string())
check = lab[lab["reviewed"] & lab["ai_agree"]]
if len(check):
    ok = (check["human_label"] == check["label_ai1"]).mean()
    print(f"Random check of {len(check)} rows where the AI annotators agreed: the reviewer kept the AI label for {ok:.1%}.")
dis = lab[lab["reviewed"] & ~lab["ai_agree"]]
if len(dis):
    print(f"Disagreements decided by the reviewer: {len(dis)}; reviewer sided with annotator 1 in "
          f"{(dis['human_label'] == dis['label_ai1']).mean():.1%}, annotator 2 in "
          f"{(dis['human_label'] == dis['label_ai2']).mean():.1%}, neither in "
          f"{((dis['human_label'] != dis['label_ai1']) & (dis['human_label'] != dis['label_ai2'])).mean():.1%}.")
""")

md(r"""
## 2. Which classifier?

All models were evaluated with **thread-grouped stratified 5-fold cross-validation**: a post and the comments under
it never fall on both sides of a train/test split, so near-duplicate context cannot leak into the test fold.
The 95% interval for macro-F1 comes from a cluster bootstrap over threads.
""")

code(r"""
cv = pd.read_csv(TAB / "model_cv_metrics.csv", index_col="model")
names = {"keyword": "Keyword dictionary (Task 1 rules)", "zeroshot": "Zero-shot BART-large-MNLI",
         "embed_lr": "Sentence embeddings + logistic regression", "finetune": "Fine-tuned DistilRoBERTa"}
show = cv.rename(index=names)[["accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "cohen_kappa",
                              "f1_price", "f1_supply", "f1_demand", "f1_other"]].round(3)
display(show)
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
print("Accuracy by product (selected model, out of fold):")
display(oof.merge(lab[["obs_id", "product"]], on="obs_id").assign(ok=lambda d: d["label"] == d[selected])
        .groupby("product")["ok"].mean().round(3))
""")

md(r"""
## 3. Classify every observation

The selected model, retrained on the whole labelled sample, classifies every observation of the event dataset.
Sentiment is `P(positive) − P(negative)` from `cardiffnlp/twitter-roberta-base-sentiment-latest`.
""")

code(r"""
ev = pd.read_csv("data/processed/event_dataset.csv", low_memory=False, parse_dates=["week_start", "date"])
cls = pd.read_csv("data/processed/classified_observations.csv.gz")
obs = ev.merge(cls[["obs_id", "pred_label", "sentiment"] + [f"p_{l}" for l in LABELS]], on="obs_id", how="left")
print(f"{len(obs):,} observations, {obs['record_id'].nunique():,} Reddit records, {obs['thread_id'].nunique():,} threads")
dist = pd.crosstab(obs["product"], obs["pred_label"]).reindex(columns=LABELS)
dist_share = dist.div(dist.sum(1), axis=0).round(3)
display(pd.concat({"count": dist, "share": dist_share}, axis=1))
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
    for e in events_ref[events_ref["product"] == prod].itertuples():
        ax.axvspan(e.start_date, e.end_date, color="0.85", zorder=0)
        ax.text(e.start_date, ax.get_ylim()[1], f" {e.event_id}", va="top", fontsize=7, color="0.35")

fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)
for ax, prod in zip(axes, PRODUCTS):
    g = sig[sig["product"] == prod].set_index("week_start")
    for lab_ in ["price", "supply", "demand"]:
        ax.plot(g.index, g[f"n_{lab_}"], color=COLORS[lab_], lw=1.2, label=lab_)
    ax.set_ylabel(f"{prod}\nposts/comments per week")
    shade(ax, prod)
axes[0].legend(ncol=3, frameon=False, loc="upper right")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Weekly price, supply and demand signals on Reddit", y=0.995)
fig.tight_layout(); fig.savefig(FIG / "weekly_signals.png", dpi=200); plt.show()

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
                     "top_text": " | ".join(top["description"].str.slice(0, 140)),
                     "top_url": " ".join(top["url"])})
unusual = pd.DataFrame(rows).sort_values(["product", "week_start"])
unusual.to_csv(TAB / "unusual_weeks.csv", index=False)
print(f"{len(unusual)} unusual product-weeks; {(unusual['documented_event'] != '').sum()} fall within 2 weeks of a documented event")
display(unusual.sort_values("max_abs_z", ascending=False).head(25)[
    ["product", "week_start", "volume", "unusual_series", "distinctive_words", "documented_event"]])
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
    ax.plot(g["week_start"], g["index"], color="#18211d", lw=1.3)
    for mkt, gm in widx[(widx["product"] == prod) & (widx["market"] != "ALL")].groupby("market"):
        ax.plot(gm["week_start"], gm["index"], color="0.7", lw=0.5, zorder=0)
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
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Weekly shipments reaching US markets (USDA AMS movement reports)", y=0.995)
fig.tight_layout(); fig.savefig(FIG / "shipments.png", dpi=200); plt.show()
""")

md(r"""
## 7. Timing: does the online signal lead or lag the price?

**Series-level.** For each product we correlate a weekly signal (its deviation from the previous 8 weeks, z-score)
with the price index deviation (also an 8-week trailing z-score) shifted by *k* weeks:
corr(signal<sub>t</sub>, price<sub>t+k</sub>). *k > 0* means the online signal **leads** the price, *k < 0* that
it **lags**. Correlations outside ±1.96/√n are marked significant at about 5%.

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
        xc_rows.append({"product": prod, "signal": name, "r_same_week": round(r0, 3),
                        "best_lag_weeks": int(best["lag_weeks"]), "r_at_best_lag": round(best["r"], 3),
                        "significant": abs(best["r"]) > best["ci95"],
                        "reading": "signal leads price" if best["lag_weeks"] > 0 else
                                   "signal lags price" if best["lag_weeks"] < 0 else "same week"})
xcorr = pd.DataFrame(xc_rows)
xcorr.to_csv(TAB / "lead_lag_crosscorrelation.csv", index=False)
display(xcorr)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
for ax, prod in zip(axes, PRODUCTS):
    for name, colr in [("volume", "#2f7d4f"), ("supply posts", COLORS["supply"]), ("price posts", COLORS["price"]),
                       ("demand posts", COLORS["demand"])]:
        xc = xc_curves[(prod, name)]
        ax.plot(xc["lag_weeks"], xc["r"], color=colr, lw=1.2, label=name)
    ax.fill_between(xc["lag_weeks"], -xc["ci95"], xc["ci95"], color="0.9", zorder=0)
    ax.axvline(0, color="0.5", lw=0.6); ax.set_title(prod); ax.set_xlabel("lag k (weeks, + = signal leads)")
axes[0].set_ylabel("corr(signal$_t$, price$_{t+k}$)"); axes[-1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "lead_lag.png", dpi=200); plt.show()
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

Change of the pooled price index during the event window relative to its mean over the 8 weeks before the event
(mean and peak change), and the same calendar-window change in the other study years as a seasonality control
(`seasonally_adjusted_pct` = event-year change minus the average other-year change, in log points).
Per-market ranges show how uniform the move was across the 10 terminal markets.
""")

code(r"""
impact = []
for e in events_ref.itertuples():
    li = nat[nat["product"] == e.product].set_index("week_start")["log_index"]
    r = S.price_impact(li, e.start_date, e.end_date)
    mk = []
    for mkt, gm in widx[(widx["product"] == e.product) & (widx["market"] != "ALL")].groupby("market"):
        mk.append(S.price_impact(gm.set_index("week_start")["log_index"], e.start_date, e.end_date)["mean_change_pct"])
    mk = [m for m in mk if pd.notna(m)]
    impact.append({"event_id": e.event_id, "product": e.product, "event": e.event_name,
                   **{k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()},
                   "market_range_pct": f"{min(mk):.0f} to {max(mk):.0f}" if mk else ""})
impact = pd.DataFrame(impact)
impact.to_csv(TAB / "event_price_impact.csv", index=False)
display(impact)
""")

md(r"""
## 9. Supply shock or demand shift?

Economics tells supply shocks from demand shifts by how price and quantity move together: a **supply shock** raises
the price while less product reaches the market; a **demand shift** moves price and quantity in the same direction.
We first read each event from the market data alone, then ask whether the online talk tells the same story.

**Market reading.**
* *Did the price move unusually?* The event-window price change (section 8) is compared with the change in every
  other window of the same length in 2022–2024, measured the same way and not overlapping a documented event of the
  product. Percentile ≥ 90: unusual rise; ≤ 10: unusual fall.
* *Did quantity fall or rise?* Weekly national shipments, mean during the event vs. the 8 weeks before, minus the
  same calendar change in the other years (seasonal adjustment). Below −5%: fell; above +5%: rose.

| Price | Shipments | Market reading |
|---|---|---|
| unusual rise | fell | **supply shock** |
| unusual rise | flat or rose | **demand increase** |
| unusual fall | fell or flat | **demand decrease** |
| unusual fall | rose | **supply increase** |
| no unusual move | any | no clear price effect |

**Online reading.** Weekly counts of texts the classifier labels *supply* and *demand* during the event vs. the 8
weeks before, as a rate ratio with a 95% Poisson interval. A kind of talk "rose" when its interval is above 1 and it
rose more than background (*other*) talk about the product. Supply talk rising (at least as much as demand talk) →
*supply*; demand talk rising more → *demand*. Note that by our codebook, reports of recalls and products pulled from
shelves count as supply talk; people avoiding a product count as demand talk.
""")

code(r"""
def counts(prod, s0, s1):
    g = obs[(obs["product"] == prod) & (obs["week_start"] >= s0) & (obs["week_start"] <= s1)]
    return {l: int((g["pred_label"] == l).sum()) for l in LABELS} | {"n": len(g), "sentiment": g["sentiment"].mean()}

def wk(d):   # Monday of the week containing d
    d = pd.Timestamp(d); return d - pd.Timedelta(days=d.weekday())

fmt = lambda t: f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f})"
verdict = []
for e in events_ref.itertuples():
    s0, s1 = wk(e.start_date), wk(e.end_date)
    w_dur, w_pre = (s1 - s0).days // 7 + 1, 8
    others = [(r.start_date, r.end_date) for r in events_ref.itertuples() if r.product == e.product]
    # market reading
    li = nat[nat["product"] == e.product].set_index("week_start")["log_index"]
    p_pct = S.placebo_percentile(li, e.start_date, e.end_date, exclude=others)["percentile"]
    q = S.price_impact(ship_log[e.product], e.start_date, e.end_date)
    q_adj = q["seasonally_adjusted_pct"]
    p_up, p_down = p_pct >= 90, p_pct <= 10
    q_dir = "fell" if q_adj < -5 else "rose" if q_adj > 5 else "flat"
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
                    "price_change_pct": imp["mean_change_pct"], "price_percentile": round(p_pct),
                    "shipments_change_pct": round(q["mean_change_pct"], 1),
                    "shipments_seasonally_adjusted_pct": round(q_adj, 1), "shipments": q_dir,
                    "market_reading": market,
                    "supply_talk_rate_ratio": fmt(rr["supply"]), "demand_talk_rate_ratio": fmt(rr["demand"]),
                    "price_talk_rate_ratio": fmt(rr["price"]), "other_talk_rate_ratio": fmt(rr["other"]),
                    "supply_talk_before_during": f"{pre['supply']} / {dur['supply']}",
                    "demand_talk_before_during": f"{pre['demand']} / {dur['demand']}",
                    "weeks_before_during": f"{w_pre} / {w_dur}",
                    "online_reading": online, "readings_agree": agree, "expected": e.hypothesis})
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

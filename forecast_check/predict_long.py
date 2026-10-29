"""Forecast check: do online signals improve forecasts of the national wholesale price index, 2019-2026?

    python forecast_check/predict_long.py

Reads forecast_check/data/panel_long.csv. Walk-forward test from 2021-01-04 to 2026-09 in 4-week blocks: for each
block, models are trained only on weeks whose target was known before the block starts. Targets are the change in
the log price index 1 and 4 weeks ahead, and whether the 4-week change is a "big move" (at least the product's
85th percentile of absolute change in the training weeks). Every feature set contains the price history; the
other sets add one source at a time. Writes three tables to outputs/tables/:
  forecast_long_regression.csv  R2 against "no change", direction accuracy, and a block-bootstrap p-value that
                                the set beats price history alone with the same model
  forecast_long_bigmove.csv     AUC and average precision for big moves, overall and by year
  forecast_long_auc_gain.csv    AUC gain over price history alone with a block-bootstrap 95% interval
About 8 minutes on a laptop CPU.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.signals import trailing_z  # noqa: E402  (z-score against the previous 8 weeks only)

TEST_START = "2021-01-04"
TABLES = ROOT / "outputs/tables"
df = pd.read_csv(ROOT / "forecast_check/data/panel_long.csv", parse_dates=["week_start"]) \
       .sort_values(["product", "week_start"]).reset_index(drop=True)
gp = df["product"]
feats: dict[str, list[str]] = {}


def add(name, s, group):
    df[name] = s
    feats.setdefault(group, []).append(name)


def tz(s, floor, by=None):
    keys = [gp] if by is None else [gp, by]
    return s.groupby(keys).transform(lambda x: trailing_z(x, 8, min_sd=floor))


# P: price history, season, product
dl = df.log_index.groupby(gp).diff()
for k in range(4):
    add(f"dlog_l{k}", dl.groupby(gp).shift(k), "P")
for w in [8, 26, 52]:
    add(f"dev{w}", df.log_index - df.log_index.groupby(gp).transform(lambda s: s.rolling(w, min_periods=4).mean()), "P")
woy = df.week_start.dt.isocalendar().week.astype(float)
add("sin_w", np.sin(2 * np.pi * woy / 52), "P")
add("cos_w", np.cos(2 * np.pi * woy / 52), "P")
add("yoy", df.log_index - df.log_index.groupby(gp).shift(52), "P")
for p in sorted(df["product"].unique()):
    add(f"is_{p}", (gp == p).astype(float), "P")
# S: shipments
lt = np.log1p(df.total_lb)
add("ship_dlog", lt.groupby(gp).diff(), "S")
add("ship_dlog_l1", df.ship_dlog.groupby(gp).shift(1), "S")
add("ship_z", tz(lt, 0.05), "S")
add("ship_yoy", lt - lt.groupby(gp).shift(52), "S")
add("import_share", df.import_lb / df.total_lb, "S")
# R: Reddit (the weekly signals of the main analysis, their lags, and the comment-weighted supply count)
for c in ["z_volume", "z_n_price", "z_n_supply", "z_n_demand", "z_sentiment"]:
    add(c, df[c], "R")
    add(c + "_l1", df[c].groupby(gp).shift(1), "R")
for c in ["share_price", "share_supply", "share_demand", "neg_share", "sentiment"]:
    add(c + "_r", df[c], "R")
add("cw_supply_z", tz(np.log1p(df.ws.fillna(0)), 0.2), "R")
add("supply_spike", (df.z_n_supply >= 3).astype(float), "R")
# G: Google Trends (z within each download window)
for c in ["g_total", "g_price", "g_shortage", "g_recall"]:
    add(c + "_z", tz(np.log1p(df[c]), 0.2, df.g_segment), "G")
# N: GDELT news volume, share of all US news, tone
for t in ["all", "shortage", "recall", "price"]:
    add(f"news_{t}_z", tz(np.log1p(df[f"news_v_{t}"]), 0.1), "N")
    add(f"news_{t}_share_d", np.log1p(df[f"news_share_{t}"]).groupby(gp).diff(), "N")
    prev = df[f"news_tone_{t}"].groupby(gp).transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    add(f"news_{t}_tone", df[f"news_tone_{t}"] - prev, "N")
# F: FDA recalls, W: Wikipedia
add("fda_recall", (df.fda_recalls > 0).astype(float), "F")
add("fda_recall_l1", df.fda_recall.groupby(gp).shift(1), "F")
add("wiki_z", tz(np.log1p(df.wiki_views), 0.05), "W")

SETS = {"price only": ["P"], "+shipments": ["P", "S"], "+Reddit": ["P", "R"], "+Google": ["P", "G"],
        "+news": ["P", "N"], "+FDA recalls": ["P", "F"], "+Wikipedia": ["P", "W"],
        "+all online": ["P", "R", "G", "N", "F", "W"], "everything": ["P", "S", "R", "G", "N", "F", "W"]}


def reg_models():
    return {"ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
            "boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      min_samples_leaf=20, l2_regularization=1.0, random_state=0),
            "forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=0.5, random_state=0,
                                            n_jobs=-1)}


def clf_models():
    return {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
            "boosting": HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                       min_samples_leaf=20, random_state=0)}


def week_blocks(weeks, b=8):
    """Moving-block bootstrap over test weeks (blocks of 8 weeks keep the autocorrelation)."""
    weeks = np.asarray(weeks)
    return [np.flatnonzero((weeks >= s) & (weeks <= e)) for s, e in
            [(u[0], u[-1]) for u in np.array_split(np.unique(weeks), max(1, len(np.unique(weeks)) // b))]]


def boot_p_mse(y, base, new, rows_by_block, n=2000, seed=0):
    """One-sided p-value that `new` has lower squared error than `base`."""
    d = (y - base) ** 2 - (y - new) ** 2
    rng = np.random.default_rng(seed)
    nb = len(rows_by_block)
    stats = np.array([d[np.concatenate([rows_by_block[i] for i in rng.integers(0, nb, nb)])].mean() for _ in range(n)])
    return float(((stats - d.mean()) >= d.mean()).mean())


reg_rows, clf_rows, gain_rows = [], [], []
for h in [1, 4]:
    df["y"] = df.log_index.groupby(gp).shift(-h) - df.log_index
    weeks = sorted(df.week_start[df.week_start >= TEST_START].unique())
    recs = []
    for i in range(0, len(weeks), 4):
        blk = weeks[i:i + 4]
        cut = blk[0] - pd.Timedelta(weeks=h)  # training targets must be known before the block starts
        tr = df[df.week_start < cut].dropna(subset=["y"]).copy()
        te = df[df.week_start.isin(blk)].dropna(subset=["y"]).copy()
        if te.empty:
            continue
        thr = tr.groupby("product").y.apply(lambda s: s.abs().quantile(0.85))
        tr["big"] = (tr.y.abs() >= tr["product"].map(thr)).astype(int)
        te["big"] = (te.y.abs() >= te["product"].map(thr)).astype(int)
        rec = te[["product", "week_start", "y", "big"]].copy()
        for s, groups in SETS.items():
            cols = sum((feats[k] for k in groups), [])
            xtr, xte = tr[cols].fillna(0).values, te[cols].fillna(0).values
            for m, mod in reg_models().items():
                mod.fit(xtr, tr.y.values)
                rec[f"R|{s}|{m}"] = mod.predict(xte)
            if h == 4:
                for m, mod in clf_models().items():
                    mod.fit(xtr, tr.big.values)
                    rec[f"C|{s}|{m}"] = mod.predict_proba(xte)[:, 1]
        recs.append(rec)
    r = pd.concat(recs).sort_values(["week_start", "product"]).reset_index(drop=True)
    blocks = week_blocks(r.week_start.values)
    y = r.y.values
    for c in [c for c in r.columns if c.startswith("R|")]:
        _, s, m = c.split("|")
        p = r[c].values
        base = r[f"R|price only|{m}"].values
        reg_rows.append({"horizon_weeks": h, "features": s, "model": m, "n_test": len(y),
                         "R2_vs_no_change": 1 - np.mean((y - p) ** 2) / np.mean(y ** 2),
                         "direction_accuracy": np.mean(np.sign(p) == np.sign(y)),
                         "p_better_than_price_only": np.nan if s == "price only" else boot_p_mse(y, base, p, blocks)})
    if h == 4:
        years = sorted(r.week_start.dt.year.unique())
        for c in [c for c in r.columns if c.startswith("C|")]:
            _, s, m = c.split("|")
            row = {"features": s, "model": m, "n_test": len(r), "n_big": int(r.big.sum()),
                   "AUC": roc_auc_score(r.big, r[c]), "AP": average_precision_score(r.big, r[c])}
            for yr in years:
                sel = (r.week_start.dt.year == yr).values
                if r.big[sel].nunique() == 2:
                    row[f"AUC_{yr}"] = roc_auc_score(r.big[sel], r[c][sel])
            clf_rows.append(row)
            if s == "price only":
                continue
            a = r[f"C|price only|{m}"].values
            b, yb = r[c].values, r.big.values
            rng = np.random.default_rng(0)
            nb = len(blocks)
            ds = []
            for _ in range(1000):
                ix = np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)])
                ds.append(roc_auc_score(yb[ix], b[ix]) - roc_auc_score(yb[ix], a[ix]))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            gain_rows.append({"features": s, "model": m, "AUC_gain": roc_auc_score(yb, b) - roc_auc_score(yb, a),
                              "ci_low": lo, "ci_high": hi, "share_of_draws_no_gain": float(np.mean(np.array(ds) <= 0))})
    print(f"h={h}: test {r.week_start.min().date()} to {r.week_start.max().date()}, {len(r)} product-weeks", flush=True)

reg, clf, gain = pd.DataFrame(reg_rows), pd.DataFrame(clf_rows), pd.DataFrame(gain_rows)
reg.round(6).to_csv(TABLES / "forecast_long_regression.csv", index=False)
clf.round(6).to_csv(TABLES / "forecast_long_bigmove.csv", index=False)
gain.round(6).to_csv(TABLES / "forecast_long_auc_gain.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(reg.round(3).to_string(index=False))
print(clf.round(3).to_string(index=False))
print(gain.round(3).to_string(index=False))

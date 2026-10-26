"""USDA AMS terminal market price indices.

A terminal market report lists many price lines per product and day (different varieties, pack sizes, grades,
origins, organic/conventional). Averaging raw prices would mix these together, so we build a fixed-composition
index with a two-way fixed-effects (time-dummy) model:

    log(price_line) = item_series_effect + time_effect + error

where an item series is one (market, commodity, variety, package, size, origin, organic) combination.
The time effects, normalised to mean 0 over the study period, are the log price index; exp(index) * 100 is
an index with study-period mean ~ 100. The model is solved by alternating projections (iterative demeaning),
which handles the unbalanced panel without building a large design matrix.

Price per line = midpoint of the "mostly" range when reported, otherwise midpoint of the low-high range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_KEYS = ["market", "commodity", "variety", "package", "item_size", "origin", "organic"]


def load_prices(path) -> pd.DataFrame:
    p = pd.read_csv(path, parse_dates=["report_date"], low_memory=False)
    mostly = (p["mostly_low_price"] + p["mostly_high_price"]) / 2
    rng = (p["low_price"] + p["high_price"]) / 2
    p["price"] = mostly.where(mostly.notna(), rng)
    p = p[p["price"] > 0].copy()
    p["log_price"] = np.log(p["price"])
    for k in SERIES_KEYS:
        p[k] = p[k].fillna("NA").astype(str)
    p["series"] = p[SERIES_KEYS].agg("|".join, axis=1)
    p["week_start"] = (p["report_date"] - pd.to_timedelta(p["report_date"].dt.weekday, unit="D")).dt.normalize()
    return p


def fe_index(df: pd.DataFrame, time_col: str, n_iter: int = 50, min_obs_per_series: int = 10) -> pd.DataFrame:
    """Time-dummy log price index for the rows in df (one product, one or many markets)."""
    d = df.groupby("series").filter(lambda g: len(g) >= min_obs_per_series)[["series", time_col, "log_price"]]
    d = d.groupby(["series", time_col], as_index=False)["log_price"].mean()   # one value per series and period
    y = d["log_price"].to_numpy()
    s_codes, s_idx = np.unique(d["series"], return_inverse=True)
    t_codes, t_idx = np.unique(d[time_col], return_inverse=True)
    a = np.zeros(len(s_codes))
    b = np.zeros(len(t_codes))
    for _ in range(n_iter):
        a = np.bincount(s_idx, y - b[t_idx]) / np.bincount(s_idx)
        b_new = np.bincount(t_idx, y - a[s_idx]) / np.bincount(t_idx)
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    b -= b.mean()
    n_series = np.bincount(t_idx, minlength=len(t_codes))
    return pd.DataFrame({time_col: t_codes, "log_index": b, "index": 100 * np.exp(b), "n_series": n_series})


def weekly_indices(prices: pd.DataFrame) -> pd.DataFrame:
    """National (all markets pooled) and per-market weekly indices for every product."""
    out = []
    for prod, g in prices.groupby("product"):
        nat = fe_index(g, "week_start")
        nat["product"], nat["market"] = prod, "ALL"
        out.append(nat)
        for mkt, gm in g.groupby("market"):
            if gm["week_start"].nunique() < 52:
                continue
            idx = fe_index(gm, "week_start", min_obs_per_series=5)
            idx["product"], idx["market"] = prod, mkt
            out.append(idx)
    return pd.concat(out, ignore_index=True)


def daily_index(prices: pd.DataFrame) -> pd.DataFrame:
    """National daily index (report days only), used for event-level timing in days."""
    out = []
    for prod, g in prices.groupby("product"):
        idx = fe_index(g, "report_date")
        idx["product"] = prod
        out.append(idx)
    return pd.concat(out, ignore_index=True)

"""Weekly social-media market signals, anomaly detection, lead/lag timing and price-impact measurement."""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = ["price", "supply", "demand"]


def week_grid(start: str, end: str) -> pd.DatetimeIndex:
    s = pd.Timestamp(start)
    s = s - pd.Timedelta(days=s.weekday())
    return pd.date_range(s, pd.Timestamp(end) - pd.Timedelta(days=1), freq="W-MON")


def weekly_signals(obs: pd.DataFrame, post_counts: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """obs: one row per observation with columns product, week_start, pred_label, sentiment, record_type,
    collection_method. post_counts: week_start, n_posts (all posts in the listed subreddits)."""
    weeks = week_grid(start, end)
    denom = post_counts.groupby("week_start")["n_posts"].sum().reindex(weeks, fill_value=0)
    rows = []
    for prod, g in obs.groupby("product"):
        w = g.groupby("week_start")
        df = pd.DataFrame(index=weeks)
        df["volume"] = w.size().reindex(weeks, fill_value=0)
        listed_posts = g[(g["record_type"] == "post") & (g["collection_method"] == "subreddit_listing")]
        df["posts_per_10k"] = (listed_posts.groupby("week_start").size().reindex(weeks, fill_value=0)
                               / denom.replace(0, np.nan) * 1e4)
        df["sentiment"] = w["sentiment"].mean().reindex(weeks)
        df["neg_share"] = w["sentiment"].apply(lambda s: (s < -0.25).mean()).reindex(weeks)
        for lab in SIGNALS + ["other"]:
            df[f"n_{lab}"] = g[g["pred_label"] == lab].groupby("week_start").size().reindex(weeks, fill_value=0)
            df[f"share_{lab}"] = df[f"n_{lab}"] / df["volume"].replace(0, np.nan)
        df["product"] = prod
        rows.append(df.rename_axis("week_start").reset_index())
    return pd.concat(rows, ignore_index=True)


def trailing_z(x: pd.Series, window: int = 8, min_periods: int = 6) -> pd.Series:
    """z-score of each week against the mean and std of the previous `window` weeks (current week excluded)."""
    base = x.shift(1).rolling(window, min_periods=min_periods)
    sd = base.std().replace(0, np.nan)
    return (x - base.mean()) / sd


def classify_change(z: pd.Series, thr: float = 2.0) -> pd.Series:
    return pd.Series(np.select([z >= thr, z <= -thr], ["increase", "decrease"], "no change"), index=z.index)


def add_anomalies(sig: pd.DataFrame, window: int = 8, thr: float = 2.0) -> pd.DataFrame:
    sig = sig.sort_values(["product", "week_start"]).copy()
    cols = {"volume": np.log1p, "sentiment": lambda s: s, "n_price": np.log1p, "n_supply": np.log1p,
            "n_demand": np.log1p}
    for col, f in cols.items():
        z = sig.groupby("product")[col].transform(lambda s, f=f: trailing_z(f(s), window))
        sig[f"z_{col}"] = z
        sig[f"change_{col}"] = classify_change(z.fillna(0), thr)
    return sig


def price_anomalies(idx: pd.DataFrame, window: int = 8, thr: float = 2.0) -> pd.DataFrame:
    idx = idx.sort_values(["product", "market", "week_start"]).copy()
    g = idx.groupby(["product", "market"])["log_index"]
    idx["dlog_price"] = g.diff()
    idx["z_price"] = g.transform(lambda s: trailing_z(s, window))
    idx["change_price"] = classify_change(idx["z_price"].fillna(0), thr)
    return idx


def cross_correlation(signal: pd.Series, price: pd.Series, max_lag: int = 8) -> pd.DataFrame:
    """corr(signal_t, price_{t+k}). k > 0: the online signal LEADS the price by k weeks; k < 0: it lags."""
    s = (signal - signal.mean()) / signal.std()
    p = (price - price.mean()) / price.std()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([s, p.shift(-k)], axis=1).dropna()
        n = len(pair)
        r = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if n > 10 else np.nan
        rows.append({"lag_weeks": k, "r": r, "n": n, "ci95": 1.96 / np.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def first_crossing(series: pd.Series, start, end, thr: float = 2.0, direction: str = "abs"):
    s = series[(series.index >= start) & (series.index <= end)].dropna()
    hit = s[s.abs() >= thr] if direction == "abs" else s[s >= thr] if direction == "up" else s[s <= -thr]
    return hit.index[0] if len(hit) else pd.NaT


def price_impact(log_index: pd.Series, start, end, pre_weeks: int = 8) -> dict:
    """Change of the price index during [start, end] relative to the `pre_weeks` weeks before the event, plus
    the same calendar-window change in the other study years (seasonality control)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    def window_change(s0, s1):
        pre = log_index[(log_index.index >= s0 - pd.Timedelta(weeks=pre_weeks)) & (log_index.index < s0)]
        ev = log_index[(log_index.index >= s0) & (log_index.index <= s1)]
        if len(pre) < 3 or len(ev) < 1:
            return np.nan, np.nan
        return ev.mean() - pre.mean(), ev.max() - pre.mean() if abs(ev.max() - pre.mean()) >= abs(
            ev.min() - pre.mean()) else ev.min() - pre.mean()

    mean_chg, peak_chg = window_change(start, end)
    other = []
    for dy in (-2, -1, 1, 2):
        s0, s1 = start + pd.DateOffset(years=dy), end + pd.DateOffset(years=dy)
        if s0 - pd.Timedelta(weeks=pre_weeks) >= log_index.index.min() and s1 <= log_index.index.max():
            other.append(window_change(s0, s1)[0])
    seasonal = np.nanmean(other) if other else np.nan
    return {"mean_change_pct": 100 * (np.exp(mean_chg) - 1), "peak_change_pct": 100 * (np.exp(peak_chg) - 1),
            "same_window_other_years_pct": 100 * (np.exp(seasonal) - 1) if other else np.nan,
            "seasonally_adjusted_pct": 100 * (np.exp(mean_chg - seasonal) - 1) if other else np.nan,
            "n_comparison_years": len(other)}

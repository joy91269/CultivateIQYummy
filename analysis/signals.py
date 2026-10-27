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
    """obs: one row per observation with columns product, week_start, p_price/p_supply/p_demand/p_other (class
    probabilities), sentiment, record_type, collection_method. post_counts: week_start, n_posts (all posts in the listed subreddits)."""
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
            # expected count = sum of the class probabilities (calibrated classifier; see notebook section 3)
            df[f"n_{lab}"] = w[f"p_{lab}"].sum().reindex(weeks, fill_value=0.0)
            df[f"share_{lab}"] = df[f"n_{lab}"] / df["volume"].replace(0, np.nan)
        df["product"] = prod
        rows.append(df.rename_axis("week_start").reset_index())
    return pd.concat(rows, ignore_index=True)


def trailing_z(x: pd.Series, window: int = 8, min_periods: int = 6, min_sd: float = 0.0) -> pd.Series:
    """z-score of each week against the mean and std of the previous `window` weeks (current week excluded).
    `min_sd` floors the baseline std so that a few unusually flat weeks cannot turn a small change into a huge z."""
    base = x.shift(1).rolling(window, min_periods=min_periods)
    sd = base.std().clip(lower=min_sd).replace(0, np.nan)
    return (x - base.mean()) / sd


def classify_change(z: pd.Series, thr: float = 2.0) -> pd.Series:
    return pd.Series(np.select([z >= thr, z <= -thr], ["increase", "decrease"], "no change"), index=z.index)


def add_anomalies(sig: pd.DataFrame, window: int = 8, thr: float = 2.0) -> pd.DataFrame:
    sig = sig.sort_values(["product", "week_start"]).copy()
    # series -> (transform, std floor). Counts are on the log scale (floor 0.2 = about 20%); sentiment in raw units.
    cols = {"volume": (np.log1p, 0.2), "sentiment": (lambda s: s, 0.05), "n_price": (np.log1p, 0.2),
            "n_supply": (np.log1p, 0.2), "n_demand": (np.log1p, 0.2)}
    for col, (f, floor) in cols.items():
        z = sig.groupby("product")[col].transform(lambda s, f=f, floor=floor: trailing_z(f(s), window, min_sd=floor))
        sig[f"z_{col}"] = z
        sig[f"change_{col}"] = classify_change(z.fillna(0), thr)
    return sig


def price_anomalies(idx: pd.DataFrame, window: int = 8, thr: float = 2.0) -> pd.DataFrame:
    idx = idx.sort_values(["product", "market", "week_start"]).copy()
    g = idx.groupby(["product", "market"])["log_index"]
    idx["dlog_price"] = g.diff()
    idx["z_price"] = g.transform(lambda s: trailing_z(s, window, min_sd=0.03))   # log points (~3%)
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


def price_impact(log_index: pd.Series, start, end, pre_weeks: int = 8, exclude=()) -> dict:
    """Change of the price index during [start, end] relative to the `pre_weeks` weeks before the event, plus
    the same calendar-window change in the other study years (seasonality control). Other-year windows that overlap
    a documented event in `exclude` [(start, end), ...] are not used as comparisons, so one event cannot distort the
    seasonal baseline of another."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    pre_td = pd.Timedelta(weeks=pre_weeks)

    def overlaps(s0, s1):
        return any(s0 - pre_td <= pd.Timestamp(e1) and s1 >= pd.Timestamp(e0) for e0, e1 in exclude)

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
        if (s0 - pre_td >= log_index.index.min() and s1 <= log_index.index.max()
                and not overlaps(s0, s1)):
            other.append(window_change(s0, s1)[0])
    seasonal = np.nanmean(other) if other else np.nan
    return {"mean_change_pct": 100 * (np.exp(mean_chg) - 1), "peak_change_pct": 100 * (np.exp(peak_chg) - 1),
            "same_window_other_years_pct": 100 * (np.exp(seasonal) - 1) if other else np.nan,
            "seasonally_adjusted_pct": 100 * (np.exp(mean_chg - seasonal) - 1) if other else np.nan,
            "n_comparison_years": len(other)}


def placebo_percentile(log_index: pd.Series, start, end, pre_weeks: int = 8, exclude=(),
                       stat: str = "mean_change_pct") -> dict:
    """Where the event's mean change of a log series (price index or shipments) falls among all windows of the
    same length, measured the same way
    (mean during the window vs. the `pre_weeks` before it), that start in other weeks of the study period and do not
    overlap a documented event window in `exclude` [(start, end), ...]. A percentile of 95 means the event-window
    change is larger than 95% of ordinary windows of that length. `stat`: "mean_change_pct" (raw change) or
    "seasonally_adjusted_pct" (change minus the same calendar window in the other years)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    ev = price_impact(log_index, start, end, pre_weeks, exclude)[stat]
    length, pre = end - start, pd.Timedelta(weeks=pre_weeks)
    changes = []
    for s0 in log_index.index:
        s1 = s0 + length
        if s0 - pre < log_index.index.min() or s1 > log_index.index.max():
            continue
        if any(s0 - pre <= pd.Timestamp(e1) and s1 >= pd.Timestamp(e0) for e0, e1 in exclude):
            continue
        changes.append(price_impact(log_index, s0, s1, pre_weeks, exclude)[stat])
    changes = np.array([c for c in changes if np.isfinite(c)])
    pct = 100 * (changes < ev).mean() if len(changes) and np.isfinite(ev) else np.nan
    return {"placebo_windows": len(changes), "percentile": pct}


def rate_ratio(n_during: int, weeks_during: int, n_before: int, weeks_before: int) -> tuple[float, float, float]:
    """Weekly-rate ratio during/before with a 95% Poisson CI (0.5 added to both counts)."""
    a, b = n_during + 0.5, n_before + 0.5
    rr = (a / weeks_during) / (b / weeks_before)
    se = np.sqrt(1 / a + 1 / b)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)

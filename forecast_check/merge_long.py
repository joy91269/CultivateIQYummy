"""Join the three collection periods into one weekly panel, 2019-01 to 2026-09.

    python forecast_check/merge_long.py [EARLY_DIR] [LATE_DIR]

EARLY_DIR and LATE_DIR are the copies made by build_period.sh (default forecast_check/periods/2019-2021 and
forecast_check/periods/2025-2026); 2022-2024 comes from this repository. Writes forecast_check/data/panel_long.csv:
one row per product x week with the national price index, shipments, Reddit, Google Trends, GDELT news,
FDA recalls and Wikipedia pageviews. The panel is committed, so predict_long.py runs without the period copies.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import prices as P, signals as S  # noqa: E402

HERE = ROOT / "forecast_check"
EARLY = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "periods/2019-2021"
LATE = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "periods/2025-2026"
PARTS = [EARLY, ROOT, LATE]
START, END = "2019-01-01", "2026-10-01"
LABELS = ["price", "supply", "demand", "other"]


def cat(name, **kw):
    frames = []
    for i, d in enumerate(PARTS):
        f = d / "data/processed" / name
        if not f.exists():
            raise SystemExit(f"missing: {f}")
        frames.append(pd.read_csv(f, low_memory=False, **kw).assign(segment=i))
    return pd.concat(frames, ignore_index=True)


# prices: one fixed-effects index over the whole period
prices = pd.concat([P.load_prices(d / "data/processed/ams_terminal_prices.csv.gz") for d in PARTS],
                   ignore_index=True).drop_duplicates()
idx = S.price_anomalies(P.weekly_indices(prices))
nat = idx[idx.market == "ALL"][["product", "week_start", "log_index", "n_series"]]

# shipments: re-aggregate the daily movement lines
mv = cat("ams_movement_daily.csv.gz", parse_dates=["ship_date"]).drop(columns="segment").drop_duplicates()
mv["week_start"] = mv["ship_date"] - pd.to_timedelta(mv["ship_date"].dt.weekday, unit="D")
sh = mv.pivot_table(index=["product", "week_start"], columns="import", values="pounds", aggfunc="sum", fill_value=0) \
       .rename(columns={False: "domestic_lb", True: "import_lb"}).reset_index()
sh["total_lb"] = sh.get("domestic_lb", 0) + sh.get("import_lb", 0)

# Reddit: same weekly signals as the main analysis; hand-reviewed labels replace classifier probabilities
ev = cat("event_dataset.csv", usecols=["obs_id", "record_id", "record_type", "collection_method", "week_start",
                                       "product", "score", "num_comments"], parse_dates=["week_start"])
cl = cat("classified_observations.csv.gz", usecols=["obs_id", "human_label", "sentiment"] + [f"p_{l}" for l in LABELS])
obs = ev.merge(cl.drop(columns="segment"), on="obs_id", how="left")
hl = obs.human_label.isin(LABELS)
for lab in LABELS:
    obs.loc[hl, f"p_{lab}"] = (obs.loc[hl, "human_label"] == lab).astype(float)
obs = obs.drop_duplicates(["obs_id"])
pc = cat("weekly_post_counts.csv", parse_dates=["week_start"]).groupby(["week_start", "subreddit"], as_index=False) \
       .n_posts.sum()
sig = S.add_anomalies(S.weekly_signals(obs, pc, START, END))
w = np.where(obs.num_comments.fillna(0) >= 10, 2.0, 1.0)  # comment-weighted supply count (section 13)
cw = obs.assign(ws=obs.p_supply * w).groupby(["product", "week_start"]).ws.sum().reset_index()
sig = sig.merge(cw, on=["product", "week_start"], how="left")

# Google Trends: each download window is scaled on its own, so keep the window id for per-window z-scores
g = cat("google_trends_weekly.csv", parse_dates=["week_start"])
g = g.sort_values("segment").drop_duplicates(["product", "series", "week_start"], keep="last")
g = g.pivot_table(index=["product", "week_start", "segment"], columns="series", values="interest") \
     .add_prefix("g_").reset_index().rename(columns={"segment": "g_segment"})

# GDELT / FDA / Wikipedia (daily -> weekly)
ex = pd.read_csv(HERE / "data/extra_daily.csv", parse_dates=["date"])
ex["week_start"] = ex["date"] - pd.to_timedelta(ex["date"].dt.weekday, unit="D")
gd = ex[ex.source == "gdelt"]
vol = gd[gd.measure == "TimelineVolRaw"].groupby(["product", "topic", "week_start"]) \
        .agg(v=("value", "sum"), n=("norm", "sum")).reset_index()
vol["share"] = vol.v / vol.n * 1e6  # articles per million US articles
ton = gd[gd.measure == "TimelineTone"].groupby(["product", "topic", "week_start"]).value.mean().rename("tone").reset_index()
gdw = vol.merge(ton, on=["product", "topic", "week_start"], how="outer") \
         .pivot_table(index=["product", "week_start"], columns="topic", values=["v", "share", "tone"])
gdw.columns = [f"news_{a}_{b}" for a, b in gdw.columns]
gdw = gdw.reset_index()
fda = ex[ex.source == "fda"].groupby(["product", "week_start"]).value.sum().rename("fda_recalls").reset_index()
wiki = ex[ex.source == "wikipedia"].groupby(["product", "week_start"]).value.sum().rename("wiki_views").reset_index()

panel = pd.MultiIndex.from_product([sorted(nat["product"].unique()), S.week_grid(START, END)],
                                   names=["product", "week_start"]).to_frame(index=False)
for t in [nat, sh, sig, g, gdw, fda, wiki]:
    panel = panel.merge(t, on=["product", "week_start"], how="left")
panel["fda_recalls"] = panel["fda_recalls"].fillna(0)
panel.to_csv(HERE / "data/panel_long.csv", index=False)
cov = panel.groupby(panel.week_start.dt.year)[["log_index", "total_lb", "volume", "g_total", "news_v_all", "wiki_views"]] \
           .apply(lambda d: d.notna().mean()).round(2)
print("share of product-weeks with data, by year\n", cov)
print("Reddit texts by year\n", panel.groupby(panel.week_start.dt.year).volume.sum())

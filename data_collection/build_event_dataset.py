"""Clean the raw Reddit data and build the fresh-produce social media event dataset (Task 1).

Steps
1. Load raw posts, comment-tree comments and city-subreddit comments; drop exact re-downloads (same id).
2. Remove bots, moderator/automated messages, deleted/removed text, NSFW posts, crossposts,
   self-promotion / paid-promotion / spam, link farms, very short texts and duplicated texts.
   Every removal is counted in data/processed/filter_log.csv.
3. One observation per (text, product) pair: a comment that mentions avocados and onions appears twice.
4. Tag each observation with market-event types using the keyword dictionary (build_keywords.py) and
   choose a primary event type.
5. Attach geography (city subreddit -> AMS terminal market; regional retailer subreddits; US state names
   mentioned in the text).
6. Write
     data/processed/event_dataset.csv      the event dataset
     data/processed/summary_table.csv/.md  number of observations per product and event type
     data/processed/filter_log.csv         how many records each cleaning rule removed

Usage (from the repository root):
    python data_collection/build_event_dataset.py
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from build_keywords import EventTagger
from common import PROCESSED, RAW, ProductMatcher, jsonl_files, load_yaml, read_jsonl

REDDIT = RAW / "reddit"

KNOWN_BOTS = {"automoderator", "remindmebot", "sneakpeekbot", "wikitextbot", "repostsleuthbot", "savevideo",
              "converter-bot", "timezone_bot", "b0trank", "goodbot_badbot", "haikusbot", "same_post_bot",
              "auto-summarizer", "tldrbot", "imdbot", "user-activity-bot", "profanitycounter", "pokesmotsbot"}
BOT_TEXT = re.compile(r"\bi am a bot\b|\bi'm a bot\b|this action was performed automatically|beep boop", re.I)
SELF_PROMO = re.compile(
    r"amzn\.to|\baffiliate\b|\breferral\b|\bref(?:erral)? code\b|\bpromo code\b|\bdiscount code\b|\buse (?:my )?code\b|"
    r"link in (?:my )?bio|\bdm me\b|check out my (?:shop|store|channel|blog|page|website|etsy)|subscribe to my|"
    r"onlyfans|t\.me/|whatsapp|#ad\b|\bpaid partnership\b|\bsponsored by\b|\bbuy now\b|\bshop now\b|"
    r"\bfree shipping\b|\bwholesale (?:price|supplier)s? (?:available|contact)", re.I)
URL = re.compile(r"https?://\S+")
US_STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
             "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
             "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
             "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
             "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
             "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
             "West Virginia", "Wisconsin", "Wyoming"]
STATE_RE = re.compile(r"\b(" + "|".join(US_STATES) + r")\b")
# subreddits of regional grocery chains: the state where most of their stores are
RETAILER_STATE = {"HEB": "Texas", "Publix": "Florida", "Wegmans": "New York"}
STATE_ABBR = {"CA": "California", "IL": "Illinois", "NY": "New York", "MA": "Massachusetts", "GA": "Georgia",
              "PA": "Pennsylvania", "MI": "Michigan", "FL": "Florida", "MD": "Maryland"}


def week_start(d: datetime) -> str:
    return (d.date() - timedelta(days=d.weekday())).isoformat()


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def load_raw() -> pd.DataFrame:
    sources = load_yaml("sources.yaml")
    group_of = {s: g for g, members in sources["post_subreddits"].items() for s in members}
    title_subs = set(sources.get("title_search_subreddits", []))
    group_of.update({s: "news" for s in title_subs})
    # the archive returns some subreddit names in a different case (e.g. "salads"); use the configured spelling
    city_subs = set(sources.get("geo_subreddits", {}))
    group_of.update({s: "city" for s in city_subs})
    canon = {s.lower(): s for s in group_of}
    rows = []
    for path in jsonl_files(REDDIT / "posts"):
        for p in read_jsonl(path):
            p["subreddit"] = canon.get(p["subreddit"].lower(), p["subreddit"])
            title, body = p.get("title") or "", p.get("selftext") or ""
            if body.strip() in ("[removed]", "[deleted]"):
                body = ""
            method = ("title_search" if p["subreddit"] in title_subs
                      else "city_listing" if p["subreddit"] in city_subs else "subreddit_listing")
            rows.append({"obs_key": f"p_{p['id']}", "record_type": "post", "collection_method": method,
                         "subreddit": p["subreddit"], "subreddit_group": group_of.get(p["subreddit"], "other"),
                         "thread_id": p["id"], "created_utc": p["created_utc"], "author": p.get("author"),
                         "title": title, "text": f"{title}\n{body}".strip(), "score": p.get("score"),
                         "num_comments": p.get("num_comments"), "distinguished": p.get("distinguished"),
                         "crosspost": bool(p.get("crosspost_parent")), "over_18": bool(p.get("over_18")),
                         "url": f"https://www.reddit.com/comments/{p['id']}"})
    for folder, method in (("comments", "comment_tree"),):
        for path in jsonl_files(REDDIT / folder):
            for c in read_jsonl(path):
                post_id = (c.get("link_id") or "").replace("t3_", "")
                sub = canon.get(c["subreddit"].lower(), c["subreddit"])
                rows.append({"obs_key": f"c_{c['id']}", "record_type": "comment", "collection_method": method,
                             "subreddit": sub, "subreddit_group": group_of.get(sub, "other"),
                             "thread_id": post_id, "created_utc": c["created_utc"], "author": c.get("author"),
                             "title": "", "text": c.get("body") or "", "score": c.get("score"),
                             "num_comments": None, "distinguished": c.get("distinguished"), "crosspost": False,
                             "over_18": False, "url": f"https://www.reddit.com/comments/{post_id}/_/{c['id']}"})
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [("raw records (incl. re-downloads)", len(df))]

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        log.append((reason, int(mask.sum())))
        df = df[~mask].copy()

    df = df.sort_values(["created_utc", "obs_key"])
    drop(df.duplicated("obs_key"), "same record downloaded twice (same Reddit id)")
    author = df["author"].fillna("").str.lower()
    drop(author.isin(KNOWN_BOTS) | author.str.endswith("bot") | author.str.endswith("_bot")
         | df["text"].str.contains(BOT_TEXT), "bot accounts / automated messages")
    drop(df["distinguished"].isin(["moderator", "admin"]), "moderator / admin notices")
    drop(df["text"].str.strip().isin(["[removed]", "[deleted]", ""]), "deleted or removed text")
    drop(df["over_18"], "NSFW posts")
    drop(df["crosspost"], "crossposts (copies of a post in another subreddit)")
    drop(df["text"].str.contains(SELF_PROMO), "self-promotion, referral codes, paid promotion")
    drop(df["text"].str.count(URL) >= 3, "link farms (3+ URLs)")
    df["norm"] = df["text"].map(norm_text)
    drop(df["norm"].str.len() < 15, "too short to carry information (<15 characters)")
    # authors who post the same text repeatedly are spam accounts: drop all their copies
    rep = df.groupby(["author", "norm"])["obs_key"].transform("count")
    drop((rep >= 3) & (df["author"] != "[deleted]"), "same author repeating the same text 3+ times")
    drop(df.duplicated("norm"), "duplicate text (copy of an earlier record)")
    return df.drop(columns=["norm"]), pd.DataFrame(log, columns=["step", "records_removed"])


def build() -> None:
    raw = load_raw()
    df, log = clean(raw)
    matcher, tagger = ProductMatcher(), EventTagger()
    geo_cfg = load_yaml("sources.yaml")["geo_subreddits"]

    authors = sorted(a for a in df["author"].dropna().unique() if a != "[deleted]")
    author_id = {a: f"U{i:06d}" for i, a in enumerate(authors, 1)}   # pseudonymous author ids

    out = []
    for r in df.itertuples(index=False):
        products = matcher.match(r.text)
        if not products:            # e.g. the only product mention was inside an excluded phrase
            continue
        when = datetime.fromtimestamp(int(r.created_utc), tz=timezone.utc)
        types = tagger.tag(r.text)
        geo = geo_cfg.get(r.subreddit)
        states = sorted(set(STATE_RE.findall(r.text)))
        geo_state = (STATE_ABBR[geo["state"]] if geo else RETAILER_STATE.get(r.subreddit)
                     or (states[0] if len(states) == 1 else ""))
        geo_source = ("city subreddit" if geo else "regional retailer subreddit" if r.subreddit in RETAILER_STATE
                      else "state named in text" if len(states) == 1 else "")
        desc = re.sub(r"\s+", " ", r.text).strip()
        for prod in products:
            out.append({
                "obs_id": f"{r.obs_key}_{prod}", "record_id": r.obs_key, "source": "reddit",
                "record_type": r.record_type, "collection_method": r.collection_method,
                "subreddit": r.subreddit, "subreddit_group": r.subreddit_group, "thread_id": r.thread_id,
                "created_utc": int(r.created_utc), "date": when.date().isoformat(), "week_start": week_start(when),
                "product": prod, "n_products_mentioned": len(products),
                "description": desc[:280] + ("..." if len(desc) > 280 else ""), "text": r.text,
                "event_types": "|".join(types) if types else "none", "primary_event_type": tagger.primary(types),
                "geo_city": geo["market"] if geo else "", "geo_state": geo_state, "geo_source": geo_source,
                "ams_market": geo["market"] if geo else "",
                "score": r.score, "num_comments": r.num_comments,
                "author_id": author_id.get(r.author, "deleted"), "url": r.url,
            })
    events = pd.DataFrame(out).sort_values(["product", "created_utc"]).reset_index(drop=True)
    log.loc[len(log)] = ("records whose only product match was an excluded phrase",
                         int(len(df) - events["record_id"].nunique()))
    log.loc[len(log)] = ("FINAL cleaned records", int(events["record_id"].nunique()))
    log.loc[len(log)] = ("FINAL observations (record x product)", len(events))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_csv(PROCESSED / "event_dataset.csv", index=False)
    log.to_csv(PROCESSED / "filter_log.csv", index=False)
    write_summary(events)
    # weekly total number of posts in the fully listed topical subreddits: denominator for normalised volume
    # (city subreddits are general-interest communities and are left out)
    city_subs = set(load_yaml("sources.yaml").get("geo_subreddits", {}))
    counts = [pd.read_csv(p) for p in sorted((REDDIT / "post_counts").glob("*.csv")) if p.stem not in city_subs]
    if counts:
        pc = pd.concat(counts)
        pc.to_csv(PROCESSED / "weekly_post_counts.csv", index=False)
    print(log.to_string(index=False))


def write_summary(events: pd.DataFrame) -> None:
    order = ["price_change", "supply_shortage", "demand_change", "weather_event", "food_safety_recall",
             "marketing_promotion", "other"]
    prim = (events.pivot_table(index="product", columns="primary_event_type", values="obs_id", aggfunc="count",
                               fill_value=0).reindex(columns=order, fill_value=0))
    prim["total_observations"] = prim.sum(axis=1)
    anym = pd.DataFrame({t: events.groupby("product")["event_types"].apply(lambda s, t=t: s.str.contains(t).sum())
                         for t in order[:-1]})
    anym.columns = [f"any_{c}" for c in anym.columns]
    extra = events.groupby("product").agg(posts=("record_type", lambda s: (s == "post").sum()),
                                          comments=("record_type", lambda s: (s == "comment").sum()),
                                          with_geography=("geo_state", lambda s: (s != "").sum()),
                                          first_date=("date", "min"), last_date=("date", "max"))
    summary = prim.join(anym).join(extra)
    summary.loc["ALL"] = summary.sum(numeric_only=True)
    summary.loc["ALL", ["first_date", "last_date"]] = [events["date"].min(), events["date"].max()]
    summary.to_csv(PROCESSED / "summary_table.csv")
    with open(PROCESSED / "summary_table.md", "w") as f:
        f.write("# Number of observations per product and market-event type\n\n")
        f.write("Primary event type = first matching type in the order food safety > weather > supply > price > "
                "promotion > demand; `any_*` columns count every observation whose text matches the type.\n\n")
        f.write(summary.to_markdown())
        f.write("\n")


if __name__ == "__main__":
    build()

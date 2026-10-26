"""Collect Reddit posts and comments about the study products from the Arctic Shift archive.

Usage (from the repository root):
    python data_collection/collect_reddit.py posts   # page through all posts of the subreddits in config/sources.yaml
    python data_collection/collect_reddit.py titles  # title keyword search in subreddits too large to page (r/news)
    python data_collection/collect_reddit.py trees   # download comment trees of product posts, keep product comments
    python data_collection/collect_reddit.py geo     # page through all posts of the city subreddits (geography)
    python data_collection/collect_reddit.py all     # all steps in order

Every step is resumable: progress is stored in data/raw/reddit/_state/, so an interrupted run continues
where it stopped. Outputs (JSON Lines, one record per line):
    data/raw/reddit/posts/<subreddit>.jsonl          matched posts
    data/raw/reddit/post_counts/<subreddit>.csv      total posts per ISO week (denominator for volume)
    data/raw/reddit/comments/<subreddit>.jsonl       product comments found inside matched posts
City subreddits (step geo) are paged exactly like the topical subreddits; their posts and counts go to the same
folders and are marked as city posts when the event dataset is built.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

from common import (RAW, ProductMatcher, USER_AGENT, append_jsonl, http_get_json, load_state, load_yaml,
                    read_jsonl, save_state, study_period)

OUT = RAW / "reddit"
STATE = OUT / "_state"
POST_FIELDS = ("id,created_utc,title,selftext,author,score,num_comments,subreddit,link_flair_text,url,"
               "distinguished,crosspost_parent,over_18")
COMMENT_FIELDS = "id,created_utc,body,author,score,link_id,parent_id,subreddit,distinguished"


def week_start(ts: int) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    return (d - timedelta(days=d.weekday())).isoformat()   # Monday of the ISO week


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- posts
def collect_posts(session: requests.Session, cfg: dict, matcher: ProductMatcher, subs: list[str]) -> None:
    start, end = study_period()
    base = cfg["arctic_shift"]["base_url"] + "/api/posts/search"
    pause = cfg["arctic_shift"]["request_pause_s"]
    for sub in subs:
        state_path = STATE / f"posts_{sub}.json"
        st = load_state(state_path) or {"last_ts": int(start.timestamp()) - 1, "edge_ids": [], "done": False,
                                        "weekly_counts": {}, "n_seen": 0, "n_matched": 0}
        if st["done"]:
            log(f"posts r/{sub}: already done ({st['n_seen']} posts, {st['n_matched']} matched)")
            continue
        log(f"posts r/{sub}: resuming after {datetime.fromtimestamp(st['last_ts'], tz=timezone.utc):%Y-%m-%d}")
        weekly = Counter(st["weekly_counts"])
        while True:
            params = {"subreddit": sub, "after": st["last_ts"], "before": int(end.timestamp()),
                      "limit": 100, "sort": "asc", "fields": POST_FIELDS}
            data = http_get_json(session, base, params)["data"] or []
            new = [p for p in data if p["id"] not in set(st["edge_ids"])]
            if not data or not new:
                if data and not new:          # all results share the boundary timestamp; step forward
                    st["last_ts"] += 1
                    st["edge_ids"] = []
                    continue
                st["done"] = True
                break
            matched = []
            for p in new:
                weekly[week_start(p["created_utc"])] += 1
                prods = matcher.match(f"{p.get('title') or ''}\n{p.get('selftext') or ''}")
                if prods:
                    p["products"] = prods
                    matched.append(p)
            append_jsonl(OUT / "posts" / f"{sub}.jsonl", matched)
            st["n_seen"] += len(new)
            st["n_matched"] += len(matched)
            st["last_ts"] = max(p["created_utc"] for p in data)
            st["edge_ids"] = [p["id"] for p in data if p["created_utc"] == st["last_ts"]]
            st["weekly_counts"] = dict(weekly)
            save_state(state_path, st)
            if st["n_seen"] % 5000 < len(new):
                log(f"  r/{sub}: {st['n_seen']} posts, {st['n_matched']} matched, at "
                    f"{datetime.fromtimestamp(st['last_ts'], tz=timezone.utc):%Y-%m-%d}")
            if len(data) < 100:
                st["done"] = True
                break
            time.sleep(pause)
        st["weekly_counts"] = dict(weekly)
        save_state(state_path, st)
        (OUT / "post_counts").mkdir(parents=True, exist_ok=True)
        with open(OUT / "post_counts" / f"{sub}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["week_start", "subreddit", "n_posts"])
            for wk in sorted(weekly):
                w.writerow([wk, sub, weekly[wk]])
        log(f"posts r/{sub}: done ({st['n_seen']} posts, {st['n_matched']} matched)")


# --------------------------------------------------------------------------------------------- trees
def flatten_tree(nodes: list, depth: int = 0):
    for node in nodes:
        if node.get("kind") != "t1":
            continue
        c = node["data"]
        yield c, depth
        replies = c.get("replies")
        if isinstance(replies, dict):
            yield from flatten_tree(replies.get("data", {}).get("children", []), depth + 1)


def collect_trees(session: requests.Session, cfg: dict, matcher: ProductMatcher, shard: str = "0/1") -> None:
    base = cfg["arctic_shift"]["base_url"] + "/api/comments/tree"
    pause = cfg["arctic_shift"]["request_pause_s"]
    min_c = cfg["arctic_shift"]["min_comments_for_tree"]
    keep = ["id", "created_utc", "body", "author", "score", "link_id", "parent_id", "subreddit", "distinguished"]
    k, n = (int(x) for x in shard.split("/"))
    for path in sorted((OUT / "posts").glob("*.jsonl"))[k::n]:
        sub = path.stem
        state_path = STATE / f"trees_{sub}.json"
        st = load_state(state_path) or {"done_ids": []}
        done = set(st["done_ids"])
        # only threads that are ABOUT a product (product named in the title) and have a discussion
        posts = [p for p in read_jsonl(path) if (p.get("num_comments") or 0) >= min_c and p["id"] not in done
                 and matcher.match(p.get("title"))]
        if not posts:
            continue
        log(f"trees r/{sub}: {len(posts)} posts to fetch")
        for i, p in enumerate(posts):
            payload = http_get_json(session, base, {"link_id": f"t3_{p['id']}", "limit": 9999})
            rows = []
            for c, depth in flatten_tree(payload.get("data") or []):
                prods = matcher.match(c.get("body"))
                if prods:
                    row = {k: c.get(k) for k in keep}
                    row.update({"products": prods, "depth": depth, "post_products": p["products"]})
                    rows.append(row)
            append_jsonl(OUT / "comments" / f"{sub}.jsonl", rows)
            done.add(p["id"])
            if i % 50 == 0 or i == len(posts) - 1:
                st["done_ids"] = sorted(done)
                save_state(state_path, st)
                log(f"  r/{sub}: {i + 1}/{len(posts)} trees")
            time.sleep(pause)


# --------------------------------------------------------------------------------------------- title search
def collect_titles(session: requests.Session, cfg: dict, matcher: ProductMatcher) -> None:
    """Title keyword search (one query per product term and year) for subreddits too large to page through."""
    start, end = study_period()
    base = cfg["arctic_shift"]["base_url"] + "/api/posts/search"
    terms = [t for spec in load_yaml("products.yaml")["products"].values() for t in spec["title_terms"]]
    years = list(range(start.year, end.year if (end.month, end.day) == (1, 1) else end.year + 1))
    for sub in cfg.get("title_search_subreddits", []):
        state_path = STATE / f"titles_{sub}.json"
        st = load_state(state_path) or {"done": [], "n_matched": 0}
        for term in terms:
            for year in years:
                key = f"{term}|{year}"
                if key in st["done"]:
                    continue
                after, before = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()) - 1, \
                    int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
                seen: set[str] = set()
                while True:
                    params = {"subreddit": sub, "title": term, "after": after, "before": before, "limit": 100,
                              "sort": "asc", "fields": POST_FIELDS}
                    data = http_get_json(session, base, params, max_tries=60, min_wait=10)["data"] or []
                    new = [p for p in data if p["id"] not in seen]
                    rows = []
                    for p in new:
                        seen.add(p["id"])
                        prods = matcher.match(f"{p.get('title') or ''}\n{p.get('selftext') or ''}")
                        if prods:
                            p["products"] = prods
                            rows.append(p)
                    append_jsonl(OUT / "posts" / f"{sub}.jsonl", rows)
                    st["n_matched"] += len(rows)
                    if len(data) < 100 or not new:
                        break
                    after = max(p["created_utc"] for p in data) - 1
                st["done"].append(key)
                save_state(state_path, st)
                log(f"titles r/{sub} '{term}' {year}: {len(seen)} posts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["posts", "titles", "trees", "geo", "all"])
    ap.add_argument("--group", nargs="*", default=None,
                    help="posts step only: restrict to these subreddit groups from config/sources.yaml "
                         "(lets several groups be collected in parallel processes)")
    ap.add_argument("--subs", nargs="*", default=None,
                    help="posts and geo steps: restrict to these subreddits (disjoint lists can run in parallel)")
    ap.add_argument("--shard", default="0/1",
                    help="trees step only: k/n processes every n-th subreddit file starting at k (parallel runs)")
    args = ap.parse_args()
    cfg = load_yaml("sources.yaml")
    matcher = ProductMatcher()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    only = set(args.subs) if args.subs else None
    if args.step in ("posts", "all"):
        subs = [s for g, members in cfg["post_subreddits"].items() if not args.group or g in args.group
                for s in members]
        collect_posts(session, cfg, matcher, [s for s in subs if not only or s in only])
    if args.step in ("geo", "all"):
        collect_posts(session, cfg, matcher, [s for s in cfg["geo_subreddits"] if not only or s in only])
    if args.step in ("titles", "all"):
        collect_titles(session, cfg, matcher)
    if args.step in ("trees", "all"):
        collect_trees(session, cfg, matcher, args.shard)


if __name__ == "__main__":
    sys.exit(main())

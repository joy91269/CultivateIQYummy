"""Shared helpers for the data-collection scripts.

All paths are resolved relative to the repository root, so the scripts can be run from anywhere:
    python data_collection/collect_reddit.py posts
"""
from __future__ import annotations

import gzip
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

USER_AGENT = "CultivateIQ-student-data-challenge/1.0 (academic research; contact via GitHub joy91269)"


def load_yaml(name: str) -> dict:
    with open(CONFIG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def study_period() -> tuple[datetime, datetime]:
    p = load_yaml("products.yaml")["study_period"]
    start = datetime.fromisoformat(p["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(p["end"]).replace(tzinfo=timezone.utc)
    return start, end


class ProductMatcher:
    """Decides which products a piece of text mentions, using the regexes in config/products.yaml."""

    def __init__(self) -> None:
        products = load_yaml("products.yaml")["products"]
        self.patterns = {}
        for pid, spec in products.items():
            inc = re.compile(spec["regex"], re.IGNORECASE)
            exc = re.compile(spec["exclude_regex"], re.IGNORECASE) if spec.get("exclude_regex") else None
            self.patterns[pid] = (inc, exc)

    def match(self, text: str | None) -> list[str]:
        if not text:
            return []
        hits = []
        for pid, (inc, exc) in self.patterns.items():
            t = exc.sub(" ", text) if exc else text
            if inc.search(t):
                hits.append(pid)
        return hits


def http_get_json(session: requests.Session, url: str, params: dict, max_tries: int = 8,
                  min_wait: float = 2.0) -> dict:
    """GET with polite retry handling for Arctic Shift.

    Arctic Shift answers expensive queries that exceed the current budget with
    {"error": "Timeout. Maybe slow down a bit"} (HTTP 422) or HTTP 429. In both cases the
    X-RateLimit-Reset header says how many seconds remain until the budget resets.
    """
    last_err = None
    for attempt in range(max_tries):
        try:
            r = session.get(url, params=params, timeout=120)
        except requests.RequestException as e:  # network hiccup
            last_err = str(e)
            time.sleep(min_wait * (attempt + 1))
            continue
        try:
            payload = r.json()
        except ValueError:
            payload = {"error": f"non-JSON response (HTTP {r.status_code})"}
        if r.status_code == 200 and payload.get("error") in (None, ""):
            return payload
        last_err = payload.get("error") or f"HTTP {r.status_code}"
        reset = r.headers.get("X-RateLimit-Reset")
        wait = float(reset) + 1.0 if reset and reset.isdigit() else min_wait * (attempt + 1)
        time.sleep(max(wait, min_wait))
    raise RuntimeError(f"giving up on {url} {params}: {last_err}")


def utc_date(ts: int | float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file; falls back to the gzipped snapshot `<path>.gz` committed to the repository."""
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_files(folder: Path) -> list[Path]:
    """All `<name>.jsonl` files in a folder, including those present only as `<name>.jsonl.gz`."""
    stems = {p.name.removesuffix(".gz") for p in folder.glob("*.jsonl*")}
    return [folder / s for s in sorted(stems)]


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)

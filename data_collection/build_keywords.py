"""Build the product and market-event keyword dictionary (Task 1 deliverable).

Reads config/products.yaml and config/event_keywords.yaml and writes
    data/processed/keyword_dictionary.csv    one row per term: category, key, term (regex), description
    data/processed/keyword_dictionary.json   same content as a nested dictionary

It also exposes `EventTagger`, which the dataset builder uses to tag each observation with event types.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from common import PROCESSED, load_yaml

# When several event types match one observation, the primary type is the first one in this order.
# Rarer, more specific event types come first so that they are not hidden by generic price/demand words.
PRIORITY = ["food_safety_recall", "weather_event", "supply_shortage", "price_change", "marketing_promotion",
            "demand_change"]


def keyword_table() -> pd.DataFrame:
    rows = []
    for pid, spec in load_yaml("products.yaml")["products"].items():
        rows.append({"category": "product", "key": pid, "term": spec["regex"],
                     "description": f"{spec['display']} (exclusions: {spec.get('exclude_regex', '')})"})
    for etype, spec in load_yaml("event_keywords.yaml")["event_types"].items():
        for term in spec["terms"]:
            rows.append({"category": "event_type", "key": etype, "term": term, "description": spec["description"]})
    return pd.DataFrame(rows)


class EventTagger:
    """Tags text with the market-event types whose keywords it contains."""

    def __init__(self) -> None:
        spec = load_yaml("event_keywords.yaml")["event_types"]
        self.patterns = {
            etype: re.compile(r"(?<![\w$])(?:" + "|".join(s["terms"]) + r")(?!\w)", re.IGNORECASE)
            for etype, s in spec.items()
        }

    def tag(self, text: str) -> list[str]:
        found = [etype for etype, pat in self.patterns.items() if pat.search(text or "")]
        return sorted(found, key=PRIORITY.index)

    @staticmethod
    def primary(types: list[str]) -> str:
        return types[0] if types else "other"


def main() -> None:
    table = keyword_table()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "keyword_dictionary.csv", index=False)
    nested: dict = {"products": {}, "event_types": {}}
    for _, r in table.iterrows():
        bucket = "products" if r["category"] == "product" else "event_types"
        nested[bucket].setdefault(r["key"], []).append(r["term"])
    (PROCESSED / "keyword_dictionary.json").write_text(json.dumps(nested, indent=2))
    print(table.groupby(["category", "key"]).size().to_string())
    print(f"wrote {len(table)} terms -> data/processed/keyword_dictionary.csv/.json")


if __name__ == "__main__":
    main()

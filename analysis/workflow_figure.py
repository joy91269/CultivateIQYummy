"""Draw the data-collection and analysis workflow figure: outputs/figures/data_workflow.png

    python -m analysis.workflow_figure
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

W, H = 3.6, 1.6
COLS = [0.2, 4.2, 8.2, 12.2]
ROWS = [6.4, 3.7, 1.0]
BOXES = {
    # key: (column, row, title, body, fill)
    "cfg": (0, 0, "Configuration", "4 products and their regexes,\n30 subreddits, event keyword\ndictionary, 8 documented events", "#eceeed"),
    "list": (1, 0, "Arctic Shift: every post", "30 subreddits, 2022-2024,\nabout 1 M posts scanned;\nkeep posts naming a product", "#e4eff9"),
    "trees": (2, 0, "Comment trees", "threads naming a product in the\ntitle with >= 3 comments;\nkeep product comments", "#e4eff9"),
    "ams": (3, 0, "USDA AMS MARS API", "10 terminal markets, daily\nreports, about 457 k\nwholesale price lines", "#fbf0e1"),
    "title": (0, 1, "Arctic Shift: title search", "r/news headlines,\none query per product\nword and year", "#e4eff9"),
    "clean": (1, 1, "Clean", "bots, moderators, NSFW, crossposts,\nself-promotion, link farms,\nshort texts, duplicates", "#eceeed"),
    "tag": (2, 1, "Task 1 event dataset", "one row per text x product;\nkeyword event types, geography,\nsummary table", "#e6f2ea"),
    "price": (3, 1, "Price index", "fixed-composition weekly index\n(item-series and week\nfixed effects)", "#fbf0e1"),
    "label": (1, 2, "Labelled sample", "stratified, one text per thread;\nAI first pass + human review", "#f5e6f3"),
    "model": (2, 2, "Classifiers + sentiment", "keyword, zero-shot, embeddings,\nfine-tuned DistilRoBERTa;\nthread-grouped 5-fold CV", "#f5e6f3"),
    "compare": (3, 2, "Task 2 comparison", "weekly signals, unusual weeks,\nlead/lag, price impact,\nsupply shock vs demand shift", "#e6f2ea"),
}
EDGES = [("cfg", "r", "list", "l"), ("list", "r", "trees", "l"), ("cfg", "b", "title", "t"),
         ("list", "b", "clean", "t"), ("title", "r", "clean", "l"), ("trees", "b", "clean", "r"),
         ("clean", "r", "tag", "l"), ("ams", "b", "price", "t"), ("tag", "b", "model", "t"),
         ("tag", "b", "label", "r"), ("label", "r", "model", "l"), ("model", "r", "compare", "l"),
         ("price", "b", "compare", "t")]


def box_xy(key):
    c, r = BOXES[key][:2]
    return COLS[c], ROWS[r]


def anchor(key, side):
    x, y = box_xy(key)
    return {"l": (x, y + H / 2), "r": (x + W, y + H / 2), "t": (x + W / 2, y + H), "b": (x + W / 2, y)}[side]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0.8, 8.6)
    ax.axis("off")
    for key, (c, r, title, body, color) in BOXES.items():
        x, y = COLS[c], ROWS[r]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec="#5d6b64", lw=0.8))
        ax.text(x + 0.15, y + H - 0.18, title, fontsize=10.5, fontweight="bold", va="top", color="#18211d")
        ax.text(x + 0.15, y + H - 0.58, body, fontsize=8.5, va="top", color="#2c3632", linespacing=1.35)
    for a, sa, b, sb in EDGES:
        ax.add_patch(FancyArrowPatch(anchor(a, sa), anchor(b, sb), arrowstyle="-|>", mutation_scale=11, lw=0.9,
                                     color="#5d6b64", shrinkA=2, shrinkB=2))
    ax.text(0.2, 8.55, "Data collection and analysis workflow", fontsize=13, fontweight="bold", va="top")
    out = ROOT / "outputs/figures/data_workflow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

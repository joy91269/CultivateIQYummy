"""Draw the data-collection and analysis workflow figure: outputs/figures/data_workflow.png

Drawn at print size (6.5 in wide) so that its text matches the report's figure text.

    python -m analysis.workflow_figure
"""
from pathlib import Path

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

for _f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf"]:
    if Path(_f).exists():
        matplotlib.font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"]})

FIG_W, FIG_H = 6.5, 2.55          # inches; axes coordinates are inches too
W, H = 1.5, 0.6                  # box size
COLS = [0.02, 1.66, 3.30, 4.94]
ROWS = [1.90, 0.98, 0.06]
SOURCE, STEP, OUTPUT = "#e8f0f7", "#f1f1ef", "#e7f1e8"
BOXES = {
    # key: (column, row, title, body, fill)
    "cfg": (0, 0, "Configuration", "4 products, 38 subreddits,\nevent keywords, 8 events", STEP),
    "list": (1, 0, "Reddit posts", "Arctic Shift archive, 2022-2024;\nposts naming a product", SOURCE),
    "trees": (2, 0, "Comment threads", "comments under product\nthreads with 3+ comments", SOURCE),
    "ams": (3, 0, "USDA AMS reports", "terminal market prices in\n10 cities; movement reports", SOURCE),
    "title": (0, 1, "r/news headlines", "headline search per\nproduct and year", SOURCE),
    "clean": (1, 1, "Cleaning", "bots, spam, promotion\nand duplicates removed", STEP),
    "tag": (2, 1, "Event dataset (Task 1)", "92,424 texts, 98,006 rows\nwith event tags", OUTPUT),
    "price": (3, 1, "Prices and shipments", "weekly price index and\nweekly shipments", STEP),
    "label": (1, 2, "Labelled sample", "1,458 texts; 4 AI annotators,\npeople decide disagreements", STEP),
    "model": (2, 2, "Classifier", "sentence encoder +\nlogistic regression", STEP),
    "compare": (3, 2, "Comparison (Task 2)", "signals against prices and\nshipments around 8 events", OUTPUT),
}
EDGES = [("cfg", "r", "list", "l"), ("list", "r", "trees", "l"), ("cfg", "b", "title", "t"),
         ("list", "b", "clean", "t"), ("title", "r", "clean", "l"), ("trees", "bl", "clean", "tr"),
         ("clean", "r", "tag", "l"), ("ams", "b", "price", "t"), ("tag", "b", "model", "t"),
         ("tag", "bl", "label", "tr"), ("label", "r", "model", "l"), ("model", "r", "compare", "l"),
         ("price", "b", "compare", "t")]


def box_xy(key):
    c, r = BOXES[key][:2]
    return COLS[c], ROWS[r]


def anchor(key, side):
    x, y = box_xy(key)
    return {"l": (x, y + H / 2), "r": (x + W, y + H / 2), "t": (x + W / 2, y + H), "b": (x + W / 2, y),
            "bl": (x + 0.2, y), "tr": (x + W - 0.2, y + H)}[side]


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    for key, (c, r, title, body, color) in BOXES.items():
        x, y = COLS[c], ROWS[r]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=color, ec="#6b7570", lw=0.6))
        ax.text(x + 0.08, y + H - 0.09, title, fontsize=7, fontweight="bold", va="top", color="#18211d")
        ax.text(x + 0.08, y + H - 0.27, body, fontsize=6.2, va="top", color="#2c3632", linespacing=1.3)
    for a, sa, b, sb in EDGES:
        ax.add_patch(FancyArrowPatch(anchor(a, sa), anchor(b, sb), arrowstyle="-|>", mutation_scale=7, lw=0.6,
                                     color="#6b7570", shrinkA=1.5, shrinkB=1.5))
    out = ROOT / "outputs/figures/data_workflow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()

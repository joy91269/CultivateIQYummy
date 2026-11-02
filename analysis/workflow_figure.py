"""Draw the study overview figure: outputs/figures/data_workflow.png

Four stages from left to right: collect, label (AI annotators and people), classify, compare. The icons are drawn
here with matplotlib patches, so the figure needs no image files. Drawn at print size (6.5 in wide).

    python -m analysis.workflow_figure
"""
from pathlib import Path

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]

for _f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf"]:
    if Path(_f).exists():
        matplotlib.font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"]})

INK, MUTED, PANEL = "#1d2723", "#5b6762", "#f3f4f1"
PRICE, SUPPLY, DEMAND = "#b4690e", "#1f6fb2", "#9a3d8f"
AI, PERSON, MARKET = "#4f6d8a", "#2f7d4f", "#b4690e"
FIG_W, FIG_H = 6.5, 2.35
PANEL_W, PANEL_H, GAP = 1.5, 2.2, 0.16
TOP = FIG_H - 0.07


# ---------- icons (x, y = centre of the icon's base; s = icon height in inches) ----------
def person(ax, x, y, s, color=PERSON):
    ax.add_patch(Wedge((x, y), 0.36 * s, 0, 180, fc=color, ec="none"))
    ax.add_patch(Circle((x, y + 0.58 * s), 0.2 * s, fc=color, ec="none"))


def robot(ax, x, y, s, color=AI):
    ax.add_patch(FancyBboxPatch((x - 0.26 * s, y), 0.52 * s, 0.3 * s, boxstyle=f"round,pad=0,rounding_size={0.06 * s}",
                                fc=color, ec="none"))
    hx, hy, hw, hh = x - 0.32 * s, y + 0.36 * s, 0.64 * s, 0.44 * s
    ax.add_patch(FancyBboxPatch((hx, hy), hw, hh, boxstyle=f"round,pad=0,rounding_size={0.1 * s}", fc=color, ec="none"))
    for dx in (-0.13, 0.13):
        ax.add_patch(Circle((x + dx * s, hy + 0.24 * s), 0.065 * s, fc="white", ec="none"))
    ax.plot([x, x], [hy + hh, hy + hh + 0.1 * s], color=color, lw=0.9, solid_capstyle="round")
    ax.add_patch(Circle((x, hy + hh + 0.13 * s), 0.045 * s, fc=color, ec="none"))


def bubbles(ax, x, y, s):
    for dx, dy, c, w in [(-0.18, 0.28, "#6d8fae", 0.62), (0.14, 0.0, "#9fb6cc", 0.62)]:
        bx, by = x + dx * s - w * s / 2, y + dy * s + 0.12 * s
        ax.add_patch(FancyBboxPatch((bx, by), w * s, 0.36 * s, boxstyle=f"round,pad=0,rounding_size={0.1 * s}",
                                    fc=c, ec="none"))
        tail = [(bx + 0.12 * s, by + 0.02 * s), (bx + 0.26 * s, by + 0.02 * s), (bx + 0.08 * s, by - 0.12 * s)]
        ax.add_patch(Polygon(tail, closed=True, fc=c, ec="none"))
        for k in range(3):
            ax.add_patch(Circle((bx + (0.17 + 0.14 * k) * s, by + 0.18 * s), 0.035 * s, fc="white", ec="none"))


def price_chart(ax, x, y, s):
    w = 0.95 * s
    ax.add_patch(Rectangle((x - w / 2, y), w, 0.75 * s, fc="white", ec=MUTED, lw=0.6))
    for k, h in enumerate([0.18, 0.26, 0.22, 0.4, 0.52]):
        ax.add_patch(Rectangle((x - w / 2 + (0.08 + 0.17 * k) * s, y + 0.05 * s), 0.11 * s, h * s,
                               fc="#7fa98f", ec="none"))
    xs = [x - w / 2 + (0.13 + 0.17 * k) * s for k in range(5)]
    ax.plot(xs, [y + v * s for v in (0.42, 0.38, 0.47, 0.6, 0.66)], color=MARKET, lw=1.1)


def network(ax, x, y, s):
    layers = [[0.15, 0.45, 0.75], [0.3, 0.6], [0.15, 0.35, 0.55, 0.75]]
    cols = [x - 0.38 * s, x, x + 0.38 * s]
    pts = [[(cx, y + v * s) for v in layer] for cx, layer in zip(cols, layers)]
    for a, b in [(0, 1), (1, 2)]:
        for p in pts[a]:
            for q in pts[b]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#b9c2bd", lw=0.4, zorder=1)
    for layer in pts[:2]:
        for p in layer:
            ax.add_patch(Circle(p, 0.06 * s, fc=AI, ec="none", zorder=2))
    for p, c in zip(pts[2], [PRICE, SUPPLY, DEMAND, "#9aa3a0"]):
        ax.add_patch(Circle(p, 0.06 * s, fc=c, ec="none", zorder=2))


def signal_chart(ax, x, y, s):
    w = 1.0 * s
    ax.add_patch(Rectangle((x - w / 2, y), w, 0.75 * s, fc="white", ec=MUTED, lw=0.6))
    ax.add_patch(Rectangle((x - w / 2 + 0.52 * s, y), 0.2 * s, 0.75 * s, fc="#e1e3df", ec="none"))
    xs = [x - w / 2 + k * w / 9 for k in range(10)]
    ax.plot(xs, [y + v * s for v in (0.2, 0.22, 0.18, 0.24, 0.21, 0.3, 0.62, 0.55, 0.3, 0.24)], color=SUPPLY, lw=1.0)
    ax.plot(xs, [y + v * s for v in (0.34, 0.33, 0.36, 0.33, 0.35, 0.4, 0.5, 0.64, 0.58, 0.45)], color=MARKET, lw=1.0)


# ---------- layout ----------
def panel(ax, k, title):
    x0 = 0.02 + k * (PANEL_W + GAP)
    ax.add_patch(FancyBboxPatch((x0, TOP - PANEL_H), PANEL_W, PANEL_H, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=PANEL, ec="none"))
    ax.text(x0 + 0.1, TOP - 0.1, f"{k + 1}", fontsize=8, fontweight="bold", color="white", va="top", ha="center",
            bbox=dict(boxstyle="circle,pad=0.18", fc=INK, ec="none"))
    ax.text(x0 + 0.24, TOP - 0.1, title, fontsize=8, fontweight="bold", color=INK, va="top")
    return x0


def note(ax, x, y, text, size=6.3, color=MUTED, ha="center", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="top", linespacing=1.25, fontweight=weight)


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1 Collect
    x0 = panel(ax, 0, "Collect")
    cx = x0 + PANEL_W / 2
    bubbles(ax, cx, 1.47, 0.42)
    note(ax, cx, 1.42, "Reddit posts and comments,\n2019-2026: 225,926 texts\nafter cleaning", color=INK)
    price_chart(ax, cx, 0.5, 0.42)
    note(ax, cx, 0.44, "USDA wholesale prices and\nshipments, 10 markets", color=INK)

    # 2 Label: AI annotators label, people decide
    x0 = panel(ax, 1, "Label")
    cx = x0 + PANEL_W / 2
    for k, name in enumerate(["Claude", "Claude", "Codex", "ChatGPT"]):
        rx = x0 + 0.24 + k * 0.34
        robot(ax, rx, 1.43, 0.34)
        note(ax, rx, 1.38, name, size=5.4)
    note(ax, cx, 1.24, "four AI annotators label", size=6.3, color=AI)
    ax.add_patch(FancyArrowPatch((cx, 1.07), (cx, 0.87), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=MUTED))
    for dx in (-0.17, 0.17):
        person(ax, cx + dx, 0.55, 0.36)
    note(ax, cx, 0.5, "team members review\nselected labels", size=6.3, color=PERSON)
    note(ax, cx, 0.2, "1,458 labelled texts", size=6.3, color=INK, weight="bold")

    # 3 Classify
    x0 = panel(ax, 2, "Classify")
    cx = x0 + PANEL_W / 2
    network(ax, cx, 1.27, 0.55)
    note(ax, cx, 1.2, "sentence encoder +\nlogistic regression", color=INK)
    ax.add_patch(FancyArrowPatch((cx, 0.9), (cx, 0.74), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=MUTED))
    for k, (lab, c) in enumerate([("price", PRICE), ("supply", SUPPLY), ("demand", DEMAND), ("other", "#9aa3a0")]):
        ax.text(x0 + 0.21 + k * 0.36, 0.64, lab, fontsize=5.8, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.1", fc=c, ec="none"))
    note(ax, cx, 0.46, "for every text; summed\ninto weekly signals", color=INK)

    # 4 Compare
    x0 = panel(ax, 3, "Compare")
    cx = x0 + PANEL_W / 2
    signal_chart(ax, cx, 1.25, 0.6)
    ax.text(cx - 0.3, 1.71, "supply talk", fontsize=5.6, color=SUPPLY, ha="center", va="bottom")
    ax.text(cx + 0.2, 1.71, "price", fontsize=5.6, color=MARKET, ha="center", va="bottom")
    note(ax, cx, 1.18, "weekly signals against prices\nand shipments, 33 events", color=INK)
    for k, q in enumerate(["Supply shock or demand shift", "Lead or lag", "Size of the price impact"]):
        note(ax, x0 + 0.12, 0.8 - k * 0.19, "• " + q, size=6.3, color=INK, ha="left")

    for k in range(3):
        xa = 0.02 + (k + 1) * PANEL_W + k * GAP
        ax.add_patch(FancyArrowPatch((xa + 0.01, TOP - PANEL_H / 2), (xa + GAP - 0.01, TOP - PANEL_H / 2),
                                     arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))

    out = ROOT / "outputs/figures/data_workflow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()

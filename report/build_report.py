"""Typeset the report as a paper-style PDF from the Markdown export of the report document.

    python report/build_report.py report/report.md          # writes report/report.pdf

What it does to the Markdown before typesetting:
- the first '# ' heading becomes the title; the byline and the draft note under it are dropped;
- the 'Summary' section becomes the abstract;
- each image placeholder of the export becomes the matching figure in outputs/figures (FIGURES below), and the
  paragraph 'Figure N. ...' right after it becomes its caption;
- a paragraph 'Table N. ...' right before a table becomes that table's caption;
- the appendix starts on a new page.
Figures and tables are numbered by LaTeX. Tables are three-line (booktabs) floats (report/tables.lua); the page
layout is in report/preamble.tex. Needs pandoc and XeLaTeX.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "report"
FIGURES = [  # (words in the image's alt text, figure file)
    ("shocks", "two_supply_shocks.png"),
    ("workflow", "data_workflow.png"),
    ("shipments", "prices_and_shipments.png"),
    ("signals", "weekly_signals.png"),
    ("lead", "lead_lag.png"),
    ("google", "google_trends.png"),
]
SUBTITLE = "Cultivate IQ \u2013 NAFSI Track 3 (Supply Chain)"
AUTHOR = "Jiuyi Zheng, Gan Xu, Zhiguang Liu, Yizhen Wang"
DATE = "November 2026"

CAPTION_FIG = re.compile(r"^\*?Figure\s+\d+[.:]\s*(.+?)\*?$")
CAPTION_TAB = re.compile(r"^\*?Table\s+\d+[.:]\s*(.+?)\*?$")
IMAGE = re.compile(r"^(?:&#91;|\\?\[)image:\s*(.+?)\\?\]$")


def figure_file(alt: str) -> str:
    low = alt.lower()
    for words, name in FIGURES:
        if words in low:
            return str(ROOT / "outputs/figures" / name)
    raise SystemExit(f"no figure file matches the image '{alt}'; add it to FIGURES in {__file__}")


def yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prepare(md: str) -> str:
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", md.strip())]
    title, abstract, out = "", [], []
    in_summary = in_refs = False
    i = 0
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
        if b.startswith("# ") and not title:
            title = b[2:].strip()
        elif not out and not in_summary and (re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}", b) or b.startswith("*Working draft")):
            pass                                                   # byline and draft note
        elif re.match(r"^##\s+Summary\s*$", b):
            in_summary = True
        elif in_summary and not b.startswith("#"):
            abstract.append(b)
        else:
            in_summary = False
            m_img = IMAGE.match(b)
            m_tab = CAPTION_TAB.match(b)
            if m_img:
                alt = m_img.group(1).strip()
                m_cap = CAPTION_FIG.match(nxt)
                caption = m_cap.group(1).strip() if m_cap else alt
                out.append(f"![{caption}]({figure_file(alt)}){{width=100%}}")
                if m_cap:
                    i += 1
            elif m_tab and nxt.startswith("|"):
                out.append(f"Table: {m_tab.group(1).strip()}")
            else:
                if re.match(r"^##\s+Appendix\s*$", b):
                    out.append("\\clearpage")
                if b.startswith("#"):
                    in_refs = bool(re.match(r"^##\s+References\s*$", b))
                elif in_refs:
                    b = "\\noindent\\hangindent=1.5em\\hangafter=1 " + b      # hanging indent for references
                out.append(b)
        i += 1
    meta = ["---", f"title: {yaml_str(title)}", f"subtitle: {yaml_str(SUBTITLE)}", f"author: {yaml_str(AUTHOR)}",
            f"date: {yaml_str(DATE)}"]
    if abstract:
        meta.append("abstract: |")
        for para in abstract:
            meta += ["  " + line for line in para.splitlines()] + ["  "]
    meta.append("---")
    return "\n".join(meta) + "\n\n" + "\n\n".join(out) + "\n"


def main(src: Path) -> None:
    prepared = src.with_suffix(".prepared.md")
    prepared.write_text(prepare(src.read_text(encoding="utf-8")), encoding="utf-8")
    pdf = src.with_suffix(".pdf")
    cmd = ["pandoc", str(prepared), "-f", "markdown-tex_math_dollars-smart+pipe_tables", "-o", str(pdf),
           "--pdf-engine=xelatex", f"--lua-filter={HERE / 'tables.lua'}", "-H", str(HERE / "preamble.tex"),
           "--shift-heading-level-by=-1", "-V", "documentclass=article", "-V", "fontsize=11pt", "-V", "indent=true", "-V", "fontfamily=fontspec", "-V", "mainfont=TeX Gyre Termes",
           "-V", "sansfont=TeX Gyre Heros", "-V", "mathfont=Latin Modern Math",
           "-V", "geometry:margin=1in", "-V", "colorlinks=true", "-V", "linkcolor=black",
           "-V", "urlcolor=[rgb]{0.1,0.25,0.5}"]
    subprocess.run(cmd, check=True)
    print("wrote", pdf)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.md")

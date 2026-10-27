"""Build the label-review page (a self-contained HTML file) from labeling/label_sample.csv.

Two AI annotators labelled every row independently (`label_ai1`, `label_ai2`). The page asks a person to decide
the rows flagged `needs_review` (all disagreements plus a random check of agreements): accept the agreed label or
pick one. Reviews are saved to the page's shared store (collection `reviews`, one document
per obs_id) and exported back with labeling/collect_reviews.py into labeling/labels.csv.

Usage: python labeling/build_review_page.py   ->  labeling/review_page.html
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r"""<title>Produce Signal Labels</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one review card at a time, sticky progress bar on top, codebook below. */
:root {
  --bg: #f4f6f5; --surface: #ffffff; --ink: #18211d; --muted: #5d6b64; --line: #d8dfdb;
  --accent: #2f7d4f; --accent-ink: #ffffff;
  --price: #b4690e; --supply: #1f6fb2; --demand: #9a3d8f; --other: #6b7570;
  --price-bg: #fbf0e1; --supply-bg: #e4eff9; --demand-bg: #f5e6f3; --other-bg: #eceeed;
  --sans: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111614; --surface: #19201d; --ink: #e6ece9; --muted: #9aa8a1; --line: #2c3632;
  --accent: #5fb985; --accent-ink: #0d1411;
  --price: #e9a24a; --supply: #6aaef0; --demand: #d886cc; --other: #a4aea9;
  --price-bg: #33261a; --supply-bg: #182a3a; --demand-bg: #31202f; --other-bg: #242b28; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111614; --surface: #19201d; --ink: #e6ece9; --muted: #9aa8a1; --line: #2c3632;
  --accent: #5fb985; --accent-ink: #0d1411;
  --price: #e9a24a; --supply: #6aaef0; --demand: #d886cc; --other: #a4aea9;
  --price-bg: #33261a; --supply-bg: #182a3a; --demand-bg: #31202f; --other-bg: #242b28; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg) }
.wrap { max-width: 860px; margin: 0 auto; padding-block: 20px 48px; display: grid; gap: 18px }
header.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  padding-block: 12px 10px; border-bottom: 1px solid var(--line); display: grid; gap: 10px }
h1 { font-size: 1.25rem; margin: 0; text-wrap: balance; letter-spacing: -0.01em }
.row { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center }
.meter { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; flex: 1 1 200px }
.meter > i { display: block; height: 100%; background: var(--accent); width: 0 }
.count { font-family: var(--mono); font-size: 0.85rem; color: var(--muted); font-variant-numeric: tabular-nums }
select, input[type=text] { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 8px }
.guide { display: grid; gap: 6px; font-size: 0.9rem; background: var(--demand-bg); border-left: 3px solid var(--demand);
  border-radius: 6px; padding: 10px 12px }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px; display: grid; gap: 14px; min-width: 0 }
.meta { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: 0.85rem; color: var(--muted); align-items: center }
.meta .mono { font-family: var(--mono) }
.chip { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 8px;
  border-radius: 999px; background: var(--other-bg); color: var(--ink) }
.text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 46vh; overflow-y: auto; padding: 12px 14px;
  border-left: 3px solid var(--line); background: var(--bg); border-radius: 0 6px 6px 0; font-size: 0.97rem }
mark { background: color-mix(in srgb, var(--accent) 25%, transparent); color: inherit; border-radius: 2px; padding: 0 1px }
.ai { font-size: 0.9rem; color: var(--muted) }
.ai b { color: var(--ink) }
.labels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px }
@media (max-width: 560px) { .labels { grid-template-columns: repeat(2, minmax(0, 1fr)) } }
.lab { font: inherit; font-weight: 600; border: 2px solid var(--line); background: var(--surface); color: var(--ink);
  border-radius: 8px; padding: 10px 8px; cursor: pointer; display: grid; gap: 2px; text-align: left }
.lab small { font-weight: 400; color: var(--muted); font-size: 0.75rem }
.lab kbd { font-family: var(--mono); font-size: 0.72rem; color: var(--muted) }
.lab.proposed { border-style: dashed }
.lab[data-l=price].on, .lab[data-l=price].proposed { border-color: var(--price) } .lab[data-l=price].on { background: var(--price-bg) }
.lab[data-l=supply].on, .lab[data-l=supply].proposed { border-color: var(--supply) } .lab[data-l=supply].on { background: var(--supply-bg) }
.lab[data-l=demand].on, .lab[data-l=demand].proposed { border-color: var(--demand) } .lab[data-l=demand].on { background: var(--demand-bg) }
.lab[data-l=other].on, .lab[data-l=other].proposed { border-color: var(--other) } .lab[data-l=other].on { background: var(--other-bg) }
.lab.on { border-style: solid }
.nav { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between }
.btn { font: inherit; font-weight: 600; border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  border-radius: 6px; padding: 8px 14px; cursor: pointer }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent) }
.btn:disabled { opacity: 0.45; cursor: default }
button:focus-visible, select:focus-visible, input:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }
.status { font-size: 0.85rem; color: var(--muted); min-height: 1.3em }
.notice { border: 1px solid var(--line); background: var(--surface); border-radius: 8px; padding: 10px 12px; font-size: 0.9rem }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px }
summary { cursor: pointer; font-weight: 600 }
.code { overflow-x: auto } table { border-collapse: collapse; font-size: 0.88rem; width: 100%; min-width: 520px }
td, th { border-top: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted) }
a { color: var(--accent) }
@media (prefers-reduced-motion: no-preference) { .meter > i { transition: width .25s ease } }
</style>

<div class="wrap">
  <header class="bar">
    <div class="row" style="justify-content: space-between">
      <h1>Produce signal labels</h1>
      <span class="count" id="count">0 / 0 reviewed</span>
    </div>
    <div class="row">
      <div class="meter" aria-hidden="true"><i id="meter"></i></div>
      <label class="count" for="filter">Show</label>
      <select id="filter">
        <option value="todo">To review (not done yet)</option>
        <option value="recheck">Second look: demand or other</option>
        <option value="adjudicate">Two annotators disagree: which is right?</option>
        <option value="queue">All texts to review</option>
        <option value="all">All texts, including ones not flagged</option>
        <option value="changed">Where my label differs from both AIs</option>
        <option value="avocado">Avocados</option>
        <option value="lettuce">Lettuce</option>
        <option value="onion">Onions</option>
        <option value="cucumber">Cucumbers</option>
      </select>
    </div>
  </header>

  <div class="notice" id="notice" hidden></div>

  <section class="card" id="card" aria-live="polite">
    <div class="meta">
      <span class="chip" id="prod">product</span>
      <span id="where"></span>
      <span class="mono" id="date"></span>
      <span class="mono" id="pos"></span>
      <a id="link" href="#" target="_blank" rel="noopener">Open on Reddit</a>
    </div>
    <div class="guide" id="guide" hidden><b>Second look: demand or other?</b>
      <span><b>Demand</b>: the text is about people buying, eating or wanting <i>this</i> product more or less. The
      product is the star of a recipe or meal idea ("what do I do with 10 lb of avocados", a cucumber-salad recipe),
      stocking up, cravings, a trend, a deal or free-guac promotion without a price, avoiding it after a scare.</span>
      <span><b>Other</b>: the product is just there. One ingredient among several, cooking or storage tips, home
      gardening, jokes, complaints about a restaurant meal, a product that is only onion- or avocado-flavoured.</span>
      <span>Still unsure? Choose other.</span></div>
    <div class="text" id="text"></div>
    <div class="ai" id="ai"></div>
    <div class="labels" role="group" aria-label="Label">
      <button class="lab" data-l="price" id="b-price"><span>Price</span><small>prices, deals, affordability</small><kbd>1</kbd></button>
      <button class="lab" data-l="supply" id="b-supply"><span>Supply</span><small>shortage, crop, recall, imports</small><kbd>2</kbd></button>
      <button class="lab" data-l="demand" id="b-demand"><span>Demand</span><small>buying, cravings, trends</small><kbd>3</kbd></button>
      <button class="lab" data-l="other" id="b-other"><span>Other</span><small>incidental mention</small><kbd>4</kbd></button>
    </div>
    <input type="text" id="note" placeholder="Note for hard cases (optional)" aria-label="Note">
    <div class="nav">
      <button class="btn" id="prev">← Previous</button>
      <span class="status" id="status"></span>
      <div class="row">
        <button class="btn primary" id="accept">Accept agreed label <kbd style="font-family:var(--mono);font-size:.75rem">Enter</kbd></button>
        <button class="btn" id="next">Next →</button>
      </div>
    </div>
  </section>

  <details>
    <summary>Codebook</summary>
    <div class="code">
      <table>
        <tr><th>Label</th><th>Use it when the text, about this product…</th></tr>
        <tr><td><b>Price</b></td><td>talks about its price, a price change, affordability, sticker shock, a price comparison, or a discount with a price.</td></tr>
        <tr><td><b>Supply</b></td><td>talks about availability, shortages, empty shelves, quality problems from supply, crop or weather conditions, harvest, imports/tariffs, transport, recalls or the product being pulled.</td></tr>
        <tr><td><b>Demand</b></td><td>is mainly about wanting, buying or eating more/less of it: purchases, stocking up, cravings, recipes or trends centred on the product, avoiding it after a scare.</td></tr>
        <tr><td><b>Other</b></td><td>no clear market signal: one ingredient among many, cooking technique, routine gardening tips, jokes, off-topic.</td></tr>
      </table>
      <p>If two signals appear, pick price, then supply, then demand. When unsure between a signal and Other, pick Other.</p>
    </div>
  </details>
</div>

<script>
const ITEMS = __ITEMS__;
const LABELS = ["price", "supply", "demand", "other"];
const PRODUCT_RE = {avocado: /\b(avocados?|guac(amole)?|hass)\b/gi, lettuce: /\b(lettuces?|romaine)\b/gi,
  onion: /\bonions?\b/gi, cucumber: /\b(cucumbers?|cukes?)\b/gi};
let reviews = {};      // obs_id -> {label, note}
let adjud = {};        // obs_id -> {label, note}: decisions between our label and the third annotator's
let order = [];        // indexes into ITEMS for the current filter
let cur = 0;           // position in order
let current = null;    // obs_id shown
let db = null, canWrite = true;
const $ = (id) => document.getElementById(id);

function esc(s) { return s.replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c])); }
function highlight(text, product) { return esc(text).replace(PRODUCT_RE[product], (m) => `<mark>${m}</mark>`); }

function rebuildOrder(keepId) {
  const f = $("filter").value;
  order = ITEMS.map((_, i) => i).filter((i) => {
    const it = ITEMS[i], r = reviews[it.obs_id];
    if (f === "todo") return pending(it) || it.obs_id === keepId && (it.needs_review || it.recheck || it.adjudicate);
    if (f === "recheck") return it.recheck;
    if (f === "adjudicate") return it.adjudicate;
    if (f === "queue") return it.needs_review || it.recheck || it.adjudicate;
    if (f === "changed") return r && r.label !== it.label_ai1 && r.label !== it.label_ai2;
    if (f === "all") return true;
    return it.product === f;
  });
  cur = Math.max(0, keepId ? order.findIndex((i) => ITEMS[i].obs_id === keepId) : 0);
}

// a flagged text is pending until reviewed; a text picked for a second look is pending until looked at again
function pending(it) {
  const r = reviews[it.obs_id];
  return (it.needs_review && !r) || (it.recheck && !(r && r.recheck)) || (it.adjudicate && !adjud[it.obs_id]);
}

// which question this text is shown for: an open review comes first, then the second look, then adjudication
function mode(it) {
  const r = reviews[it.obs_id];
  if ($("filter").value === "adjudicate" && it.adjudicate) return "adjudicate";
  if (it.needs_review && !r) return "review";
  if (it.recheck && !(r && r.recheck)) return "recheck";
  if (it.adjudicate && (!adjud[it.obs_id] || !it.needs_review)) return "adjudicate";
  return "review";
}

function render() {
  const queue = ITEMS.filter((it) => it.needs_review || it.recheck || it.adjudicate);
  const done = queue.filter((it) => !pending(it)).length;
  $("count").textContent = `${done} / ${queue.length} reviewed`;
  $("meter").style.width = `${(100 * done / Math.max(1, queue.length)).toFixed(1)}%`;
  if (!order.length) {
    $("card").hidden = true;
    showNotice($("filter").value === "todo" ? "Every flagged text has been reviewed. Tell Claude in the chat and the labels will be collected." : "No texts match this filter.");
    return;
  }
  $("card").hidden = false;
  if (db && canWrite) hideNotice();
  const it = ITEMS[order[cur]];
  current = it.obs_id;
  const r = reviews[it.obs_id];
  $("prod").textContent = it.product;
  $("where").textContent = `${it.record_type} · r/${it.subreddit}`;
  $("date").textContent = it.date;
  $("pos").textContent = `${cur + 1} of ${order.length}`;
  $("link").href = it.url;
  const md = mode(it);
  const r2 = md === "adjudicate" ? adjud[it.obs_id] : r;
  $("guide").hidden = md !== "recheck";
  $("text").innerHTML = highlight(it.text, it.product);
  $("text").scrollTop = 0;
  const agree = it.label_ai1 === it.label_ai2 && md !== "adjudicate";
  // adjudication is blind: the two labels are shown as A and B, in an order that varies from text to text
  const ab = it.ab_swap ? [[it.codex_label, it.codex_reason], [it.final_label, it.reason_ai1]]
                        : [[it.final_label, it.reason_ai1], [it.codex_label, it.codex_reason]];
  $("ai").innerHTML = md === "adjudicate"
    ? `Two annotators disagree. A: <b>${ab[0][0]}</b> (${esc(ab[0][1] || "")}) · B: <b>${ab[1][0]}</b> (${esc(ab[1][1] || "")}). Which is right? A third label is fine too.`
    : agree
    ? `Both AI annotators: <b>${it.label_ai1}</b>${it.reason_ai1 ? " · " + esc(it.reason_ai1) : ""}`
    : `AI annotators disagree: <b>${it.label_ai1}</b> (${esc(it.reason_ai1 || "")}) vs <b>${it.label_ai2}</b> (${esc(it.reason_ai2 || "")}). Pick one.`;
  $("accept").hidden = !agree;
  for (const l of LABELS) {
    const b = $("b-" + l);
    b.classList.toggle("on", !!r2 && r2.label === l);
    b.classList.toggle("proposed", md === "adjudicate" ? (l === it.final_label || l === it.codex_label)
                                                        : (l === it.label_ai1 || l === it.label_ai2));
    b.setAttribute("aria-pressed", String(!!r2 && r2.label === l));
  }
  $("note").value = r2?.note || "";
  $("prev").disabled = cur === 0;
  $("next").disabled = cur >= order.length - 1;
  $("status").textContent = r2 ? `Saved: ${r2.label}` : "";
}

function showNotice(msg) { $("notice").textContent = msg; $("notice").hidden = false; }
function hideNotice() { $("notice").hidden = true; }

async function choose(label) {
  const it = ITEMS[order[cur]];
  if (!it) return;
  const body = {label, note: $("note").value.trim(), ai1: it.label_ai1, ai2: it.label_ai2, product: it.product,
    at: Date.now()};
  const md = mode(it);
  let coll = "reviews";
  if (md === "adjudicate") {
    coll = "adjudication";
    Object.assign(body, {ours: it.final_label, third: it.codex_label});
    adjud[it.obs_id] = body;
  } else {
    if (md === "recheck" || (it.recheck && reviews[it.obs_id]?.recheck)) body.recheck = true;
    reviews[it.obs_id] = body;
  }
  $("status").textContent = "Saving…";
  if (db && canWrite) {
    try { await db.collection(coll).doc(it.obs_id).set(body); $("status").textContent = `Saved: ${label}`; }
    catch (e) {
      if (e && e.code === "invalid_argument") { canWrite = false; showNotice("You can view these labels but not change them."); }
      $("status").textContent = "Not saved: " + (e && e.message ? e.message : "the store refused the write");
      return;
    }
  } else {
    $("status").textContent = "Not saved: saving works only when this page is opened from claude.ai while signed in.";
  }
  advance(it.obs_id);
}

function advance(fromId) {
  const f = $("filter").value;
  if (f === "todo") { rebuildOrder(); render(); return; }
  if (cur < order.length - 1) cur += 1;
  render();
}

for (const l of LABELS) $("b-" + l).addEventListener("click", () => choose(l));
$("accept").addEventListener("click", () => {
  const it = ITEMS[order[cur]];
  if (it && it.label_ai1 === it.label_ai2) choose(it.label_ai1);
});
$("next").addEventListener("click", () => { if (cur < order.length - 1) { cur += 1; render(); } });
$("prev").addEventListener("click", () => { if (cur > 0) { cur -= 1; render(); } });
$("filter").addEventListener("change", () => { rebuildOrder(); render(); });
$("note").addEventListener("keydown", (e) => e.stopPropagation());
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key;
  if (["1", "2", "3", "4"].includes(k)) { choose(LABELS[+k - 1]); e.preventDefault(); }
  else if (k === "Enter") { $("accept").click(); e.preventDefault(); }
  else if (k === "ArrowRight") $("next").click();
  else if (k === "ArrowLeft") $("prev").click();
});

rebuildOrder();
render();

(async () => {
  const claude = window.claude;
  db = claude ? await claude.use("db") : null;
  if (!db) { showNotice("Saving is not available in this view. Open the page from claude.ai while signed in to record labels."); return; }
  const user = await claude.use("user");
  const may = user ? await user.can("data.write") : null;
  if (may === false) { canWrite = false; showNotice("You can view these labels but not change them."); }
  db.collection("reviews").limit(1000).onSnapshot((snap) => {
    const next = {};
    for (const d of snap.docs) {
      const v = d.data();
      if (v && v.label) next[d.id] = {label: v.label, note: v.note || "", recheck: !!v.recheck};
    }
    reviews = next;
    rebuildOrder(current);
    render();
  }, (e) => showNotice("Saved labels could not be loaded (" + (e.code || "error") + "). Reload the page."));
  db.collection("adjudication").limit(1000).onSnapshot((snap) => {
    const next = {};
    for (const d of snap.docs) { const v = d.data(); if (v && v.label) next[d.id] = {label: v.label, note: v.note || ""}; }
    adjud = next;
    rebuildOrder(current);
    render();
  }, () => {});
})();
</script>
"""


def main() -> None:
    s = pd.read_csv(ROOT / "labeling/label_sample.csv")
    cols = ["obs_id", "product", "record_type", "subreddit", "date", "url", "text", "label_ai1", "reason_ai1",
            "label_ai2", "reason_ai2", "needs_review", "recheck", "adjudicate", "final_label", "codex_label",
            "codex_reason", "ab_swap"]
    # second look at demand vs other: every reviewed text where demand was one of the choices (labels.csv)
    labels_csv = ROOT / "labeling/labels.csv"
    re_ids = set()
    if labels_csv.exists():
        lab = pd.read_csv(labels_csv)
        first = json.loads((ROOT / "labeling/reviews_first_pass.json").read_text()) \
            if (ROOT / "labeling/reviews_first_pass.json").exists() else {}
        rev = lab[lab["obs_id"].isin(first)]      # texts reviewed before the second look (rounds 1-2)
        re_ids = set(rev.loc[rev[["label_ai1", "label_ai2", "human_label"]].eq("demand").any(axis=1), "obs_id"])
    s["recheck"] = s["obs_id"].isin(re_ids)
    # adjudication: every gold text where both annotators from the other model family (Codex and ChatGPT) agree
    # with each other but not with our label; shown blind as option A / option B
    s["adjudicate"], s["final_label"], s["codex_label"], s["codex_reason"], s["ab_swap"] = False, "", "", "", False
    codex_path = ROOT / "labeling/annotations/annotator3_codex.jsonl"
    chatgpt_path = ROOT / "labeling/annotations/annotator4_chatgpt.jsonl"
    if labels_csv.exists() and codex_path.exists() and chatgpt_path.exists():
        rd = lambda f: pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()]).set_index("obs_id")
        cx, cg = rd(codex_path), rd(chatgpt_path)
        gold = lab[lab["tier"].fillna("gold") == "gold"].set_index("obs_id") if "tier" in lab else lab.set_index("obs_id")
        third, fourth = cx["label"].reindex(gold.index), cg["label"].reindex(gold.index)
        dis = gold.index[(third == fourth) & third.ne(gold["label"])]
        pick = pd.Series(dis)
        pd.DataFrame({"obs_id": pick, "final_label": gold.loc[pick, "label"].to_numpy(),
                      "label_source": gold.loc[pick, "label_source"].to_numpy(),
                      "codex_label": cx.loc[pick, "label"].to_numpy()}).to_csv(ROOT / "labeling/adjudication_sample.csv",
                                                                             index=False)
        m = s["obs_id"].isin(set(pick))
        s.loc[m, "adjudicate"] = True
        s.loc[m, "final_label"] = s.loc[m, "obs_id"].map(gold["label"])
        s.loc[m, "codex_label"] = s.loc[m, "obs_id"].map(cx["label"])
        s.loc[m, "codex_reason"] = s.loc[m, "obs_id"].map(cx["reason"])
        s.loc[m, "ab_swap"] = (s.index[m].to_series() % 2 == 1).to_numpy()
    s["text"] = s["text"].astype(str).str.slice(0, 2500)
    for c in ["reason_ai1", "reason_ai2"]:
        s[c] = s[c].fillna("")
    s["needs_review"] = s["needs_review"].astype(bool)
    items = s[cols].to_dict(orient="records")
    html = TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "labeling/review_page.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(items)} items -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

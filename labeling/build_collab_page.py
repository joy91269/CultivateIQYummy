"""Build the labelling page for a second human annotator (a collaborator), who labels blind.

The page shows 160 texts that the first reviewer has already decided, with no AI labels and no earlier decisions, so
that the two people's labels can be compared (human-human agreement):
    the 73 texts adjudicated between the two model families, the 40 silver labels spot-checked, and 47 texts drawn at
    random from the texts reviewed in rounds 1-3.
Labels are saved to the page's shared store, collection `collab`, one document per person and text
(`<person id>_<obs_id>`), and exported with labeling/collect_collab.py.

Usage: python labeling/build_collab_page.py   ->  labeling/collab_page.html, labeling/collab_sample.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r"""<title>Produce Labels: Second Annotator</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f4f6f5; --surface: #ffffff; --ink: #18211d; --muted: #5d6b64; --line: #d8dfdb; --accent: #2f7d4f;
  --price: #b4690e; --supply: #1f6fb2; --demand: #9a3d8f; --other: #6b7570;
  --price-bg: #fbf0e1; --supply-bg: #e4eff9; --demand-bg: #f5e6f3; --other-bg: #eceeed; --mark: #fff1a8;
  --sans: "Public Sans", "Segoe UI", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111614; --surface: #19201d; --ink: #e6ece9; --muted: #9aa8a1; --line: #2c3632; --accent: #5fb985;
  --price: #e9a24a; --supply: #6aaef0; --demand: #d886cc; --other: #a4aea9;
  --price-bg: #33261a; --supply-bg: #182a3a; --demand-bg: #31202f; --other-bg: #242b28; --mark: #5a4b12; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111614; --surface: #19201d; --ink: #e6ece9; --muted: #9aa8a1; --line: #2c3632; --accent: #5fb985;
  --price: #e9a24a; --supply: #6aaef0; --demand: #d886cc; --other: #a4aea9;
  --price-bg: #33261a; --supply-bg: #182a3a; --demand-bg: #31202f; --other-bg: #242b28; --mark: #5a4b12; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg) }
.wrap { max-width: 860px; margin: 0 auto; padding-block: 20px 48px; display: grid; gap: 16px }
header.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); padding-block: 12px 10px;
  border-bottom: 1px solid var(--line); display: grid; gap: 8px }
h1 { font-size: 1.25rem; margin: 0; text-wrap: balance }
.row { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: space-between }
.meter { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; flex: 1 1 200px }
.meter > i { display: block; height: 100%; background: var(--accent); width: 0 }
.count { font-family: var(--mono); font-size: .85rem; color: var(--muted); font-variant-numeric: tabular-nums }
.intro { color: var(--muted); font-size: .92rem; margin: 0; max-width: 65ch }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px; display: grid; gap: 14px; min-width: 0 }
.meta { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: .85rem; color: var(--muted); align-items: center }
.chip { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; padding: 2px 8px; border-radius: 999px;
  background: var(--other-bg); color: var(--ink) }
.text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 46vh; overflow-y: auto; font-size: 1rem; border-left: 3px solid var(--line); padding-left: 12px }
mark { background: var(--mark); color: inherit; border-radius: 2px; padding: 0 1px }
.question { font-weight: 600; margin: 0 }
.labels { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px }
.lab { font: inherit; text-align: left; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; cursor: pointer;
  background: var(--surface); color: var(--ink); display: grid; gap: 2px; position: relative }
.lab span { font-weight: 600 } .lab small { color: var(--muted); font-size: .8rem }
.lab kbd { position: absolute; top: 8px; right: 10px; font-family: var(--mono); font-size: .75rem; color: var(--muted) }
.lab[data-l=price].on { background: var(--price-bg); border-color: var(--price) }
.lab[data-l=supply].on { background: var(--supply-bg); border-color: var(--supply) }
.lab[data-l=demand].on { background: var(--demand-bg); border-color: var(--demand) }
.lab[data-l=other].on { background: var(--other-bg); border-color: var(--other) }
.lab:focus-visible, .btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }
.nav { display: flex; flex-wrap: wrap; gap: 8px; align-items: center }
.btn { font: inherit; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 8px; padding: 6px 12px; cursor: pointer }
.btn:disabled { opacity: .45; cursor: default }
input[type=text] { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; flex: 1 1 240px; min-width: 0 }
.status { font-size: .85rem; color: var(--muted) }
.notice { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 8px; padding: 12px 14px }
details.codebook { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px }
details.codebook summary { cursor: pointer; font-weight: 600 }
.tablewrap { overflow-x: auto }
table { border-collapse: collapse; width: 100%; font-size: .9rem; margin-top: 10px }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line) }
ol.rules { padding-left: 1.2em; margin: 10px 0 0; display: grid; gap: 4px; font-size: .9rem }
</style>

<div class="wrap">
  <header class="bar">
    <div class="row">
      <h1>Produce market labels</h1>
      <span class="count" id="count">0 / 0 labelled</span>
    </div>
    <div class="meter" aria-hidden="true"><i id="meter"></i></div>
  </header>
  <p class="intro">Each text is a Reddit post or comment that mentions a product (highlighted). Decide what it says
  about the <b>market for that product</b>. Please judge on your own; there are no right answers shown. Read the
  codebook below once before you start. Keys: 1–4 to choose, ← → to move.</p>
  <div class="notice" id="notice" hidden></div>
  <section class="card" id="card" aria-live="polite">
    <div class="meta"><span class="chip" id="prod"></span><span id="where"></span><span class="count" id="pos"></span>
      <a id="link" href="#" target="_blank" rel="noopener">Open on Reddit</a></div>
    <div class="text" id="text"></div>
    <p class="question" id="q"></p>
    <div class="labels" role="group" aria-label="Label">
      <button class="lab" data-l="price" id="b-price"><span>Price</span><small>its price, a price change, affordability</small><kbd>1</kbd></button>
      <button class="lab" data-l="supply" id="b-supply"><span>Supply</span><small>shortage, quality, crop, imports, recall</small><kbd>2</kbd></button>
      <button class="lab" data-l="demand" id="b-demand"><span>Demand</span><small>buying or eating more or less of it</small><kbd>3</kbd></button>
      <button class="lab" data-l="other" id="b-other"><span>Other</span><small>no market signal</small><kbd>4</kbd></button>
    </div>
    <div class="nav">
      <button class="btn" id="prev">← Previous</button><button class="btn" id="next">Next →</button>
      <input type="text" id="note" placeholder="Optional note (hard case, why)" aria-label="Note">
      <span class="status" id="status"></span>
    </div>
  </section>
  <details class="codebook">
    <summary>Codebook</summary>
    <div class="tablewrap"><table>
      <tr><th>Label</th><th>Use it when the text…</th><th>Example</th></tr>
      <tr><td>price</td><td>talks about the product's price, a price change, affordability, a discount with a price, the extra charge for guacamole</td><td>"Avocados were $2.49 each at Kroger, insane."</td></tr>
      <tr><td>supply</td><td>talks about availability, shortages, empty shelves, poor quality from supply problems, crop or weather conditions, imports, recalls, a restaurant running out</td><td>"No romaine anywhere, store says shortage."</td></tr>
      <tr><td>demand</td><td>is mainly about buying or eating more or less of the product: a recipe where it is the main ingredient, what to do with a lot of it, stocking up, trends, a promotion without a price (free guacamole), avoiding or throwing it out after a scare</td><td>"Not touching onions until the E. coli thing is over."</td></tr>
      <tr><td>other</td><td>has no clear market signal: one ingredient among several, cooking or storage tips, home gardening, jokes, restaurant meal complaints, product-flavoured snacks, avocado oil</td><td>"How do you keep cut avocado from browning?"</td></tr>
    </table></div>
    <ol class="rules">
      <li>If the product is only one of several ingredients or mentioned in passing: other, even in a recipe.</li>
      <li>Two signals in one text: price beats supply beats demand.</li>
      <li>Recalls, outbreaks, product pulled: supply. People avoiding it: demand.</li>
      <li>Home-garden problems: other, unless they describe crop conditions in general or a bad season.</li>
      <li>A whole restaurant meal's price: other. The price of the product or a product add-on: price.</li>
      <li>When in doubt between a signal and other: other.</li>
    </ol>
  </details>
</div>

<script>
const ITEMS = __ITEMS__;
const LABELS = ["price", "supply", "demand", "other"];
const PRODUCT_RE = {avocado: /\b(avocados?|guac(amole)?|hass)\b/gi, lettuce: /\b(lettuces?|romaine)\b/gi,
  onion: /\bonions?\b/gi, cucumber: /\b(cucumbers?|cukes?)\b/gi};
let done = {};         // obs_id -> {label, note} for the person viewing
let cur = 0, db = null, me = null, canWrite = true;
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
const notice = (m) => { $("notice").textContent = m; $("notice").hidden = !m; };

function render() {
  const n = Object.keys(done).filter((k) => ITEMS.some((it) => it.obs_id === k)).length;
  $("count").textContent = `${n} / ${ITEMS.length} labelled`;
  $("meter").style.width = `${(100 * n / ITEMS.length).toFixed(1)}%`;
  const it = ITEMS[cur], r = done[it.obs_id];
  $("prod").textContent = it.product;
  $("where").textContent = `${it.record_type} · r/${it.subreddit} · ${it.date}`;
  $("pos").textContent = `${cur + 1} of ${ITEMS.length}`;
  $("link").href = it.url;
  $("text").innerHTML = esc(it.text).replace(PRODUCT_RE[it.product], (m) => `<mark>${m}</mark>`);
  $("text").scrollTop = 0;
  $("q").textContent = `What does this say about the market for ${it.product}s?`.replace("lettuces", "lettuce");
  for (const l of LABELS) { const b = $("b-" + l); b.classList.toggle("on", !!r && r.label === l); b.setAttribute("aria-pressed", String(!!r && r.label === l)); }
  $("note").value = r?.note || "";
  $("prev").disabled = cur === 0; $("next").disabled = cur >= ITEMS.length - 1;
  $("status").textContent = r ? `Saved: ${r.label}` : "";
}

async function choose(label) {
  const it = ITEMS[cur];
  const body = {obs_id: it.obs_id, label, note: $("note").value.trim(), by: me, at: Date.now()};
  done[it.obs_id] = body;
  render();
  if (!db || !me || !canWrite) { $("status").textContent = "Not saved: open this page from claude.ai while signed in."; return; }
  $("status").textContent = "Saving…";
  try { await db.collection("collab").doc(`${me}_${it.obs_id}`).set(body); $("status").textContent = `Saved: ${label}`; }
  catch (e) { $("status").textContent = "Not saved: " + (e && e.message ? e.message : "the store refused the write"); return; }
  const nextOpen = ITEMS.findIndex((x, i) => i > cur && !done[x.obs_id]);
  cur = nextOpen >= 0 ? nextOpen : Math.min(cur + 1, ITEMS.length - 1);
  render();
  if (Object.keys(done).length >= ITEMS.length) notice("All texts are labelled. Thank you! Tell Gan you are done.");
}

for (const l of LABELS) $("b-" + l).addEventListener("click", () => choose(l));
$("next").addEventListener("click", () => { if (cur < ITEMS.length - 1) { cur += 1; render(); } });
$("prev").addEventListener("click", () => { if (cur > 0) { cur -= 1; render(); } });
$("note").addEventListener("keydown", (e) => e.stopPropagation());
document.addEventListener("keydown", (e) => {
  if (["1", "2", "3", "4"].includes(e.key)) { choose(LABELS[+e.key - 1]); e.preventDefault(); }
  else if (e.key === "ArrowRight") $("next").click();
  else if (e.key === "ArrowLeft") $("prev").click();
});
render();

(async () => {
  const claude = window.claude;
  db = claude ? await claude.use("db") : null;
  const user = claude ? await claude.use("user") : null;
  me = user ? await user.id() : null;
  if (!db || !me) { notice("Saving is not available in this view. Open the page from claude.ai while signed in."); return; }
  const may = await user.can("data.write");
  if (may === false) { canWrite = false; notice("You can view this page but not save labels. Ask Gan to give you edit access."); }
  db.collection("collab").where("by", "==", me).onSnapshot((snap) => {
    const next = {};
    for (const d of snap.docs) { const v = d.data(); if (v && v.label) next[v.obs_id] = {label: v.label, note: v.note || ""}; }
    done = next;
    const first = ITEMS.findIndex((x) => !done[x.obs_id]);
    if (first >= 0 && done[ITEMS[cur].obs_id]) cur = first;
    render();
  }, (e) => notice("Saved labels could not be loaded (" + (e.code || "error") + "). Reload the page."));
})();
</script>
"""


def main(seed: int = 2027) -> None:
    lab = pd.read_csv(ROOT / "labeling/labels.csv")
    smp = pd.read_csv(ROOT / "labeling/label_sample.csv").set_index("obs_id")
    adj = json.loads((ROOT / "labeling/adjudication.json").read_text())
    rev = lab[lab["reviewed"]]
    silver_checked = rev.loc[rev["tier"] == "silver", "obs_id"].tolist()
    earlier = rev.loc[(rev["tier"] == "gold") & ~rev["obs_id"].isin(adj), "obs_id"]
    pick = list(adj) + silver_checked + earlier.sample(min(47, len(earlier)), random_state=seed).tolist()
    pick = pd.Series(pick).drop_duplicates().sample(frac=1, random_state=seed).tolist()     # shuffled order
    part = {o: "adjudicated" for o in adj} | {o: "silver check" for o in silver_checked}
    sample = pd.DataFrame({"obs_id": pick, "set": [part.get(o, "earlier review") for o in pick]})
    sample.to_csv(ROOT / "labeling/collab_sample.csv", index=False)
    items = [{"obs_id": o, "product": smp.loc[o, "product"], "record_type": smp.loc[o, "record_type"],
              "subreddit": smp.loc[o, "subreddit"], "date": str(smp.loc[o, "date"]), "url": smp.loc[o, "url"],
              "text": str(smp.loc[o, "text"])[:2500]} for o in pick]
    html = TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "labeling/collab_page.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(items)} texts ({sample['set'].value_counts().to_dict()}) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

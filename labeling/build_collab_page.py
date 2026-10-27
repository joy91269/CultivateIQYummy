"""Build the labelling page for a second human annotator (a collaborator), who labels blind.

The page shows 80 texts that the first reviewer has already decided, with no AI labels and no earlier decisions, so
that the two people's labels can be compared (human-human agreement): 40 of the texts adjudicated between the two
model families, 20 of the spot-checked silver labels and 20 texts reviewed in rounds 1-3, all drawn at random.
The interface and codebook are in Chinese and English, and each text has a literal Chinese translation made by an AI
model (Claude Opus, instructed to translate faithfully without interpreting; labeling/collab_translations.jsonl),
because the collaborator reads English less easily. The English original is always shown first.
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
.zh { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .98rem; background: var(--bg); border-radius: 8px;
  padding: 10px 12px; max-height: 40vh; overflow-y: auto }
.zh:empty { display: none }
.zhlab { display: block; font-size: .78rem; color: var(--muted); margin-bottom: 4px }
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
      <h1>农产品市场标注 · Produce market labels</h1>
      <span class="count" id="count">0 / 0</span>
    </div>
    <div class="meter" aria-hidden="true"><i id="meter"></i></div>
  </header>
  <p class="intro">每条是一个提到某种农产品（黄色高亮）的 Reddit 帖子或评论。请判断它对<b>这个产品的市场</b>说了什么。
  页面不显示任何参考答案，请独立判断。开始前先看一遍下方的标注规则。英文原文在上，中文是 AI 逐字翻译，仅帮助理解；有出入时以英文原文为准。
  快捷键：1–4 选择，← → 翻页。</p>
  <div class="notice" id="notice" hidden></div>
  <section class="card" id="card" aria-live="polite">
    <div class="meta"><span class="chip" id="prod"></span><span id="where"></span><span class="count" id="pos"></span>
      <a id="link" href="#" target="_blank" rel="noopener">Open on Reddit</a></div>
    <div class="text" id="text"></div>
    <div class="zh" id="zh"></div>
    <p class="question" id="q"></p>
    <div class="labels" role="group" aria-label="Label">
      <button class="lab" data-l="price" id="b-price"><span>价格 Price</span><small>产品的价格、涨价降价、买不买得起</small><kbd>1</kbd></button>
      <button class="lab" data-l="supply" id="b-supply"><span>供给 Supply</span><small>缺货、质量差、收成天气、进口、召回</small><kbd>2</kbd></button>
      <button class="lab" data-l="demand" id="b-demand"><span>需求 Demand</span><small>人们多买多吃或少买少吃</small><kbd>3</kbd></button>
      <button class="lab" data-l="other" id="b-other"><span>其他 Other</span><small>没有市场信号</small><kbd>4</kbd></button>
    </div>
    <div class="nav">
      <button class="btn" id="prev">← 上一条</button><button class="btn" id="next">下一条 →</button>
      <input type="text" id="note" placeholder="备注（可选，难判断时写原因）" aria-label="备注">
      <span class="status" id="status"></span>
    </div>
  </section>
  <details class="codebook" open>
    <summary>标注规则 · Codebook</summary>
    <div class="tablewrap"><table>
      <tr><th>标签</th><th>什么时候用</th><th>例子</th></tr>
      <tr><td>价格 price</td><td>说到这个产品的价格、涨价降价、贵不贵、带具体价格的打折，或者 Chipotle 鳄梨酱要另外加钱</td><td>"牛油果在 Kroger 卖 2.49 美元一个，疯了。"</td></tr>
      <tr><td>供给 supply</td><td>买不到、缺货、货架空、因为供应问题质量差、收成或天气、进口、召回或下架、餐厅卖完了</td><td>"到处都没有罗马生菜，店里说缺货。"</td></tr>
      <tr><td>需求 demand</td><td>主要在说人们多买多吃或少买少吃这个产品：以它为主料的菜谱、东西太多了怎么用掉、囤货、跟风、不带价格的促销（免费鳄梨酱券）、出了食品安全问题后不敢吃或扔掉</td><td>"大肠杆菌那事没过去之前我不碰洋葱。"</td></tr>
      <tr><td>其他 other</td><td>没有明确市场信号：只是好几种配料之一、做法或储存技巧、自家种菜、玩笑、抱怨餐厅饭菜、只是某种口味的零食、牛油果油</td><td>"切开的牛油果怎么才不变黑？"</td></tr>
    </table></div>
    <ol class="rules">
      <li>产品只是好几种配料之一，或者只是顺带提到：其他，即使是菜谱也一样。</li>
      <li>一段话里有两种信号：价格 优先于 供给 优先于 需求。</li>
      <li>召回、疫情爆发、产品下架：供给。人们因此不吃、不买：需求。</li>
      <li>自家菜园的问题：其他；除非讲的是整体的收成或年景。</li>
      <li>餐厅整份饭的价格：其他。产品本身或单独加这个产品的价格：价格。</li>
      <li>在"某种信号"和"其他"之间拿不准：选其他。</li>
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
  $("count").textContent = `已完成 ${n} / ${ITEMS.length}`;
  $("meter").style.width = `${(100 * n / ITEMS.length).toFixed(1)}%`;
  const it = ITEMS[cur], r = done[it.obs_id];
  $("prod").textContent = it.product;
  $("where").textContent = `${it.record_type} · r/${it.subreddit} · ${it.date}`;
  $("pos").textContent = `${cur + 1} of ${ITEMS.length}`;
  $("link").href = it.url;
  $("text").innerHTML = esc(it.text).replace(PRODUCT_RE[it.product], (m) => `<mark>${m}</mark>`);
  $("text").scrollTop = 0;
  const zhName = {avocado: "牛油果", lettuce: "生菜", onion: "洋葱", cucumber: "黄瓜"}[it.product];
  $("q").textContent = `这条对「${zhName}」的市场说了什么？`;
  $("zh").innerHTML = it.zh ? `<span class="zhlab">中文翻译（AI 逐字翻译，仅供理解）</span>${esc(it.zh)}` : "";
  for (const l of LABELS) { const b = $("b-" + l); b.classList.toggle("on", !!r && r.label === l); b.setAttribute("aria-pressed", String(!!r && r.label === l)); }
  $("note").value = r?.note || "";
  $("prev").disabled = cur === 0; $("next").disabled = cur >= ITEMS.length - 1;
  $("status").textContent = r ? `已保存：${r.label}` : "";
}

async function choose(label) {
  const it = ITEMS[cur];
  const body = {obs_id: it.obs_id, label, note: $("note").value.trim(), by: me, at: Date.now()};
  done[it.obs_id] = body;
  render();
  if (!db || !me || !canWrite) { $("status").textContent = "未保存：请登录 claude.ai 后从那里打开本页面。"; return; }
  $("status").textContent = "保存中…";
  try { await db.collection("collab").doc(`${me}_${it.obs_id}`).set(body); $("status").textContent = `已保存：${label}`; }
  catch (e) { $("status").textContent = "Not saved: " + (e && e.message ? e.message : "the store refused the write"); return; }
  const nextOpen = ITEMS.findIndex((x, i) => i > cur && !done[x.obs_id]);
  cur = nextOpen >= 0 ? nextOpen : Math.min(cur + 1, ITEMS.length - 1);
  render();
  if (Object.keys(done).length >= ITEMS.length) notice("全部完成，谢谢！请告诉 Gan 已经标完。");
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
  if (!db || !me) { notice("当前视图无法保存。请登录 claude.ai 后从那里打开本页面。"); return; }
  const may = await user.can("data.write");
  if (may === false) { canWrite = false; notice("你可以查看但不能保存。请让 Gan 在分享菜单里给你编辑权限。"); }
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
    silver = rev.loc[rev["tier"] == "silver", "obs_id"]
    earlier = rev.loc[(rev["tier"] == "gold") & ~rev["obs_id"].isin(adj), "obs_id"]
    pick = (pd.Series(list(adj)).sample(40, random_state=seed).tolist() + silver.sample(20, random_state=seed).tolist()
            + earlier.sample(20, random_state=seed).tolist())
    pick = pd.Series(pick).sample(frac=1, random_state=seed).tolist()            # shuffled order
    part = {o: "adjudicated" for o in adj} | {o: "silver check" for o in silver}
    sample = pd.DataFrame({"obs_id": pick, "set": [part.get(o, "earlier review") for o in pick]})
    sample.to_csv(ROOT / "labeling/collab_sample.csv", index=False)
    tr_path = ROOT / "labeling/collab_translations.jsonl"
    zh = {}
    if tr_path.exists():
        zh = {r["obs_id"]: r["zh"] for r in map(json.loads, tr_path.read_text().splitlines()) if r}
    items = [{"obs_id": o, "product": smp.loc[o, "product"], "record_type": smp.loc[o, "record_type"],
              "subreddit": smp.loc[o, "subreddit"], "date": str(smp.loc[o, "date"]), "url": smp.loc[o, "url"],
              "text": str(smp.loc[o, "text"])[:2500], "zh": zh.get(o, "")} for o in pick]
    html = TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "labeling/collab_page.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(items)} texts ({sample['set'].value_counts().to_dict()}), {sum(bool(i['zh']) for i in items)} "
          f"translated -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

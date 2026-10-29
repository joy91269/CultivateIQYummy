# CultivateIQ Yummy — Social Media Price Signals for Fresh Produce

Submission for the **Cultivate IQ – NAFSI Track 3 (Supply Chain)** student data challenge (fall 2026).

We turn Reddit discussion about four fresh produce products into weekly **price, supply and demand signals** and
compare them with **USDA AMS terminal market wholesale prices**, 2022–2024, and check them against **Google search
interest** in the US.

| | |
|---|---|
| Products | avocados, lettuce (iceberg and romaine), onions, cucumbers |
| Period | 1 January 2022 – 31 December 2024 |
| Online data | Reddit posts and comments from 29 topical subreddits (consumers, grocery retailers, food service, growers, economy) and 9 city subreddits, plus r/news headlines, via the public Arctic Shift archive API: 1.5 M posts scanned, 92,424 texts kept |
| Market benchmark | USDA AMS Specialty Crops terminal market reports, 10 cities, and AMS daily movement (shipment) reports (MyMarketNews MARS API) |
| Search benchmark | Google Trends weekly US search interest for each product alone and with "price", "shortage" and "recall" (16 terms) |
| Labels | 2,946 texts labelled price / supply / demand / other, one per Reddit thread (`labeling/`): 1,458 **gold** labels (two Claude annotators; every disagreement and a check of agreements decided by a person; a blind check by OpenAI Codex and ChatGPT with blind adjudication; 340 of them moved up from silver after a person or ChatGPT confirmed them; used for training and all accuracy figures) and 1,488 **silver** labels (the two Claude annotators agree; not used by the final model) |
| Classifier | all-mpnet-base-v2 sentence embeddings + logistic regression trained on the gold labels; macro-F1 0.72 (95% CI 0.69–0.75), accuracy 0.82 in thread-grouped 5-fold CV (keyword rules, zero-shot BART, fine-tuned DistilRoBERTa and three sentence encoders with and without silver labels compared) |
| Validation events | 8 documented events with dates and news sources (`config/events_reference.csv`) |

## Results at a glance

- **Two documented events were supply shocks, and the market data confirm it.** Prices rose while shipments fell:
  the February 2022 halt of Mexican avocado imports (+15% after seasonal adjustment, shipments −33%) and the 2022
  INSV lettuce shortage (+94%, shipments −13%; a carton of 24 iceberg heads went from a median $30 to $68).
- **Supply talk is the signal that finds events.** Weeks with a supply-talk spike (z ≥ 3 against the previous eight
  weeks) fall within two weeks of a documented event 36% of the time (8 of 22) against 11% by chance (p = 0.002);
  spikes in price talk, demand talk, volume or sentiment are not clearly above chance
  (`outputs/tables/spike_event_enrichment.csv`).
- **Reddit flags supply events; prices show which ones mattered.** Supply talk rose around six of the eight events:
  both supply shocks (24-fold during the avocado import halt), the avocado inspection pause and three of the four
  food-safety events, where recalls count as supply talk. Only the two supply shocks moved prices; the food-safety
  outbreaks, the recall and a viral recipe trend did not move national wholesale prices (−7% to +3% after seasonal
  adjustment).
- **Timing:** supply talk about the avocado halt surged the day after its Saturday announcement, and prices rose the
  following week. Lettuce prices began to rise in mid-August 2022; the rise in lettuce talk later that month was
  about the Wendy's E. coli outbreak, and most talk about the shortage itself came in mid-November, the week prices
  peaked. Across all weeks, talk and prices correlate weakly at any lag (|r| ≤ 0.29; 2 of 20 product–series pairs
  with rotation-test p < 0.05).
- **Google searches show the same events.** Spikes in US searches for a product's shortage or recall fall near a
  documented event 31% of the time (11 of 36) against 11% by chance (p = 0.0015), and cover seven of the eight
  events. In six events, Reddit supply talk and these searches peaked in the same week. Both also spiked in the week
  of 23 October 2023 for onions, when Gills Onions recalled diced onions over a Salmonella outbreak, an event not in
  our list (`outputs/tables/trends_spike_event_enrichment.csv`, `event_search_peaks.csv`,
  `reddit_google_spike_agreement.csv`).
- **Comments filter false alarms; upvotes do not help.** Counting posts with at least 10 comments twice removes six
  supply spikes that lie near no event (mostly a few garden or restaurant complaints that drew no replies) and keeps
  every event found with equal weights: 8 of 18 spikes lie near an event (44%, p = 0.0004), and more of them coincide
  with a Google spike. Chosen on 2022–2023, it finds the same 2024 events as equal weights. Weighting by upvotes
  finds fewer events. The main signal keeps equal weights (`outputs/tables/weighting_robustness.csv`).
- **Online signals do not forecast national wholesale prices.** Over 2019–2026, adding Reddit, Google Trends, news
  coverage, FDA recalls or Wikipedia pageviews to price history improves neither forecasts 1 and 4 weeks ahead nor
  the warning of big price moves (`forecast_check/`, `outputs/tables/forecast_long_*.csv`).

Details: report (`report/`), notebook sections 5, 7–9, 11 and 13, tables in `outputs/tables/` (`event_supply_vs_demand.csv`,
`event_price_impact.csv`, `event_price_examples_usd.csv`, `lead_lag_crosscorrelation.csv`, `event_timing.csv`).

## Repository layout

```
config/                     products, keyword dictionary, subreddits, AMS reports, documented events
data_collection/            Task 1: dataset generation code (collect, clean, tag, summarise)
data/raw/reddit/            raw Reddit snapshot (gzipped JSON Lines) and weekly post counts
data/raw/google_trends/     Google Trends downloads, one file per search term
data/processed/             event dataset, keyword dictionary, summary table, AMS prices, model outputs
labeling/                   codebook, sample and review scripts, AI annotations, hand-reviewed labels
analysis/                   Task 2: classifiers, price index, signals, timing (imported by the notebook)
notebooks/                  task2_signals_and_prices.ipynb (Task 2 analysis) and its builder script
outputs/figures, tables/    figures and tables used in the report
forecast_check/             forecast check, 2019–2026: do online signals help forecast prices? (its own README)
docs/data_dictionary.md     data dictionary and provenance (source, access date, license, coverage, ...)
report/                     PDF report
```

## Deliverables

| Challenge deliverable | File |
|---|---|
| Event dataset (CSV) | `data/processed/event_dataset.csv` |
| Product and event keyword list | `data/processed/keyword_dictionary.csv` (also `.json`); source lists in `config/products.yaml`, `config/event_keywords.yaml` |
| Summary table | `data/processed/summary_table.csv` (also `.md`) |
| Dataset generation code | `data_collection/` |
| Task 2 notebook | `notebooks/task2_signals_and_prices.ipynb` |
| Environment | `requirements.txt` |
| Data dictionary and provenance | `docs/data_dictionary.md` |
| Report | `report/` |

## Setup

Python 3.12 or newer (developed with Python 3.13 on Linux, CPU only).

```bash
git clone https://github.com/joy91269/CultivateIQYummy.git
cd CultivateIQYummy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only re-collecting the AMS prices needs a key: create a free MyMarketNews account at
<https://mymarketnews.ams.usda.gov>, click your name, then **Show API key**, and put it in a file `.env` in the
repository root:

```
MARS_API_KEY=your-key
```

## Reproduce the results

**Quick path (about 10 minutes):** all collected data and model outputs are committed, so the analysis can be rerun
directly:

```bash
bash run_all.sh --analysis
```

This rebuilds and executes `notebooks/task2_signals_and_prices.ipynb`, which writes every figure to
`outputs/figures/` and every table to `outputs/tables/`, and reruns the forecast check
(`forecast_check/predict_long.py`, about 8 minutes).
`python scripts/check_reproduction.py` then compares every table in `outputs/tables/` with the committed version
and reports which ones reproduce.

**Full path (about 20 hours on a 2-core CPU):** re-collects all data and retrains the models.

```bash
bash run_all.sh
```

or step by step:

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Keyword dictionary | `python data_collection/build_keywords.py` | seconds | `data/processed/keyword_dictionary.*` |
| 2. Reddit posts | `python data_collection/collect_reddit.py posts` | ~3 h | `data/raw/reddit/posts/`, `post_counts/` |
| 3. r/news headlines | `python data_collection/collect_reddit.py titles` | ~3 h (rate limited) | `data/raw/reddit/posts/news.jsonl` |
| 3b. City subreddits | `python data_collection/collect_reddit.py geo` | ~1.5 h | `data/raw/reddit/posts/<city>.jsonl` |
| 4. Comment trees | `python data_collection/collect_reddit.py trees` | ~2 h | `data/raw/reddit/comments/` |
| 5. AMS prices | `python data_collection/collect_ams.py` | ~10 min | `data/processed/ams_terminal_prices.csv.gz` |
| 5b. AMS shipments | `python data_collection/collect_ams_movement.py` | ~1 h | `data/processed/ams_shipments_weekly.csv` |
| 5c. Google Trends | `python data_collection/collect_google_trends.py` | ~3 min | `data/processed/google_trends_weekly.csv` |
| 6. Event dataset | `python data_collection/build_event_dataset.py` | ~1 min | `event_dataset.csv`, `summary_table.*`, `filter_log.csv` |
| 7. Models | `python -m analysis.run_models all` | ~10 h | `outputs/tables/model_cv_*.csv`, `data/processed/classified_observations.csv.gz` |
| 8. Notebook and forecast check | `bash run_all.sh --analysis` | ~10 min | `outputs/` |

Notes
- Collection steps are resumable: rerunning a command continues where it stopped.
- Arctic Shift is a live archive; content deleted on Reddit can disappear from it, so a fresh collection can differ
  slightly from the committed snapshot in `data/raw/reddit/`. Step 6 rebuilds the dataset from that snapshot when
  the raw files are not re-downloaded.
- The labelled sample (`labeling/labels.csv`) is part of the repository; `labeling/make_label_sample.py` shows how it
  was drawn, `labeling/codebook.md` how it was labelled (including the AI annotation and human review procedure),
  and `labeling/annotations/` holds the labels and reasons of all four AI annotators (two Claude models; OpenAI Codex
  and ChatGPT, whose task is in `labeling/codex_task/`). `labeling/adjudication.json` holds the blind A/B decisions,
  `labeling/select_upgrade.py` and `labeling/crosscheck2/` the silver texts moved up to gold and ChatGPT's blind
  labels for them, and `labeling/build_collab_page.py` / `collect_collab.py` the second human annotator's sample and
  agreement.
- Model steps cache their intermediate results in `.cache/` (not committed) and resume after an interruption.
- If model downloads from Hugging Face fail behind a restrictive proxy, set `HF_HUB_DISABLE_XET=1`.
- `python report/build_report.py report/report.md` typesets the report (Markdown export of the report document)
  as a PDF with three-line tables and numbered figures; it needs pandoc and XeLaTeX.

### Reproduce with Kiro on AWS

We reproduced the quick path independently with the Kiro CLI (2.35.1), signed in through the challenge's AWS IAM
Identity Center account, on an Apple-silicon Mac (macOS, CPU only).

1. Install the Kiro CLI:

   ```bash
   curl -fsSL https://cli.kiro.dev/install | bash
   ```

2. Sign in with the start URL from the AWS invitation (region `us-east-1`); the CLI prints a code to enter in the
   browser:

   ```bash
   kiro-cli login --license pro --identity-provider <start URL> --region us-east-1 --use-device-flow
   ```

3. In an empty folder, give Kiro the task and let it run unattended:

   ```bash
   kiro-cli chat --no-interactive --trust-all-tools "Clone https://github.com/joy91269/CultivateIQYummy and reproduce its results by following the README exactly:
   1. Create a Python virtual environment and install requirements.txt (Python 3.12 or newer).
   2. Run \`bash run_all.sh --analysis\`.
   3. Run \`python scripts/check_reproduction.py\`.
   Do not change any code or data. When done, report the commands you ran, how long each took, and the full output of the last command."
   ```

Kiro used Python 3.13.18 and finished in about 2 minutes:

| Step | Time |
|---|---|
| `pip install -r requirements.txt` (pip cache already filled by an earlier run) | 34 s |
| `bash run_all.sh --analysis` | 47 s |
| `python scripts/check_reproduction.py` | under 1 s; 20 of 20 tables reproduce |

An earlier Kiro run found two problems, both fixed in commit `3464586`: the pinned numpy and scipy need Python 3.12
rather than 3.11, and one example text in `unusual_weeks.csv` depended on whether pyarrow was installed.

## Data sources and licences

See `docs/data_dictionary.md`. Reddit content is public and was accessed through the Arctic Shift research archive;
user names are replaced by pseudonymous ids. USDA AMS Market News data are US Government works in the public domain.
Search interest data: source Google Trends (https://trends.google.com).
Forecast check only: news coverage from the GDELT Project (https://www.gdeltproject.org), food recalls from openFDA
(public domain) and Wikipedia pageviews from the Wikimedia Foundation (CC0).

# CultivateIQ Yummy — Social Media Price Signals for Fresh Produce

Submission for the **Cultivate IQ – NAFSI Track 3 (Supply Chain)** student data challenge (fall 2026).

We turn Reddit discussion about four fresh produce products into weekly **price, supply and demand signals** and
compare them with **USDA AMS terminal market wholesale prices and shipments**, January 2019 – September 2026, around
33 documented market events, and check them against **Google search interest** in the US.

| | |
|---|---|
| Products | avocados, lettuce (iceberg and romaine), onions, cucumbers |
| Period | 1 January 2019 – 30 September 2026, collected in three periods with the same pipeline (2019–2021, 2022–2024, 2025–2026) |
| Online data | Reddit posts and comments from 29 topical subreddits (consumers, grocery retailers, food service, growers, economy) and 9 city subreddits, plus r/news headlines, via the public Arctic Shift archive API: 225,926 texts kept, 92,424 of them in 2022–2024 |
| Market benchmark | USDA AMS Specialty Crops terminal market reports, 10 cities, and AMS daily movement (shipment) reports (MyMarketNews MARS API) |
| Search benchmark | Google Trends weekly US search interest for each product alone and with "price", "shortage" and "recall" (16 terms) |
| Labels | 2,946 texts from 2022–2024 labelled price / supply / demand / other, one per Reddit thread (`labeling/`): 1,458 **gold** labels (two Claude annotators; every disagreement and a check of agreements decided by a person; a blind check by OpenAI Codex and ChatGPT with blind adjudication; 340 of them moved up from silver after a person or ChatGPT confirmed them; used for training and all accuracy figures) and 1,488 **silver** labels (the two Claude annotators agree; not used by the final model) |
| Classifier | all-mpnet-base-v2 sentence embeddings + logistic regression trained on the gold labels and applied to every year; macro-F1 0.72 (95% CI 0.69–0.75), accuracy 0.82 in thread-grouped 5-fold CV (keyword rules, zero-shot BART, fine-tuned DistilRoBERTa and three sentence encoders with and without silver labels compared) |
| Validation events | 33 documented events with dates and sources (`config/events_reference.csv`): every CDC multistate outbreak linked to one of the products, supply disruptions reported in trade press (import halts, border actions, tariffs, crop disease and weather), and national demand shocks (COVID-19 closures, a viral recipe) |

## Results at a glance

All results cover January 2019 – September 2026 and the 33 documented events.

- **Supply talk is the signal that finds events.** Weeks with a supply-talk spike (z ≥ 3 against the previous eight
  weeks) fall within two weeks of a documented event 42% of the time (23 of 55) against 21% by chance (p = 0.0003).
  Spikes in price talk, demand talk, volume or sentiment are not clearly above chance
  (`outputs/tables/spike_event_enrichment.csv`).
- **Google searches show the same events.** Spikes in US searches for a product's shortage or recall fall near a
  documented event 42% of the time (35 of 84, p < 0.0001). Reddit supply talk spiked around 20 of the 33 events and
  these searches around 21; in 13 events both peaked in the same week (`trends_spike_event_enrichment.csv`,
  `event_search_peaks.csv`).
- **Every unusual price rise came with a supply disruption.** Prices rose more than in 90% of ordinary windows in
  eight events. In six of them shipments fell as well, the mark of a supply shock: the March 2019 threat to close the
  US–Mexico border (a carton of 48 Hass avocados went from a median $30 to $60), the 2021 Pacific Northwest heat wave
  that cut the onion crop, the 2022 INSV lettuce shortage ($30 to $68 for a carton of 24 iceberg heads), the March
  2025 tariff on Mexican cucumbers, and two food-safety outbreaks that fell inside the onion and lettuce shortages.
  No food-safety outbreak, COVID-19 closure or viral trend raised prices on its own; the 2024 SunFed cucumber recall
  and the COVID-19 closures for cucumbers lowered them (`event_supply_vs_demand.csv`, `event_price_impact.csv`).
- **Reddit flags supply events; prices show which ones mattered.** Supply talk rose around 16 of the 33 events,
  including 8 of the 10 food-safety outbreaks, where recalls count as supply talk. Most of these did not move national
  wholesale prices.
- **Timing:** supply talk about the 2022 avocado import halt surged the day after its Saturday announcement, and
  prices rose the following week. Lettuce prices began to rise in mid-August 2022; most talk about the shortage came
  in mid-November, the week prices peaked. Across all weeks, talk and prices correlate weakly at any lag (|r| ≤ 0.22),
  mostly in the same week (`lead_lag_crosscorrelation.csv`, `event_timing.csv`).
- **Weighting texts by upvotes or comments finds no more events than counting each text once**
  (`weighting_robustness.csv`).
- **Online signals do not forecast national wholesale prices.** Adding Reddit, Google Trends, news coverage, FDA
  recalls or Wikipedia pageviews to price history improves neither forecasts 1 and 4 weeks ahead nor the warning of big
  price moves (`forecast_check/`, `outputs/tables/forecast_long_*.csv`).

Details: report (`report/`), notebook sections 5, 7–9, 11 and 13, tables in `outputs/tables/` (`event_supply_vs_demand.csv`,
`event_price_impact.csv`, `event_price_examples_usd.csv`, `lead_lag_crosscorrelation.csv`, `event_timing.csv`).

## Repository layout

```
config/                     products, keyword dictionary, subreddits, AMS reports, documented events
data_collection/            Task 1: dataset generation code (collect, clean, tag, summarise)
data/raw/reddit/            raw Reddit snapshot (gzipped JSON Lines) and weekly post counts
data/raw/google_trends/     Google Trends downloads, one file per search term
data/processed/             2022–2024: event dataset, keyword dictionary, summary table, AMS prices, model outputs
data/processed_2019_2021/   the same files for 2019–2021 (event dataset gzipped)
data/processed_2025_2026/   the same files for January 2025 – September 2026
labeling/                   codebook, sample and review scripts, AI annotations, hand-reviewed labels
analysis/                   Task 2: classifiers, price index, signals, timing (imported by the notebook)
notebooks/                  task2_signals_and_prices.ipynb (Task 2 analysis) and its builder script
outputs/figures, tables/    figures and tables used in the report
forecast_check/             forecast check (do online signals help forecast prices?) and the scripts that collect
                            the 2019–2021 and 2025–2026 periods (its own README)
docs/data_dictionary.md     data dictionary and provenance (source, access date, license, coverage, ...)
report/                     PDF report
```

## Deliverables

| Challenge deliverable | File |
|---|---|
| Event dataset (CSV) | `data/processed/event_dataset.csv` (2022–2024), `data/processed_2019_2021/event_dataset.csv.gz`, `data/processed_2025_2026/event_dataset.csv.gz` |
| Product and event keyword list | `data/processed/keyword_dictionary.csv` (also `.json`); source lists in `config/products.yaml`, `config/event_keywords.yaml` |
| Summary table | `summary_table.csv` (also `.md`) in each of the three `data/processed*/` folders |
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

**Full path (about 20 hours on a 2-core CPU for 2022–2024):** re-collects all data and retrains the models.
`config/products.yaml` sets the main period (2022–2024, the years of the labelled sample); the other two periods are
collected with the same pipeline and classifier by `forecast_check/build_period.sh` (about 10 hours each, see
`forecast_check/README.md`).

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

Kiro used Python 3.13.18 and finished in about 10 minutes:

| Step | Time |
|---|---|
| `pip install -r requirements.txt` (pip cache already filled by an earlier run) | 35 s |
| `bash run_all.sh --analysis` (the forecast check takes most of it) | 8 min 25 s |
| `python scripts/check_reproduction.py` | under 1 s; 23 of 23 tables reproduce |

An earlier Kiro run found two problems, both fixed in commit `3464586`: the pinned numpy and scipy need Python 3.12
rather than 3.11, and one example text in `unusual_weeks.csv` depended on whether pyarrow was installed.

## Data sources and licences

See `docs/data_dictionary.md`. Reddit content is public and was accessed through the Arctic Shift research archive;
user names are replaced by pseudonymous ids. USDA AMS Market News data are US Government works in the public domain.
Search interest data: source Google Trends (https://trends.google.com).
Forecast check only: news coverage from the GDELT Project (https://www.gdeltproject.org), food recalls from openFDA
(public domain) and Wikipedia pageviews from the Wikimedia Foundation (CC0).

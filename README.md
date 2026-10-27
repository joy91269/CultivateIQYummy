# CultivateIQ Yummy — Social Media Price Signals for Fresh Produce

Submission for the **Cultivate IQ – NAFSI Track 3 (Supply Chain)** student data challenge (fall 2026).

We turn Reddit discussion about four fresh produce products into weekly **price, supply and demand signals** and
compare them with **USDA AMS terminal market wholesale prices**, 2022–2024.

| | |
|---|---|
| Products | avocados, lettuce (iceberg and romaine), onions, cucumbers |
| Period | 1 January 2022 – 31 December 2024 |
| Online data | Reddit posts and comments from 29 topical subreddits (consumers, grocery retailers, food service, growers, economy) and 9 city subreddits, plus r/news headlines, via the public Arctic Shift archive API: 1.5 M posts scanned, 92,424 texts kept |
| Market benchmark | USDA AMS Specialty Crops terminal market reports, 10 cities, and AMS daily movement (shipment) reports (MyMarketNews MARS API) |
| Labels | 2,946 texts labelled price / supply / demand / other, one per Reddit thread (`labeling/`): 1,118 **gold** labels (two Claude annotators, every disagreement and a random check of agreements decided by a person, a blind check by OpenAI Codex and ChatGPT with blind adjudication; used for training and all accuracy figures) and 1,828 **silver** labels (the two Claude annotators agree; training only) |
| Classifier | gte-modernbert-base sentence embeddings + logistic regression trained on gold + silver labels; macro-F1 0.67 (95% CI 0.63–0.72), accuracy 0.86 on the gold labels in thread-grouped 5-fold CV (keyword rules, zero-shot BART, fine-tuned DistilRoBERTa and three sentence encoders with and without silver labels compared) |
| Validation events | 8 documented events with dates and news sources (`config/events_reference.csv`) |

## Results at a glance

- **Two documented events were supply shocks, and the market data confirm it.** Prices rose while shipments fell:
  the February 2022 halt of Mexican avocado imports (+15% after seasonal adjustment, shipments −33%) and the 2022
  INSV lettuce shortage (+94%, shipments −13%; a carton of 24 iceberg heads went from a median $30 to $68).
- **Reddit reacts to supply disruptions and recalls at once.** Avocado supply talk rose 38-fold in the week of the
  import halt; lettuce supply and price talk crossed their alert level in the same week wholesale prices began to rise
  in August 2022, 3½ weeks before the shortage was in the news.
- **Supply talk is the signal that finds events.** Weeks with a supply-talk spike (z ≥ 3) fall within two weeks of
  a documented event 35% of the time against 11% by chance (p = 0.003) and cover 7 of the 8 events; spikes in price
  talk, demand talk, volume or sentiment do no better than chance (`outputs/tables/spike_event_enrichment.csv`).
- **Food-safety outbreaks and a viral recipe trend did not move national wholesale prices** (−7% to +3% after
  seasonal adjustment). Online they appear as bursts of supply talk (recalls), while demand talk did not rise.
- **Timing:** for sudden shocks the online signal moves in the same week as prices (a calendar week earlier when the
  news breaks at a weekend). Across all weeks, correlations are weak; lettuce supply talk leads lettuce prices by about
  two weeks and price talk lags them by two weeks (rotation test p < 0.01).

Details: report (`report/`), notebook sections 5 and 7–9, tables in `outputs/tables/` (`event_supply_vs_demand.csv`,
`event_price_impact.csv`, `event_price_examples_usd.csv`, `lead_lag_crosscorrelation.csv`, `event_timing.csv`).

## Repository layout

```
config/                     products, keyword dictionary, subreddits, AMS reports, documented events
data_collection/            Task 1: dataset generation code (collect, clean, tag, summarise)
data/raw/reddit/            raw Reddit snapshot (gzipped JSON Lines) and weekly post counts
data/processed/             event dataset, keyword dictionary, summary table, AMS prices, model outputs
labeling/                   codebook, sample and review scripts, AI annotations, hand-reviewed labels
analysis/                   Task 2: classifiers, price index, signals, timing (imported by the notebook)
notebooks/                  task2_signals_and_prices.ipynb (Task 2 analysis) and its builder script
outputs/figures, tables/    figures and tables used in the report
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

Python 3.11 or newer (developed with Python 3.13 on Linux, CPU only).

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

**Quick path (about 5 minutes):** all collected data and model outputs are committed, so the analysis can be rerun
directly:

```bash
bash run_all.sh --analysis
```

This rebuilds and executes `notebooks/task2_signals_and_prices.ipynb`, which writes every figure to
`outputs/figures/` and every table to `outputs/tables/`.

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
| 6. Event dataset | `python data_collection/build_event_dataset.py` | ~1 min | `event_dataset.csv`, `summary_table.*`, `filter_log.csv` |
| 7. Models | `python -m analysis.run_models all` | ~10 h | `outputs/tables/model_cv_*.csv`, `data/processed/classified_observations.csv.gz` |
| 8. Notebook | `bash run_all.sh --analysis` | ~5 min | `outputs/` |

Notes
- Collection steps are resumable: rerunning a command continues where it stopped.
- Arctic Shift is a live archive; content deleted on Reddit can disappear from it, so a fresh collection can differ
  slightly from the committed snapshot in `data/raw/reddit/`. Step 6 rebuilds the dataset from that snapshot when
  the raw files are not re-downloaded.
- The labelled sample (`labeling/labels.csv`) is part of the repository; `labeling/make_label_sample.py` shows how it
  was drawn, `labeling/codebook.md` how it was labelled (including the AI annotation and human review procedure),
  and `labeling/annotations/` holds the labels and reasons of all four AI annotators (two Claude models; OpenAI Codex
  and ChatGPT, whose task is in `labeling/codex_task/`). `labeling/adjudication.json` holds the blind A/B decisions,
  and `labeling/build_collab_page.py` / `collect_collab.py` the second human annotator's sample and agreement.
- Model steps cache their intermediate results in `.cache/` (not committed) and resume after an interruption.
- If model downloads from Hugging Face fail behind a restrictive proxy, set `HF_HUB_DISABLE_XET=1`.

## Data sources and licences

See `docs/data_dictionary.md`. Reddit content is public and was accessed through the Arctic Shift research archive;
user names are replaced by pseudonymous ids. USDA AMS Market News data are US Government works in the public domain.

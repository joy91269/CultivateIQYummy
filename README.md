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
| Labels | 794 texts labelled price / supply / demand / other by two AI annotators, with every disagreement and a random check reviewed by a person (`labeling/`) |
| Classifier | gte-modernbert-base sentence embeddings + logistic regression; macro-F1 0.67, accuracy 0.86 in thread-grouped 5-fold CV (six models compared) |
| Validation events | 8 documented events with dates and news sources (`config/events_reference.csv`) |

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

**Full path (about 8–10 hours on a 2-core CPU):** re-collects all data and retrains the models.

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
| 7. Models | `python -m analysis.run_models all` | ~6 h | `outputs/tables/model_cv_*.csv`, `data/processed/classified_observations.csv.gz` |
| 8. Notebook | `bash run_all.sh --analysis` | ~5 min | `outputs/` |

Notes
- Collection steps are resumable: rerunning a command continues where it stopped.
- Arctic Shift is a live archive; content deleted on Reddit can disappear from it, so a fresh collection can differ
  slightly from the committed snapshot in `data/raw/reddit/`. Step 6 rebuilds the dataset from that snapshot when
  the raw files are not re-downloaded.
- The labelled sample (`labeling/labels.csv`) is part of the repository; `labeling/make_label_sample.py` shows how it
  was drawn, `labeling/codebook.md` how it was labelled (including the AI annotation and human review procedure),
  and `labeling/annotations/` holds both AI annotators' labels and reasons.
- Model steps cache their intermediate results in `.cache/` (not committed) and resume after an interruption.
- If model downloads from Hugging Face fail behind a restrictive proxy, set `HF_HUB_DISABLE_XET=1`.

## Data sources and licences

See `docs/data_dictionary.md`. Reddit content is public and was accessed through the Arctic Shift research archive;
user names are replaced by pseudonymous ids. USDA AMS Market News data are US Government works in the public domain.

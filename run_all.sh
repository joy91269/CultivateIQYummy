#!/usr/bin/env bash
# Full reproduction, from raw data to the notebook. Run from the repository root.
#   bash run_all.sh            everything (about 20 hours on a 2-core CPU, mostly downloads and model inference)
#   bash run_all.sh --analysis only the analysis on the committed data (about 5 minutes)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" != "--analysis" ]]; then
  # Task 1: dataset generation
  python data_collection/build_keywords.py
  python data_collection/collect_reddit.py posts     # resumable; ~3 h
  python data_collection/collect_reddit.py titles    # r/news title search (rate limited)
  python data_collection/collect_reddit.py geo       # posts of the 9 city subreddits (geography); ~1 h
  python data_collection/collect_reddit.py trees     # comment trees of product posts; ~2 h
  python data_collection/collect_ams.py              # needs MARS_API_KEY in .env; ~10 min
  python data_collection/collect_ams_movement.py     # shipments; same key; ~1 h
  python data_collection/collect_google_trends.py    # US search interest; ~3 min
  python data_collection/build_event_dataset.py
  # Task 2: models (the hand-reviewed labels are in labeling/labels.csv)
  python -m analysis.run_models all                  # ~10 h on CPU
fi

python notebooks/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  notebooks/task2_signals_and_prices.ipynb
echo "Done. Figures in outputs/figures, tables in outputs/tables."

#!/usr/bin/env bash
# Collect one extra period with the main pipeline, in a copy of the repository.
#   bash forecast_check/build_period.sh 2019-01-01 2022-01-01 forecast_check/periods/2019-2021
#   bash forecast_check/build_period.sh 2025-01-01 2026-10-01 forecast_check/periods/2025-2026
# Arguments: start date, end date (exclusive), target folder. Run from the repository root; needs MARS_API_KEY
# in .env. Only the study period changes. The classifier is the one trained on the labelled 2022-2024 sample, so
# no new labels are needed. Each period takes about 10 hours, mostly Reddit downloads; every step is resumable.
set -euo pipefail
START=$1; END=$2; DEST=$3
mkdir -p "$DEST"
rsync -a --exclude .git --exclude .venv --exclude .cache --exclude forecast_check/periods \
  --exclude 'data/raw/reddit/*' --exclude 'data/raw/google_trends/*' --exclude 'data/raw/ams*' \
  --exclude 'data/processed/*' --exclude outputs ./ "$DEST/"
cd "$DEST"
python - "$START" "$END" <<'EOF'
import re, sys
p = "config/products.yaml"; s = open(p).read()
s = re.sub(r'(study_period:\s*\n\s*start: )"[^"]*"', rf'\1"{sys.argv[1]}"', s)
s = re.sub(r'(\n\s*end: )"[^"]*"', rf'\1"{sys.argv[2]}"', s, count=1)
open(p, "w").write(s)
EOF
grep -A2 study_period config/products.yaml
python data_collection/build_keywords.py
python data_collection/collect_reddit.py posts
python data_collection/collect_reddit.py titles
python data_collection/collect_reddit.py geo
python data_collection/collect_reddit.py trees
python data_collection/collect_ams.py
python data_collection/collect_ams_movement.py
python data_collection/collect_google_trends.py
python data_collection/build_event_dataset.py
python -m analysis.run_models predict      # applies the trained classifier to the new texts
python -m analysis.run_models sentiment
echo "period $START to $END done in $DEST"

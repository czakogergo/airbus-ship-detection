#!/usr/bin/env bash
set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

python src/01-preprocessing.py
python src/02-training.py
python src/03-evaluation.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
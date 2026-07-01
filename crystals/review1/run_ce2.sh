#!/bin/bash
# C-E2 (converge carry-over ladder). Repo-relative; run from a fresh clone.
# Prereqs: venv from ../requirements.txt active; checkpoint fetched (../fetch_checkpoint.sh);
#   CUES_TRAIN_CSV pointing at the alex_nolemat_lowhull training CSV; HF_TOKEN + MP_API_KEY set.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CUES_TRAIN_CSV:?set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV}"
python -u "$HERE/ce2_converge.py" > "$HERE/ce2.log" 2>&1

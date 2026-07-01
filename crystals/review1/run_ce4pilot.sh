#!/bin/bash
# C-E4 pilot (pre-measurement signals for the ladder prediction). Repo-relative.
# Prereqs: venv active; checkpoint fetched; CUES_TRAIN_CSV set; HF_TOKEN + MP_API_KEY set.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CUES_TRAIN_CSV:?set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV}"
python -u "$HERE/ce4_pilot.py" > "$HERE/ce4_pilot.log" 2>&1

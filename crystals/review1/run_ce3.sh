#!/bin/bash
# C-E3 (joint wide-gap & stable baselines). Repo-relative; run from a fresh clone.
# Prereqs: venv active; checkpoint fetched; CUES_TRAIN_CSV set; HF_TOKEN + MP_API_KEY set.
# Gap uses the composition XGBoost surrogate (../data/xgb_composition_dft_band_gap.json), NOT the
# heavy structural MLIP/Dielectrics (Ray) stack. Stability = single-MACE e_above_hull.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CUES_TRAIN_CSV:?set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV}"
python -u "$HERE/ce3_joint.py" > "$HERE/ce3.log" 2>&1

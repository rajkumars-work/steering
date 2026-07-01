#!/bin/bash
# Auto-retry supervisor for the C-E14 stability re-run. The GPU is shared with a persistent
# web/app.py inference service; during busy windows generate_one returns atoms=None for every
# prompt (run #2 + probe both got 0% -> all-NaN). The python script preflights scorer+generator
# and exits 75 (RETRY_EXIT) on a contended/transient window instead of grinding hours into NaN.
# This supervisor retries on exit 75 until a clean window carries a run to completion (exit 0),
# within a wall-clock deadline. Hard failure (other nonzero) stops. Launch detached:
#   nohup setsid bash run_ce14_stab_super.sh >/dev/null 2>&1 &
D=/home/ubuntu/code/py/dielectric/eval/within_distribution_steering/review1
cd /home/ubuntu/code/py/dielectric
export PYTHONPATH=/home/ubuntu/code/py/dielectric
export HF_TOKEN=$(cat ~/.ssh/hf-read.key|tr -d '[:space:]')
export MP_API_KEY=$(cat ~/.ssh/materials_project.key|tr -d '[:space:]')
LOG="$D/ce14_stability_rerun.log"
SUP="$D/ce14_stability_super.log"
RETRY_SLEEP=600          # 10 min between retries -- lets a contended GPU window clear
DEADLINE=$(( $(date +%s) + 16*3600 ))   # keep trying for up to 16h (a full run is ~5.5h)
echo "SUPER-START $(date -u +%FT%TZ) deadline=$(date -u -d @$DEADLINE +%FT%TZ)" > "$SUP"
att=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  att=$((att+1))
  echo "SUPER attempt $att start $(date -u +%FT%TZ)" >> "$SUP"
  /opt/dlami/nvme/recast/.venv/bin/python -u "$D/ce14_stability_rerun.py" > "$LOG" 2>&1
  rc=$?
  echo "SUPER attempt $att exit=$rc $(date -u +%FT%TZ)" >> "$SUP"
  cp "$LOG" "$D/ce14_stability_rerun.attempt${att}.log"
  if [ $rc -eq 0 ]; then
    echo "SUPER SUCCESS on attempt $att $(date -u +%FT%TZ)" >> "$SUP"; break
  elif [ $rc -eq 75 ]; then
    echo "SUPER retryable (rc=75); sleeping ${RETRY_SLEEP}s" >> "$SUP"; sleep $RETRY_SLEEP
  else
    echo "SUPER HARD FAILURE rc=$rc; stopping" >> "$SUP"; break
  fi
done
echo "SUPER-END $(date -u +%FT%TZ) attempts=$att" >> "$SUP"

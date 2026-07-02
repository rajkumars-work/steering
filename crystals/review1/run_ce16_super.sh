#!/bin/bash
# Auto-retry supervisor for C-E16 (strongest/audit-calibrated knob). Same pattern as the C-E14
# stability supervisor: the ce16 python preflights scorer+generator and exits 75 (RETRY_EXIT) if a
# contended GPU window returns atoms=None for every prompt, instead of grinding into empty pools.
# This supervisor retries on exit 75 until a clean window completes (exit 0), within a deadline.
# The GPU-sharing web demo (recast-demo.service) is stopped by the launcher before this runs, so
# contention should not occur; the preflight/retry is kept as a belt-and-suspenders guard.
# Launch detached:  nohup setsid bash run_ce16_super.sh >/dev/null 2>&1 &
D=/home/ubuntu/code/py/dielectric/eval/within_distribution_steering/review1
cd /home/ubuntu/code/py/dielectric
export PYTHONPATH=/home/ubuntu/code/py/dielectric:/home/ubuntu/code/py/ed:/home/ubuntu/packages/lemat-genbench/src
export HF_TOKEN=$(cat ~/.ssh/hf-read.key|tr -d '[:space:]')
export MP_API_KEY=$(cat ~/.ssh/materials_project.key|tr -d '[:space:]')
LOG="$D/ce16_strongest_knob.log"
SUP="$D/ce16_super.log"
RETRY_SLEEP=600
DEADLINE=$(( $(date +%s) + 16*3600 ))
echo "SUPER-START $(date -u +%FT%TZ) deadline=$(date -u -d @$DEADLINE +%FT%TZ)" > "$SUP"
att=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  att=$((att+1))
  echo "SUPER attempt $att start $(date -u +%FT%TZ)" >> "$SUP"
  /opt/dlami/nvme/recast/.venv/bin/python -u "$D/ce16_strongest_knob.py" > "$LOG" 2>&1
  rc=$?
  echo "SUPER attempt $att exit=$rc $(date -u +%FT%TZ)" >> "$SUP"
  cp "$LOG" "$D/ce16_strongest_knob.attempt${att}.log"
  if [ $rc -eq 0 ]; then
    echo "SUPER SUCCESS on attempt $att $(date -u +%FT%TZ)" >> "$SUP"; break
  elif [ $rc -eq 75 ]; then
    echo "SUPER retryable (rc=75); sleeping ${RETRY_SLEEP}s" >> "$SUP"; sleep $RETRY_SLEEP
  else
    echo "SUPER HARD FAILURE rc=$rc; stopping" >> "$SUP"; break
  fi
done
echo "SUPER-END $(date -u +%FT%TZ) attempts=$att" >> "$SUP"

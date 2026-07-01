#!/bin/bash
D=/home/ubuntu/code/py/dielectric/eval/within_distribution_steering/review1
cd /home/ubuntu/code/py/dielectric
export PYTHONPATH=/home/ubuntu/code/py/dielectric
export HF_TOKEN=$(cat ~/.ssh/hf-read.key|tr -d '[:space:]')
export MP_API_KEY=$(cat ~/.ssh/materials_project.key|tr -d '[:space:]')
/opt/dlami/nvme/recast/.venv/bin/python -u "$D/ce14_stability_rerun.py" > "$D/ce14_stability_rerun.log" 2>&1
echo "CE14-STAB-RERUN-EXIT $(date -u +%FT%TZ)" >> "$D/ce14_stability_rerun.log"

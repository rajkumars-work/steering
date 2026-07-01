#!/usr/bin/env python3
"""C-E1 (partial): bootstrap CIs on the crystal numbers we already have saved arrays for.
The ladder rungs that lack saved per-item data (BVS/stability r2, Combined sweep) get their CIs
from the C-E2/C-E3 reruns instead. No GPU.

Outputs review1/ce1_bootstrap_existing.json + a printed table.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _repo import RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.join(RESULTS, "claim1_survival_crystals_density.json")
DPO = os.path.join(RESULTS, "dpo_logprob_fig.json")
B = 10000
rng = np.random.default_rng(0)


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


out = {}

# --- density carry-over r^2 (bootstrap over chemistries) ---
d = json.load(open(DENSITY))
dm = np.array([b["data_mean"] for b in d["bins"]])
mm = np.array([b["model_mean"] for b in d["bins"]])
n = len(dm)
r2s = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    x, y = dm[idx], mm[idx]
    if np.std(x) > 0 and np.std(y) > 0:
        r2s.append(np.corrcoef(x, y)[0, 1] ** 2)
r2_point = float(np.corrcoef(dm, mm)[0, 1] ** 2)
lo, hi = ci(r2s)
out["density_carryover_r2"] = {"point": round(r2_point, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "n_chemistries": n, "method": "bootstrap over chemistries (B=%d)" % B}

# --- DPO log-prob gap (bootstrap over structures) ---
g = json.load(open(DPO))
base = np.array(g["base_lp"]); dpo = np.array(g["dpo_lp"]); uniform = g["uniform"]
nb = len(base)
bm, dmn, gap, frac_below = [], [], [], []
for _ in range(B):
    idx = rng.integers(0, nb, nb)
    bm.append(base[idx].mean()); dmn.append(dpo[idx].mean())
    gap.append(base[idx].mean() - dpo[idx].mean())
    frac_below.append(np.mean(dpo[idx] < uniform))
out["dpo_logprob"] = {
    "base_mean": {"point": round(float(base.mean()), 3), "ci95": [round(x, 3) for x in ci(bm)]},
    "dpo_mean": {"point": round(float(dpo.mean()), 3), "ci95": [round(x, 3) for x in ci(dmn)]},
    "base_minus_dpo_gap": {"point": round(float(base.mean() - dpo.mean()), 3), "ci95": [round(x, 3) for x in ci(gap)]},
    "frac_dpo_below_uniform": {"point": round(float(np.mean(dpo < uniform)), 3), "ci95": [round(x, 3) for x in ci(frac_below)]},
    "uniform_baseline": round(uniform, 3), "n_structures": nb, "method": "bootstrap over structures (B=%d)" % B}

json.dump(out, open(os.path.join(HERE, "ce1_bootstrap_existing.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
print("CE1-BOOTSTRAP-EXISTING-DONE")

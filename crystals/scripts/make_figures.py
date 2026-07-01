#!/usr/bin/env python3
"""Regenerate the crystal-domain Claim 1 figures from the shipped results.

    python scripts/make_figures.py        # writes crystals/figures/*.png

Needs only numpy + matplotlib. Reads results/claim1_survival_spectrum.json (the difficulty-ladder
run); no model or GPU. Mirror of images/scripts/make_figures.py.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "claim1_survival_spectrum.json")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

res = json.load(open(RES))

# ---- (B) budget-survival r^2 ladder: density -> BVS-GII -> e_above_hull ----
order = ["density", "bvs_gii", "e_above_hull"]
labels = {"density": "density\n(geometry)", "bvs_gii": "BVS-GII\n(bonding)",
          "e_above_hull": "e_above_hull\n(stability)"}
r2 = [res["survival"][p].get("per_bin_mean_r2", float("nan")) for p in order]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(len(order))
ax.plot(x, r2, "-o", color="#2b6cb0", lw=2, ms=9)
for xi, yi in zip(x, r2):
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([labels[p] for p in order])
ax.set_ylim(0, 1.05); ax.set_ylabel("per-chemistry mean $r^2$ (data vs model)")
ax.set_title("Claim 1: budget survival degrades with property difficulty")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_claim1_survival_ladder.png"), dpi=160)
plt.close(fig)

# ---- (A) convergence ladder: pass rate, data vs model ----
checks = ["smact_pass", "bvs_pass", "validity_pass", "metastable"]
clabel = {"smact_pass": "SMACT", "bvs_pass": "BVS", "validity_pass": "validity", "metastable": "metastable"}
data_r = [res["ladder"][c]["data_rate"] for c in checks]
model_r = [res["ladder"][c]["model_rate"] for c in checks]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = np.arange(len(checks)); w = 0.38
ax.bar(x - w/2, data_r, w, label="data", color="#a0aec0")
ax.bar(x + w/2, model_r, w, label="model", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels([clabel[c] for c in checks])
ax.set_ylim(0, 1.05); ax.set_ylabel("pass rate")
ax.set_title("Claim 1: convergence ladder (easy → hard)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_claim1_convergence_ladder.png"), dpi=160)
plt.close(fig)

print("wrote", os.path.join(FIG, "fig_claim1_survival_ladder.png"))
print("wrote", os.path.join(FIG, "fig_claim1_convergence_ladder.png"))

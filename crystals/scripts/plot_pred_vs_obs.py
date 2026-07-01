#!/usr/bin/env python3
"""fig_recast_pred_vs_obs.png (Claim 4 — plain-pretraining diagonal).

Per-chemistry predicted (data mean) vs observed (model mean) DENSITY, points on the diagonal,
annotated with r². Pure plotting — reads the density run's results JSON; no model or GPU.

    python scripts/plot_pred_vs_obs.py        # -> figures/fig_recast_pred_vs_obs.png

Input: results/claim1_survival_crystals_density.json (produced by
scripts/claim1_survival_crystals.py, whose `bins` list carries per-bin data_mean/model_mean).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results", "claim1_survival_crystals_density.json")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

d = json.load(open(SRC))
bins = d["bins"]
x = np.array([b["data_mean"] for b in bins])      # predicted = data per-chemistry mean
y = np.array([b["model_mean"] for b in bins])     # observed  = model per-chemistry mean
r2 = d.get("per_bin_mean_r2")
if r2 is None:
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)

lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
pad = 0.05 * (hi - lo); lo -= pad; hi += pad

fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1.2, zorder=1, label="y = x")
ax.scatter(x, y, s=42, color="#2b6cb0", alpha=0.8, edgecolor="white", linewidth=0.5, zorder=2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("predicted density  (data per-chemistry mean, g/cm³)")
ax.set_ylabel("observed density  (model per-chemistry mean, g/cm³)")
ax.set_title("Claim 4: model reproduces the per-chemistry density")
ax.annotate(f"$r^2 = {r2:.2f}$\n{len(bins)} chemistries", xy=(0.05, 0.86),
            xycoords="axes fraction", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_recast_pred_vs_obs.png"), dpi=160)
plt.close(fig)
print("wrote figures/fig_recast_pred_vs_obs.png (r2=%.3f, n=%d)" % (r2, len(bins)))

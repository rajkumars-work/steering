#!/usr/bin/env python3
"""Figures for the review.1 crystal experiments. Pure plotting (numpy+matplotlib, no GPU); reads
the result JSONs from ../results/. Writes to ../figures/ and the paper figures dir.

  fig_ce2_ladder.png      — converged carry-over ladder r² (±95% CI) + m-sweep convergence
  fig_ce3_joint.png       — joint (wide-gap∧stable) hit-rate (±Wilson CI): telling vs showing
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT_DIRS = [os.path.join(ROOT, "figures"),
            "/home/ubuntu/code/py/dielectric/docs/steering_paper/figures"]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=160)
    plt.close(fig)
    print("wrote", name)


# ---- C-E2: converged ladder + m-sweep ----
ce2 = json.load(open(os.path.join(RES, "ce2_converge.json")))
props = ["density", "bvs_gii", "e_above_hull"]
lab = {"density": "density", "bvs_gii": "BVS-GII", "e_above_hull": "stability"}
cl = ce2["converged_ladder"]
pts = [cl[p]["r2_point"] for p in props]
los = [cl[p]["r2_point"] - cl[p]["ci95"][0] for p in props]
his = [cl[p]["ci95"][1] - cl[p]["r2_point"] for p in props]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
x = np.arange(len(props))
ax1.errorbar(x, pts, yerr=[los, his], fmt="o", color="#2b6cb0", ms=10, capsize=6, lw=2)
for xi, yi in zip(x, pts):
    ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=11, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([lab[p] for p in props])
ax1.set_ylim(0, 1.05); ax1.set_ylabel("per-chemistry carry-over $r^2$")
ax1.set_title("Converged ladder (3 seeds, m=30) ±95% CI"); ax1.grid(axis="y", alpha=0.3)

conv = ce2["convergence"]
ms = sorted(int(m) for m in conv[props[0]])
for p, c in zip(props, ["#2b6cb0", "#38a169", "#dd6b20"]):
    ax2.plot(ms, [conv[p][str(m)]["seed_mean"] for m in ms], "-o", color=c, label=lab[p])
ax2.set_xlabel("samples / chemistry (m)"); ax2.set_ylabel("$r^2$ (seed mean)")
ax2.set_ylim(0, 1.05); ax2.set_title("Fit stabilizes with m"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig.tight_layout(); save(fig, "fig_ce2_ladder.png")


# ---- C-E3: joint hit-rate bars ----
ce3 = json.load(open(os.path.join(RES, "ce3_joint.json")))["conditions"]
order = ["telling_naive", "telling_strong", "showing"]
nm = {"telling_naive": "telling\n(naive)", "telling_strong": "telling\n(strong)", "showing": "showing"}
rates = [ce3[k]["joint_rate"] for k in order]
cis = [ce3[k]["wilson_ci95"] for k in order]
lo = [r - c[0] for r, c in zip(rates, cis)]
hi = [c[1] - r for r, c in zip(rates, cis)]

fig, ax = plt.subplots(figsize=(5.4, 3.9))
x = np.arange(len(order))
cols = ["#a0aec0", "#dd6b20", "#2b6cb0"]
ax.bar(x, rates, 0.6, color=cols, yerr=[lo, hi], capsize=6)
for xi, r in zip(x, rates):
    ax.annotate(f"{r*100:.1f}%", (xi, r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([nm[k] for k in order])
ax.set_ylabel("joint hit-rate  (wide-gap ∧ stable)")
ax.set_ylim(0, max(rates) * 1.35)
ax.set_title("Claim 3: joint hit-rate ±Wilson 95% CI")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "fig_ce3_joint.png")
print("DONE")

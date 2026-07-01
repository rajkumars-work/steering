#!/usr/bin/env python3
"""C-E12 figures + threshold sensitivity. Reads ce12_coverage.json (per-structure scatter), writes
the coverage bar chart and the 2-D (gap, eps0) scatter per method, and recomputes coverage under a
sensitivity threshold set. Pure plotting; no GPU."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ce12_coverage.json")
OUT_DIRS = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "steering/crystals/figures"),
            "/home/ubuntu/code/py/steering/crystals/figures",
            "/home/ubuntu/code/py/dielectric/docs/steering_paper/figures"]
OUT_DIRS = sorted(set(d for d in OUT_DIRS if os.path.isdir(os.path.dirname(d)) or True))
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
d = json.load(open(SRC))
M = d["methods"]; order = ["telling", "conditioning", "showing"]
nm = {"telling": "telling\n(1 chem)", "conditioning": "conditioning\n(bg-vhigh)", "showing": "showing\n(mix A+B)"}
A_GAP, A_EPS = d["corner_A"]; B_GAP, B_EPS = d["corner_B"]


def save(fig, name):
    for od in OUT_DIRS:
        fig.savefig(os.path.join(od, name), dpi=160)
    plt.close(fig); print("wrote", name)


# --- coverage bar ---
fig, ax = plt.subplots(figsize=(5.4, 3.9))
cov = [M[k]["coverage"] for k in order]; ci = [M[k]["coverage_ci95"] for k in order]
lo = [c - x[0] for c, x in zip(cov, ci)]; hi = [x[1] - c for c, x in zip(cov, ci)]
ax.bar(range(3), cov, 0.6, color=["#a0aec0", "#dd6b20", "#2b6cb0"], yerr=[lo, hi], capsize=6)
for i, c in enumerate(cov): ax.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([nm[k] for k in order])
ax.set_ylabel("coverage = min(frac A, frac B)"); ax.set_ylim(0, max(cov) * 1.3 + 0.02)
ax.set_title("C-E12: only showing covers both corners of the frontier"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "fig_ce12_coverage.png")

# --- 2-D scatter per method ---
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True)
for ax, k in zip(axes, order):
    sc = M[k]["scatter"]; g = np.array([r["gap"] for r in sc]); e = np.array([r["eps0"] for r in sc])
    col = ["#2b6cb0" if r["corner"] == "A" else "#dd6b20" if r["corner"] == "B" else "#cbd5e0" for r in sc]
    ax.axvspan(A_GAP, max(6, g.max()), ymin=0, ymax=A_EPS / 60, alpha=0.06, color="blue")
    ax.scatter(g, e, s=14, c=col, alpha=0.6, edgecolor="none")
    ax.axvline(A_GAP, ls=":", c="#2b6cb0", lw=1); ax.axhline(A_EPS, ls=":", c="#2b6cb0", lw=1)
    ax.axvline(B_GAP, ls=":", c="#dd6b20", lw=1); ax.axhline(B_EPS, ls=":", c="#dd6b20", lw=1)
    ax.set_title(f"{k}\ncoverage={M[k]['coverage']:.2f}"); ax.set_xlabel("band gap (eV)")
axes[0].set_ylabel("dielectric ε₀")
fig.suptitle("C-E12 (gap, ε₀): corner A = high-gap/low-ε (blue), corner B = low-gap/high-ε (orange)", fontsize=10)
fig.tight_layout(); save(fig, "fig_ce12_scatter.png")

# --- threshold sensitivity (re-classify saved scatter) ---
def cov_at(ag, ae, bg, be):
    out = {}
    for k in order:
        sc = M[k]["scatter"]; n = len(sc)
        A = sum(1 for r in sc if r["gap"] >= ag and r["eps0"] <= ae)
        B = sum(1 for r in sc if r["gap"] <= bg and r["eps0"] >= be)
        out[k] = round(min(A / n, B / n), 3)
    return out
sens = {"primary_A>=%.1f,eps<=%.0f_B<=%.1f,eps>=%.0f" % (A_GAP, A_EPS, B_GAP, B_EPS): cov_at(A_GAP, A_EPS, B_GAP, B_EPS),
        "loose_A>=2.5,eps<=28_B<=1.5,eps>=31": cov_at(2.5, 28, 1.5, 31),
        "tight_A>=3.5,eps<=23_B<=0.8,eps>=35": cov_at(3.5, 23, 0.8, 35)}
json.dump(sens, open(os.path.join(HERE, "ce12_sensitivity.json"), "w"), indent=2)
print("sensitivity:", json.dumps(sens))
print("CE12-FIGURE-DONE")

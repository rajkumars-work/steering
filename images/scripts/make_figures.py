#!/usr/bin/env python3
"""Regenerate the image-domain paper figures from the shipped data (distributions/).

    python scripts/make_figures.py        # writes images/figures/*.png

Needs only numpy + matplotlib — no model, no GPU. Reads the shadow (claim1_perclass.npz)
and the run results (claim3_images.json, claim2_chi2.json). Produces the four image-domain
figures the Evidence section uses:
  fig_brightness_split        (Claim 1: the split, fine vs coarse bins)
  fig_label_vs_exemplar_spread(Claim 2: telling's vs showing's reach)
  fig_chi2_reach              (Claim 2: showing's reach vs concentration, under the ceiling)
  fig_joint_target_bars       (Claim 3: animal-AND-nature hit rate)
(The crystal-side figures — survival ladder, MLE diagonal, DPO log-prob — are produced by the
crystals/ track.)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "distributions")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
d = np.load(os.path.join(DIST, "claim1_perclass.npz"))
cnt = d["count"].astype(float)


def split(c, g, v):
    w = c / c.sum(); gbar = (w * g).sum()
    return float((w * v).sum()), float((w * (g - gbar) ** 2).sum())   # within T, between E


# ---- Claim 1: brightness split, fine bins vs coarse (from the shadow) ----
g, v = d["g_brightness"], d["v_brightness"]
Tf, Ef = split(cnt, g, v)
grp = np.random.default_rng(0).integers(0, 50, size=len(cnt))
n = np.bincount(grp, cnt, 50); sg = np.bincount(grp, cnt * g, 50)
sgg = np.bincount(grp, cnt * (v + g * g), 50); nz = n > 0
gc = np.zeros(50); gc[nz] = sg[nz] / n[nz]; vc = np.zeros(50); vc[nz] = sgg[nz] / n[nz] - gc[nz] ** 2
Tc, Ec = split(n, gc, vc)
fig, ax = plt.subplots(figsize=(4.4, 3.4)); x = np.arange(2)
ax.bar(x, [Tf, Tc], 0.55, label="within-bin (telling's reach)", color="#4C72B0")
ax.bar(x, [Ef, Ec], 0.55, bottom=[Tf, Tc], label="between-bin (showing's reach)", color="#DD8452")
for i, (T, E) in enumerate([(Tf, Ef), (Tc, Ec)]):
    ax.text(i, T + E + 0.0005, f"total {T+E:.4f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["1000 fine bins\n(classes)", "50 coarse bins\n(merged)"])
ax.set_ylabel("brightness variance"); ax.set_ylim(0, 0.024)
ax.legend(fontsize=8, loc="lower center"); ax.set_title("Total fixed; coarsening shifts the split inward", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_brightness_split.png"), dpi=150)

# ---- Claim 2: telling's reach (one bin) vs showing's reach (across bins) ----
gA, vA = d["g_sim_animal"], d["v_sim_animal"]
w = cnt / cnt.sum(); gbar = float((w * gA).sum()); sqrtT = float((w * vA).sum()) ** 0.5
lo, hi = float(gA.min()), float(gA.max())
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.hist(gA, bins=45, weights=w, color="#dddddd", label="where the bins sit")
ax.axvline(gbar, color="k", lw=1.0, ls=":")
ax.annotate("start", xy=(gbar, 0), xytext=(gbar, 0.066), ha="center", fontsize=8)
ax.annotate("", xy=(gbar + sqrtT, 0.040), xytext=(gbar - sqrtT, 0.040), arrowprops=dict(arrowstyle="<->", color="#4C72B0", lw=2.2))
ax.text(gbar, 0.046, "telling: within one bin", color="#4C72B0", ha="center", fontsize=8.5)
ax.annotate("", xy=(hi, 0.020), xytext=(lo, 0.020), arrowprops=dict(arrowstyle="<->", color="#DD8452", lw=2.2))
ax.text((lo + hi) / 2, 0.010, "showing: across all the bins", color="#DD8452", ha="center", fontsize=8.5)
ax.set_yticks([]); ax.set_xlabel("target value (how much an image looks like an animal)")
ax.set_title("How far telling and showing can move the average", fontsize=10)
ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(0, 0.085); ax.set_xlim(lo - 0.015, hi + 0.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_label_vs_exemplar_spread.png"), dpi=150)

# ---- Claim 2: showing's reach grows with concentration, under the ceiling ----
c2 = json.load(open(os.path.join(DIST, "claim2_chi2.json")))
names = {"sim_animal": "animal score", "brightness": "brightness"}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
for ax, (t, td) in zip(axes, c2["targets"].items()):
    chi = [L["chi2"] for L in td["levels"]]; ceil = [L["ceiling"] for L in td["levels"]]
    real = [L["realized_shift"] for L in td["levels"]]
    ax.fill_between(chi, 0, ceil, color="#DD8452", alpha=0.15, label="reachable (≤ ceiling)")
    ax.plot(chi, ceil, "--", color="#DD8452", lw=1.8, label="ceiling  √(χ²·E)")
    ax.plot(chi, real, "-o", color="#4C72B0", lw=2, ms=6, label="realized shift")
    ax.set_xscale("log"); ax.set_xlabel("recipe concentration  χ²"); ax.set_title(names.get(t, t), fontsize=10)
    ax.set_ylim(0, max(ceil) * 1.08)
axes[0].set_ylabel("mean shift from baseline"); axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Showing's reach grows with concentration, and stays under the budget ceiling", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_chi2_reach.png"), dpi=150)

# ---- Claim 3: animal AND nature joint hit rate ----
p = json.load(open(os.path.join(DIST, "claim3_images.json")))["pools"]
vals = [p["telling_animal"]["joint_hit_pct"], p["telling_nature"]["joint_hit_pct"], p["showing_joint"]["joint_hit_pct"]]
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.bar(range(3), vals, 0.6, color=["#C44E52", "#C44E52", "#55A868"])
for i, vv in enumerate(vals): ax.text(i, vv + 0.6, f"{vv:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(['telling\n"animal"', 'telling\n"nature"', "showing\njoint bins"])
ax.set_ylabel("joint hit rate (%)"); ax.set_ylim(0, 42)
ax.set_title('"animal AND nature": only showing composes it', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_joint_target_bars.png"), dpi=150)

print("wrote 4 image figures to", FIG)

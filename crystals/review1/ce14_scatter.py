#!/usr/bin/env python3
"""C-E14 primary figure: bin/knob shift ratio vs the between-fraction E/(E+T), one point per property.
Budget law: showing's reach ∝ √(χ²·E), telling's ∝ √(χ²·T) -> ratio should rise with E/(E+T).
Reads ce14_panel.json (+ ce14_gap_et.json for gap's E/(E+T)). Writes fig_ce14_bins_vs_knobs.png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "ce14_panel.json")))["panel"]
gap_et = json.load(open(os.path.join(HERE, "ce14_gap_et.json")))["E_over_EplusT"]
if P["gap"].get("E_over_EplusT") is None:
    P["gap"]["E_over_EplusT"] = gap_et

order = ["stability", "gap", "density"]
xs = [P[p]["E_over_EplusT"] for p in order]
ys = [P[p]["bin_over_knob"] for p in order]
# ratio CI can go non-positive when knob_shift (denominator) straddles 0 (stability);
# clip the lower whisker to a positive floor so it renders on the log axis (true CI is in the json).
FLOOR = 0.04
lo = [max(P[p]["ratio_ci95"][0], FLOOR) for p in order]
hi = [P[p]["ratio_ci95"][1] for p in order]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.axhline(1.0, color="0.6", lw=1, ls="--", zorder=1)
ax.text(0.02, 1.15, "bins = knobs", color="0.5", fontsize=8, transform=ax.get_yaxis_transform())
yerr = np.array([[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]])
ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=9, color="#1f4e79", ecolor="#9bb8d6",
            elinewidth=1.6, capsize=4, zorder=3)
for p, x, y in zip(order, xs, ys):
    ax.annotate(f"{p}\n(ratio {y:g}×)", (x, y), textcoords="offset points",
                xytext=(10, 6), fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("between-chemistry fraction of variance,  E/(E+T)")
ax.set_ylabel("bin / knob shift ratio  (log scale)")
ax.set_title("C-E14: where does steering live? bins vs property-tag knobs")
ax.set_xlim(0.45, 1.0)
ax.grid(True, which="both", axis="y", alpha=0.25)
fig.tight_layout()
out = os.path.join(HERE, "fig_ce14_bins_vs_knobs.png")
fig.savefig(out, dpi=150)
print("wrote", out)
print("points:", list(zip(order, xs, ys)))

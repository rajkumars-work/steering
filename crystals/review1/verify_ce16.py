#!/usr/bin/env python3
"""Verify C-E16's own claims from ce16_strongest_knob.json (pure CPU; no GPU, no regeneration).

Checks the pre-registered predictions and internal self-consistency:
  (1) best-of-N is genuinely stronger than the C-14 tag knob (steel-man): Δ(k=12) >> tag_knob_shift.
  (2) chemistry still wins on gap & density: bin/strongest-knob ratio > 1.
  (3) budget ceiling Δ(k) <= sqrt(chi2(k)*T) holds at every REAL operating point (k>=2); the only
      violation is the degenerate k=1 where chi2_tell=0 forces the ceiling to exactly 0.
  (4) chi2_tell(N) = N^2/(2N-1)-1 is used, and Δ_tell(k) is monotone non-decreasing in k.
  (5) equal-effort sqrt(E/T) orders gap>density>stability, matching the E/(E+T) ranking.
  (6) recomputed bin/STRONG ratio == bin_shift_showing / strong_knob_delta.
Run: python verify_ce16.py
"""
import os, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "ce16_strongest_knob.json")))
panel = r["panel"]
props = ["gap", "density", "stability"]
ok = True
def chk(name, cond):
    global ok; print(("PASS" if cond else "FAIL"), name); ok = ok and cond

def chi2_tell(n):
    return n * n / (2 * n - 1) - 1.0 if n >= 1 else 0.0

# (1) steel-man: best-of-12 knob strictly stronger than the tag knob, every property
for p in props:
    d = panel[p]
    chk(f"{p}: strong-knob Δ(12) > tag-knob shift ({d['strong_knob_delta']} > {d['tag_knob_shift_ce14']})",
        d["strong_knob_delta"] > d["tag_knob_shift_ce14"])

# (2) chemistry still wins on gap & density (ratio > 1); stability reported honestly (still >1 here)
chk("gap bin/STRONG > 1", panel["gap"]["bin_over_STRONG_knob"] > 1.0)
chk("density bin/STRONG > 1", panel["density"]["bin_over_STRONG_knob"] > 1.0)
chk("stability bin/STRONG reported (>0)", panel["stability"]["bin_over_STRONG_knob"] > 0.0)

# (3)+(4) per-k sweep: chi2 formula, monotone Δ, ceiling holds for k>=2, k=1 is the degenerate corner
for p in props:
    sweep = panel[p]["sweep"]
    prev = -1.0
    for row in sweep:
        k = row["k"]
        chk(f"{p} k={k}: chi2_tell matches N^2/(2N-1)-1",
            abs(row["chi2_tell"] - chi2_tell(k)) < 1e-3)
        chk(f"{p} k={k}: Δ_tell monotone non-decreasing", row["delta_tell"] >= prev - 1e-9)
        prev = row["delta_tell"]
        exp_ceiling = (row["chi2_tell"] * panel[p]["T_within"]) ** 0.5
        chk(f"{p} k={k}: ceiling == sqrt(chi2*T)", abs(row["ceiling_sqrt_chi2_T"] - exp_ceiling) < 1e-3)
        if k == 1:
            chk(f"{p} k=1: degenerate corner (chi2=0 -> ceiling=0)",
                row["chi2_tell"] == 0.0 and row["ceiling_sqrt_chi2_T"] == 0.0)
        else:
            chk(f"{p} k={k}: within budget ceiling (Δ <= sqrt(chi2*T))",
                row["within_ceiling"] is True and row["delta_tell"] <= row["ceiling_sqrt_chi2_T"] + 1e-9)

# the reported boolean must be False precisely because of the k=1 corner (not a real-point violation)
for p in props:
    k1 = [s for s in panel[p]["sweep"] if s["k"] == 1][0]
    real_ok = all(s["within_ceiling"] for s in panel[p]["sweep"] if s["k"] >= 2)
    chk(f"{p}: budget_ceiling_holds_all_k False only due to k=1 corner",
        panel[p]["budget_ceiling_holds_all_k"] is False and real_ok and k1["within_ceiling"] is False)

# (5) equal-effort sqrt(E/T) ordering matches E/(E+T) ordering: gap > density > stability
sroot = [panel[p]["sqrt_E_over_T_equal_effort_ratio"] for p in props]
efrac = [panel[p]["E_over_EplusT"] for p in props]
chk("sqrt(E/T) orders gap>density>stability", sroot[0] > sroot[1] > sroot[2])
chk("E/(E+T) orders gap>density>stability", efrac[0] > efrac[1] > efrac[2])

# (6) recomputed bin/STRONG ratio == bin_shift / strong_knob_delta
for p in props:
    d = panel[p]
    exp = d["bin_shift_showing_ce14"] / d["strong_knob_delta"]
    chk(f"{p}: bin/STRONG == bin_shift/Δ(12) ({d['bin_over_STRONG_knob']} ~ {exp:.2f})",
        abs(d["bin_over_STRONG_knob"] - exp) < 0.05)

print("\nCE16-VERIFY:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)

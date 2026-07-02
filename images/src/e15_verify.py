"""E15 verify — checks E15's own claims from e15_key_selection.json (handoff convention).

Claims tested (pre-registered, commit 5bafc8b):
  C1  the realized bin/knob ratio RISES with data-side E/(E+T): pooled Spearman > 0.
  C2  the random-partition key is at the FLOOR: for every target, random_50 has the lowest
      (or within-noise of the lowest) E/(E+T) AND the lowest realized ratio.
Exit non-zero if a hard check fails.
"""
import json, os, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{_ROOT}/distributions/e15_key_selection.json"))
ok = True
print("== E15 verify ==")
rho = J.get("spearman_ratio_vs_efrac_pooled", float("nan"))
c1 = rho > 0
ok &= bool(c1)
print(f"  C1  pooled Spearman(ratio, E/(E+T)) = {rho:.3f}  (>0 ?) -> {c1}")
for t in J["targets"]:
    pts = [p for p in J["points"] if p["target"] == t]
    rand = [p for p in pts if p["key"] == "random_50"][0]
    lo_ef = min(p["efrac"] for p in pts); lo_r = min(p["ratio_bin_over_knob"] for p in pts)
    c2 = (abs(rand["efrac"] - lo_ef) < 1e-9) and (rand["ratio_bin_over_knob"] <= lo_r + 1e-6)
    ok &= bool(c2)
    order = " > ".join(f"{p['key'].replace('_','')}:{p['ratio_bin_over_knob']:.2f}"
                       for p in sorted(pts, key=lambda x: -x["efrac"]))
    print(f"  C2  {t:12s} random floor? {c2}   ratio by E/(E+T)-desc: {order}")
print("E15-VERIFY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

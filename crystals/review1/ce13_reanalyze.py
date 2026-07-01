#!/usr/bin/env python3
"""C-E13 REVISED metric (per 2026-06-29 revision): primary = NUMBER OF CORNERS COVERED (a corner is
'covered' if its batch-fraction's bootstrap 95% CI excludes 0), with min(fractions) as a secondary
density number. Plus the mandatory PREMISE GATE: full pairwise correlation matrix of the triple
(all must be ≲0). Re-analysis of the existing ce13_coverage.json run (no re-generation)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COV = json.load(open(os.path.join(HERE, "ce13_coverage.json")))
TS = json.load(open(os.path.join(HERE, "ce13_tripleselect.json")))
OUT = os.path.join(HERE, "ce13_revised.json")
rng = np.random.default_rng(0)


def frac_ci(arr):
    arr = np.array(arr, float); n = len(arr)
    if n == 0: return 0.0, [0.0, 0.0]
    boot = [arr[rng.integers(0, n, n)].mean() for _ in range(5000)]
    return float(arr.mean()), [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


# premise gate
C = TS["corr_matrix"]; trip = ["gap", "eps0", "bvs_gii"]
pairs = {f"{a}-{b}": C[a][b] for i, a in enumerate(trip) for b in trip[i+1:]}
all_neg = all(v < 0 for v in pairs.values())

res = {"premise_gate": {"triple": trip, "pairwise_corr": pairs, "all_negative": all_neg,
                        "verdict": "PASS" if all_neg else "FAIL — not a mutually anti-correlated triple; "
                        "eps0-bvs is positive (data ~2-D). Least-correlated available triple; contamination flagged."},
       "methods": {}}
for m in ["telling", "conditioning", "showing"]:
    sc = COV["methods"][m]["scatter"]
    corners = {}
    for c in ["gap", "eps", "bvs"]:
        f, ci = frac_ci([r["in_" + c] for r in sc])
        corners[c] = {"frac": round(f, 3), "ci95": ci, "covered": ci[0] > 0}
    n_cov = sum(c["covered"] for c in corners.values())
    res["methods"][m] = {"n_corners_covered": n_cov, "corners": corners,
                         "min_fraction_secondary": round(min(c["frac"] for c in corners.values()), 3)}

print("=== PREMISE GATE ===")
print(" pairwise:", res["premise_gate"]["pairwise_corr"], "| all_negative:", all_neg)
print(" verdict:", res["premise_gate"]["verdict"])
print("\n=== REVISED PRIMARY METRIC: number of corners covered ===")
for m in ["telling", "conditioning", "showing"]:
    r = res["methods"][m]
    cc = {c: (r["corners"][c]["frac"], r["corners"][c]["covered"]) for c in r["corners"]}
    print(f"  {m:13s} corners_covered={r['n_corners_covered']}  {cc}  min={r['min_fraction_secondary']}")
json.dump(res, open(OUT, "w"), indent=2)
print("\nCE13-REVISED-DONE")

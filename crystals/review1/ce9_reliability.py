#!/usr/bin/env python3
"""C-E9 (part 1, no GPU): disentangle "stability doesn't carry over" from "noisy stability label",
by restricting the carry-over r² to chemistries where the label is RELIABLE and seeing if r² rises.

Reliability signal (data-side, label-only): the training structures are real, curated low-hull
crystals, so a faithful scorer should place them near the hull. A chemistry is "reliable" when the
single-MACE e_above_hull of its DATA structures is near 0 and tight (|mean| and spread small) — i.e.
the scorer is not systematically mis-labeling that chemical system. We sort chemistries by data-side
reliability and recompute stability carry-over r² on the most-reliable subset (sweep the fraction
kept). Pure re-analysis of ce2_peritem.json.

Read: if r² stays ~0.49 on the cleanest-label chemistries, stability genuinely doesn't carry
(Claim 1 strengthens); if it rises a lot, the low value was partly label noise.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERITEM = os.path.join(os.path.dirname(HERE), "results", "ce2_peritem.json")
OUT = os.path.join(HERE, "ce9_reliability.json")
rng = np.random.default_rng(0)


def pool_by_chem(peritem, prop):
    gd, gm = {}, {}
    for seed in peritem:
        for b in peritem[seed].values():
            k = (b["elements"], b["natoms"])
            for r in b["data"]:
                if r.get(prop) is not None: gd.setdefault(k, []).append(r[prop])
            for r in b["model"]:
                if r.get(prop) is not None: gm.setdefault(k, []).append(r[prop])
    return gd, gm


def r2_ci(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None, None
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0: boot.append(np.corrcoef(x[idx], y[idx])[0, 1] ** 2)
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)] if boot else None
    return round(r2, 3), ci


def main():
    peritem = json.load(open(PERITEM))["peritem"]
    gd, gm = pool_by_chem(peritem, "e_above_hull")
    chems = [k for k in gd if k in gm and len(gd[k]) >= 3 and len(gm[k]) >= 3]
    # reliability score per chemistry: smaller = more reliable (data eah near 0 and tight)
    rel = {k: abs(np.mean(gd[k])) + np.std(gd[k]) for k in chems}
    order = sorted(chems, key=lambda k: rel[k])           # most reliable first
    dmean = {k: np.mean(gd[k]) for k in chems}; mmean = {k: np.mean(gm[k]) for k in chems}

    res = {"n_chem_total": len(chems), "baseline_r2_all": None, "sweep": []}
    xa = np.array([dmean[k] for k in chems]); ya = np.array([mmean[k] for k in chems])
    r2a, cia = r2_ci(xa, ya)
    res["baseline_r2_all"] = {"r2": r2a, "ci95": cia, "n": len(chems)}
    print(f"all chemistries (n={len(chems)}): stability r2={r2a} CI {cia}")
    for frac in [0.5, 0.67, 0.75, 0.9]:
        k = max(3, int(round(frac * len(order))))
        keep = order[:k]
        x = np.array([dmean[c] for c in keep]); y = np.array([mmean[c] for c in keep])
        r2, ci = r2_ci(x, y)
        res["sweep"].append({"frac_kept": frac, "n": k, "r2": r2, "ci95": ci,
                             "max_data_eah_in_subset": round(float(max(abs(dmean[c]) for c in keep)), 4)})
        print(f"  most-reliable {int(frac*100)}% (n={k}): stability r2={r2} CI {ci}")
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, "\nCE9-RELIABILITY-DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C-E8: key-sensitivity ablation. Re-key the saved C-E2 per-item data under 2-3 chemistry keys and
report how (a) the data-side within/between (T/E) split and (b) the carry-over ladder r² move, and
whether the qualitative ordering density > bonding(BVS) > stability is stable. Pure re-analysis of
ce2_peritem.json — NO GPU, NO new generation.

Keys:
  els_nat  : (element-set, atom-count)            [current C-E2 key]
  els_only : element-set                          [coarser: merge atom-counts]
  anion    : anion family (first of O,S,Se,Te,N,P,As,F,Cl,Br,I,H; else 'metallic')  [coarsest]
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERITEM = os.path.join(os.path.dirname(HERE), "results", "ce2_peritem.json")
OUT = os.path.join(HERE, "ce8_keys.json")
PROPS = ["density", "bvs_gii", "e_above_hull"]
ANIONS = ["O", "S", "Se", "Te", "N", "P", "As", "F", "Cl", "Br", "I", "H"]
rng = np.random.default_rng(0)


def keyfn(name, els, nat):
    if name == "els_nat":  return (els, nat)
    if name == "els_only": return els
    if name == "anion":
        a = [x for x in ANIONS if x in els.split()]
        return a[0] if a else "metallic"


def collect(peritem, key, prop):
    """pool data & model values of `prop` per group across seeds -> (group_data_lists, group_model_lists)."""
    gd, gm = {}, {}
    for seed in peritem:
        for b in peritem[seed].values():
            k = keyfn(key, b["elements"], b["natoms"])
            for r in b["data"]:
                if r.get(prop) is not None: gd.setdefault(k, []).append(r[prop])
            for r in b["model"]:
                if r.get(prop) is not None: gm.setdefault(k, []).append(r[prop])
    return gd, gm


def split_TE(groups):
    """data-side within (T = mean of per-group var) and between (E = var of per-group means)."""
    means = [np.mean(v) for v in groups.values() if len(v) >= 2]
    vars = [np.var(v) for v in groups.values() if len(v) >= 2]
    if len(means) < 2: return None
    return float(np.mean(vars)), float(np.var(means)), len(means)


def carryover_r2(gd, gm):
    ks = [k for k in gd if k in gm and len(gd[k]) >= 3 and len(gm[k]) >= 3]
    if len(ks) < 3: return None, len(ks), None
    x = np.array([np.mean(gd[k]) for k in ks]); y = np.array([np.mean(gm[k]) for k in ks])
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(ks), len(ks))
        xb, yb = x[idx], y[idx]
        if np.std(xb) > 0 and np.std(yb) > 0: boot.append(np.corrcoef(xb, yb)[0, 1] ** 2)
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)] if boot else None
    return round(r2, 3), len(ks), ci


def main():
    peritem = json.load(open(PERITEM))["peritem"]
    res = {"keys": {}}
    print(f"{'key':9s} {'#groups':>7s} | " + " | ".join(f"{p[:9]:>22s}" for p in PROPS))
    for key in ["els_nat", "els_only", "anion"]:
        res["keys"][key] = {"carryover_r2": {}, "data_TE_split": {}}
        cells = []
        ngroups = None
        for prop in PROPS:
            gd, gm = collect(peritem, key, prop)
            r2, n, ci = carryover_r2(gd, gm)
            te = split_TE(gd)
            res["keys"][key]["carryover_r2"][prop] = {"r2": r2, "n_groups": n, "ci95": ci}
            if te:
                T, E, nt = te
                res["keys"][key]["data_TE_split"][prop] = {"T_within": round(T, 5), "E_between": round(E, 5),
                                                           "between_frac": round(E / (T + E), 3) if (T + E) else None}
            ngroups = n
            cells.append(f"r2={r2} (n={n})")
        print(f"{key:9s} {str(ngroups):>7s} | " + " | ".join(f"{c:>22s}" for c in cells))
    # ordering stability
    print("\nordering density>BVS>stability per key:")
    for key in res["keys"]:
        r = res["keys"][key]["carryover_r2"]
        vals = [r[p]["r2"] for p in PROPS]
        ok = all(v is not None for v in vals) and vals[0] >= vals[1] >= vals[2]
        print(f"  {key:9s}: {vals}  -> {'STABLE' if ok else 'check'}")
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote", OUT, "\nCE8-KEYS-DONE")


if __name__ == "__main__":
    main()

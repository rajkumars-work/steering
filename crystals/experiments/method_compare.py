#!/usr/bin/env python3
"""Illustrative comparison of the two target-hitting methods at Combined target = 0.5:
  Method 1 (MIXING): blend HIGH (rare-earth) + LOW (H/Pt) at lambda=(0.5-0.14)/(0.85-0.14).
  Method 2 (NATURAL): the single chemistry bin (scan pool 20) that naturally sits at 0.5.

For each: 3 reps x n=100 (structures SAVED), then compare:
  Combined mean/std, mean e_above_hull, natoms dist, element histogram + diversity,
  capacity (unique (E,N), expected-unique vs n, duplication onset), chi^2 (push, over 64 bins),
  and a bond-valence proxy (BVAnalyzer success rate + mean |net BV charge|).

    python experiments/method_compare.py   # -> experiments/out/method_compare/comparison.json
"""
import os, json, random, time
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from pymatgen.io.ase import AseAtomsAdaptor
from _repo import CKPT, DIST, HIGH_J, LOW_J, VERSION, DEVICE, OUTDIR

OUT = os.path.join(OUTDIR, "method_compare"); os.makedirs(OUT + "/struct", exist_ok=True)
N, REPS, TARGET = 100, 3, 0.5
HI, LO = 0.85, 0.14
NAT_BIN = 20

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def loadpool(p):
    d = json.load(open(p)); t = d["tuples"] if isinstance(d, dict) and "tuples" in d else d
    return [(x["elements"], int(x["natoms"]), int(x["count"])) for x in t]

def cap_stats(pool):
    """capacity: expected #unique in n draws, and n where dup fraction crosses 5%/20%."""
    cnt = np.array([c for _, _, c in pool], float); p = cnt / cnt.sum(); U = len(pool)
    def exp_unique(n): return float((1 - (1 - p) ** n).sum())
    eu100 = exp_unique(N)
    def dup_frac(n): return 1 - exp_unique(n) / n
    n5 = next((n for n in range(2, 5000) if dup_frac(n) > 0.05), None)
    n20 = next((n for n in range(2, 5000) if dup_frac(n) > 0.20), None)
    eff = 1.0 / (p ** 2).sum()   # Hill N2 / inverse-Simpson effective # bins
    return {"unique_en": U, "eff_unique": round(eff, 1), "exp_unique_at_100": round(eu100, 1),
            "n_dup5pct": n5, "n_dup20pct": n20}

def main():
    tuples = loadpool(DIST)
    # deterministic 64-cluster scan (same as before) for the natural bin + chi^2 J_P
    elems = sorted({e for els, _, _ in tuples for e in els.split()}); eidx = {e: i for i, e in enumerate(elems)}
    X = np.zeros((len(tuples), len(elems) + 1))
    for i, (els, n, c) in enumerate(tuples):
        for e in els.split(): X[i, eidx[e]] = 1.0
        X[i, -1] = n / 40.0
    labels = KMeans(n_clusters=64, random_state=0, n_init=4).fit_predict(X)
    clusters = {}
    for t, lab in zip(tuples, labels): clusters.setdefault(int(lab), []).append(t)
    bin_size = {b: sum(c for _, _, c in v) for b, v in clusters.items()}; tot = sum(bin_size.values())
    Jp = {b: bin_size[b] / tot for b in clusters}

    HIGH, LOW, NAT = loadpool(HIGH_J), loadpool(LOW_J), clusters[NAT_BIN]
    lam = (TARGET - LO) / (HI - LO); k = round(lam * N)
    log(f"target {TARGET}: MIX lambda={lam:.3f} ({k} HIGH + {N-k} LOW) | NATURAL bin {NAT_BIN} ({len(NAT)} EN)")

    # chi^2(mu_P || J_P) over the 64 bins for each recipe
    def chi2(mu):  # mu: dict bin-> prob
        return sum(mu.get(b, 0) ** 2 / Jp[b] for b in clusters) - 1
    # MIX recipe weight per bin: HIGH bins (8,18,32,38,46,55) get lam split by size; LOW bin 60 gets 1-lam
    HIGH_BINS = [8, 18, 32, 38, 46, 55]; LOW_BINS = [60]
    hs = sum(bin_size[b] for b in HIGH_BINS); ls = sum(bin_size[b] for b in LOW_BINS)
    mu_mix = {**{b: lam * bin_size[b] / hs for b in HIGH_BINS}, **{b: (1 - lam) * bin_size[b] / ls for b in LOW_BINS}}
    mu_nat = {NAT_BIN: 1.0}
    chi2_mix, chi2_nat = chi2(mu_mix), chi2(mu_nat)

    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from lemat_genbench.metrics.sun_metric import SUNMetric
    from ase.io import write
    log("loading model + mace + SUNMetric ...")
    model, sp = load_model(CKPT, DEVICE); calc = load_stability_calc(device=DEVICE); sun = SUNMetric()

    def sample(pool, n, seed):
        rng = random.Random(seed); keys = [(e, nn) for e, nn, _ in pool]; w = [c for _, _, c in pool]
        return [f"{e} | {nn} | " for e, nn in rng.choices(keys, weights=w, k=n)]

    def bvs_metric(structs):
        from pymatgen.analysis.bond_valence import BVAnalyzer
        bva = BVAnalyzer(); ok = 0; nets = []
        for s in structs:
            try:
                val = bva.get_valences(s); ok += 1
                nets.append(abs(sum(v * n for v, n in zip(val, [1] * len(val)))))
            except Exception:
                pass
        return {"bv_assign_rate": round(ok / max(len(structs), 1), 3),
                "mean_abs_net_charge": round(float(np.mean(nets)), 3) if nets else None}

    def run_method(name, pool_sampler):
        combineds, eahs, natoms_all, elem_hist, uniq_en, all_structs = [], [], [], Counter(), set(), []
        for rep in range(REPS):
            prompts = pool_sampler(rep)
            uniq_en |= set(tuple(p.split("|")[:2]) for p in prompts)
            atoms = []
            for src in prompts:
                try:
                    g = generate_one(model, sp, src, "alex", VERSION, DEVICE, top_k=10)
                    if g is not None and getattr(g, "atoms", None) is not None: atoms.append(g.atoms)
                except Exception: pass
            structs = []
            for a in atoms:
                try: eah = compute_e_above_hull(a, calc=calc, timeout=120).get("e_above_hull")
                except Exception: eah = None
                s = AseAtomsAdaptor.get_structure(a)
                if eah is not None: s.properties["e_above_hull_mean"] = float(eah); eahs.append(float(eah))
                natoms_all.append(len(a));
                for el in set(a.get_chemical_symbols()): elem_hist[el] += 1
                structs.append(s); all_structs.append(s)
            m = sun.compute(structs).metrics
            combineds.append(round(m.get("combined_sun_msun_rate", 0), 4))
            log(f"  {name} rep{rep}: combined={combineds[-1]}")
        write(f"{OUT}/struct/{name}.extxyz", [AseAtomsAdaptor.get_atoms(s) for s in all_structs])
        return {"combined_reps": combineds, "combined_mean": round(float(np.mean(combineds)), 4),
                "combined_std": round(float(np.std(combineds)), 4),
                "mean_e_above_hull": round(float(np.mean(eahs)), 4) if eahs else None,
                "natoms_mean": round(float(np.mean(natoms_all)), 1), "natoms_p50": int(np.median(natoms_all)),
                "natoms_min": int(min(natoms_all)), "natoms_max": int(max(natoms_all)),
                "n_distinct_elements": len(elem_hist), "top_elements": elem_hist.most_common(10),
                "bvs": bvs_metric(all_structs)}

    res = {"target": TARGET,
           "MIX": {"lambda": round(lam, 3), "k_high": k, "chi2": round(chi2_mix, 3),
                   "capacity_high": cap_stats(HIGH), "capacity_low": cap_stats(LOW),
                   **run_method("mix", lambda rep: sample(HIGH, k, 100 + rep) + sample(LOW, N - k, 200 + rep))},
           "NATURAL": {"bin": NAT_BIN, "chi2": round(chi2_nat, 3), "capacity": cap_stats(NAT),
                       **run_method("natural", lambda rep: sample(NAT, N, 300 + rep))}}
    json.dump(res, open(f"{OUT}/comparison.json", "w"), indent=2)
    open(f"{OUT}/status.txt", "w").write(f"METHOD COMPARE DONE {time.strftime('%FT%TZ')}\n")
    log("ALL DONE")

if __name__ == "__main__":
    main()

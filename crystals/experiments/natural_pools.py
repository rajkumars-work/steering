#!/usr/bin/env python3
"""Method 2 (natural pools): for each 0.1 Combined target, validate the single chemistry
sub-pool that naturally sits closest to it (no mixing of extremes). Reconstructs the same
deterministic 64-cluster scan, takes the closest bin per target, re-scores it at n=100, and
saves each bin as a reusable pool JSON.

    python experiments/natural_pools.py   # -> experiments/out/natural_pools/{results.json,pools/}
"""
import os, json, random, time
import numpy as np
from sklearn.cluster import KMeans
from pymatgen.io.ase import AseAtomsAdaptor
from _repo import CKPT, DIST, VERSION, DEVICE, OUTDIR

OUT = os.path.join(OUTDIR, "natural_pools"); os.makedirs(OUT + "/pools", exist_ok=True)
N = 100
TARGET_POOL = {0.1: 60, 0.2: 1, 0.3: 37, 0.4: 53, 0.5: 20, 0.6: 22, 0.7: 6, 0.8: 15}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    tuples = [(d["elements"], int(d["natoms"]), int(d["count"])) for d in json.load(open(DIST))]
    elems = sorted({e for els, _, _ in tuples for e in els.split()})
    eidx = {e: i for i, e in enumerate(elems)}
    X = np.zeros((len(tuples), len(elems) + 1))
    for i, (els, n, c) in enumerate(tuples):
        for e in els.split():
            X[i, eidx[e]] = 1.0
        X[i, -1] = n / 40.0
    labels = KMeans(n_clusters=64, random_state=0, n_init=4).fit_predict(X)   # same as the scan
    clusters = {}
    for t, lab in zip(tuples, labels):
        clusters.setdefault(int(lab), []).append(t)

    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from lemat_genbench.metrics.sun_metric import SUNMetric
    log("loading model + mace + SUNMetric ...")
    model, sp = load_model(CKPT, DEVICE)
    calc = load_stability_calc(device=DEVICE)
    sun = SUNMetric()

    def sample(pool, n, seed):
        rng = random.Random(seed)
        keys = [(e, nn) for e, nn, _ in pool]
        w = [c for _, _, c in pool]
        return [f"{e} | {nn} | " for e, nn in rng.choices(keys, weights=w, k=n)]

    def gen_score(prompts):
        atoms = []
        for src in prompts:
            try:
                g = generate_one(model, sp, src, "alex", VERSION, DEVICE, top_k=10)
                if g is not None and getattr(g, "atoms", None) is not None:
                    atoms.append(g.atoms)
            except Exception:
                pass
        structs = []
        for a in atoms:
            try:
                eah = compute_e_above_hull(a, calc=calc, timeout=120).get("e_above_hull")
            except Exception:
                eah = None
            s = AseAtomsAdaptor.get_structure(a)
            if eah is not None:
                s.properties["e_above_hull_mean"] = float(eah)
            structs.append(s)
        if not structs:
            return {"n": 0, "combined": None}
        m = sun.compute(structs).metrics
        return {"n": len(structs), "combined": round(m.get("combined_sun_msun_rate", 0), 4),
                "sun": round(m.get("sun_rate", 0), 4), "msun": round(m.get("msun_rate", 0), 4)}

    def top_elems(pool, k=8):
        from collections import Counter
        c = Counter()
        for e, _, cnt in pool:
            for x in e.split():
                c[x] += cnt
        return [e for e, _ in c.most_common(k)]

    results = []
    for T in sorted(TARGET_POOL):
        lab = TARGET_POOL[T]
        pool = clusters[lab]
        r = gen_score(sample(pool, N, seed=4000 + lab))
        elems_top = top_elems(pool)
        results.append({"target": T, "bin": lab, "validated_combined": r.get("combined"),
                        "err": None if r.get("combined") is None else round(r["combined"] - T, 3),
                        "sun": r.get("sun"), "msun": r.get("msun"), "n_en": len(pool), "elems": elems_top})
        # save the bin as a reusable pool artifact
        json.dump({"description": f"natural pool for Combined target {T} (validated {r.get('combined')}, n={N})",
                   "target": T, "scan_bin": lab, "n_en_tuples": len(pool),
                   "tuples": [{"elements": e, "natoms": n, "count": c} for e, n, c in pool]},
                  open(f"{OUT}/pools/natural_T{int(T*10):02d}.json", "w"), indent=1)
        log(f"target {T}: bin {lab} -> validated {r.get('combined')} (err {results[-1]['err']}) [{','.join(elems_top[:4])}]")
        json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

    open(f"{OUT}/status.txt", "w").write(f"NATURAL POOLS DONE {time.strftime('%FT%TZ')}\n")
    log("ALL DONE")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find the highest- and lowest-Combined sub-pools WITHIN the an_lh (E,N) distribution.

Strategy: scan chemistry sub-pools, then greedily REFINE toward each extreme — re-cluster
the current best (or worst) pool, score, keep the best (worst) bin, repeat — pushing
Combined up (down) until the pool gets too small or it plateaus. Each final extreme is
re-scored at higher n (winner's-curse control). NO mixing/combining — extremes only.

This is the experiment that DISCOVERS the steering range (Combined ~0.03 -> 0.82) and the two
anchor distributions shipped in this repo. Loads model + MACE + SUNMetric ONCE, loops. Writes
incremental JSON. Detached, multi-hour (~4-5 GPU-h).

    python experiments/scan_extremes.py      # -> experiments/out/scan_extremes/{scan,HIGH,LOW,summary}.json
"""
import os, json, random, time
import numpy as np
from sklearn.cluster import KMeans
from pymatgen.io.ase import AseAtomsAdaptor
from _repo import CKPT, DIST, VERSION, DEVICE, OUTDIR

OUT = os.path.join(OUTDIR, "scan_extremes"); os.makedirs(OUT, exist_ok=True)
K_SCAN = 64        # sub-pools in the initial exhaustive scan
N_SCAN = 24        # prompts/pool in scan & refine rounds
REFINE_K = 6       # sub-bins per refinement round
N_VALID = 120      # prompts to validate each final extreme (tight headline number)
MAX_ROUNDS = 6     # refinement depth per extreme
MIN_POOL = 24      # allow small extreme pools (concentrate on niche chemistries)
START_TOPK = 3     # push starts from the union of the top-K / bottom-K scan pools (robust vs a fluke single pool)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    dist = json.load(open(DIST))
    all_tuples = [(d["elements"], int(d["natoms"]), int(d["count"])) for d in dist]
    log(f"loaded an_lh distribution: {len(all_tuples)} (E,N) tuples")

    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from lemat_genbench.metrics.sun_metric import SUNMetric
    log("loading model + mace + SUNMetric(LeMat-Bulk) ...")
    model, sp = load_model(CKPT, DEVICE)
    calc = load_stability_calc(device=DEVICE)
    sun = SUNMetric()

    def cluster(tuples, k):
        k = min(k, len(tuples))
        if k <= 1:
            return {0: tuples}
        elems = sorted({e for els, _, _ in tuples for e in els.split()})
        eidx = {e: i for i, e in enumerate(elems)}
        X = np.zeros((len(tuples), len(elems) + 1))
        for i, (els, n, c) in enumerate(tuples):
            for e in els.split():
                X[i, eidx[e]] = 1.0
            X[i, -1] = n / 40.0
        labels = KMeans(n_clusters=k, random_state=0, n_init=4).fit_predict(X)
        out = {}
        for t, lab in zip(tuples, labels):
            out.setdefault(int(lab), []).append(t)
        return out

    def sample(tuples, n, seed):
        rng = random.Random(seed)
        keys = [(e, nn) for e, nn, _ in tuples]
        w = [c for _, _, c in tuples]
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
        return {"gen": len(atoms), "n": len(structs), "stable": m.get("stable_count"),
                "meta": m.get("metastable_count"), "sun": round(m.get("sun_rate", 0), 4),
                "msun": round(m.get("msun_rate", 0), 4),
                "combined": round(m.get("combined_sun_msun_rate", 0), 4)}

    def top_elems(tuples, k=6):
        from collections import Counter
        c = Counter()
        for els, _, cnt in tuples:
            for e in els.split():
                c[e] += cnt
        return [e for e, _ in c.most_common(k)]

    # ---------- Stage 1: exhaustive scan ----------
    scan_cl = cluster(all_tuples, K_SCAN)
    scan = {}
    for lab in sorted(scan_cl):
        t = time.time()
        r = gen_score(sample(scan_cl[lab], N_SCAN, 1000 + lab))
        r["size"] = len(scan_cl[lab]); r["elems"] = top_elems(scan_cl[lab])
        scan[lab] = r
        log(f"scan pool {lab:2d} (size {len(scan_cl[lab]):5d}) combined={r.get('combined')} {','.join(r['elems'][:4])} {time.time()-t:.0f}s")
        json.dump(scan, open(f"{OUT}/scan.json", "w"), indent=2)
    valid = {k: v for k, v in scan.items() if v.get("combined") is not None}
    ranked = sorted(valid, key=lambda k: valid[k]["combined"])
    log(f"SCAN spread: min={valid[ranked[0]]['combined']} (pool {ranked[0]})  "
        f"max={valid[ranked[-1]]['combined']} (pool {ranked[-1]})  over {len(valid)} pools")

    # ---------- Stage 2: greedy refine toward each extreme ----------
    def push(start_tuples, direction):
        cur = start_tuples
        traj = [{"round": 0, "pool_size": len(cur)}]
        for rnd in range(1, MAX_ROUNDS + 1):
            if len(cur) < MIN_POOL:
                break
            k = min(REFINE_K, max(2, len(cur) // 60))
            cl = cluster(cur, k)
            sc = {}
            for lab in cl:
                sc[lab] = gen_score(sample(cl[lab], N_SCAN, 2000 + rnd * 100 + lab))
            v = {l: s for l, s in sc.items() if s.get("combined") is not None}
            if not v:
                break
            pick = (max if direction == "high" else min)(v, key=lambda l: v[l]["combined"])
            cur = cl[pick]
            traj.append({"round": rnd, "scan_combined": v[pick]["combined"],
                         "pool_size": len(cur), "elems": top_elems(cur)})
            log(f"  [{direction}] round {rnd}: combined={v[pick]['combined']} pool_size={len(cur)} {','.join(top_elems(cur)[:4])}")
            json.dump({"high" if direction == "high" else "low": traj},
                      open(f"{OUT}/push_{direction}.json", "w"), indent=2)
        final = gen_score(sample(cur, N_VALID, 9000))   # validate final extreme
        return {"trajectory": traj, "final_pool_size": len(cur),
                "final_combined_validated": final, "final_elems": top_elems(cur, 10)}

    # start each push from the UNION of the top-K / bottom-K scan pools (robust vs a single fluke pool)
    high_start = [t for l in ranked[-START_TOPK:] for t in scan_cl[l]]
    low_start = [t for l in ranked[:START_TOPK] for t in scan_cl[l]]
    log(f"HIGH start = union of pools {ranked[-START_TOPK:]} ({len(high_start)} tuples); "
        f"LOW start = union of pools {ranked[:START_TOPK]} ({len(low_start)} tuples)")

    log("=== pushing HIGH ===")
    high = push(high_start, "high")
    json.dump(high, open(f"{OUT}/HIGH.json", "w"), indent=2)
    log(f"HIGH final validated combined = {high['final_combined_validated'].get('combined')} (n={N_VALID})")

    log("=== pushing LOW ===")
    low = push(low_start, "low")
    json.dump(low, open(f"{OUT}/LOW.json", "w"), indent=2)
    log(f"LOW final validated combined = {low['final_combined_validated'].get('combined')} (n={N_VALID})")

    json.dump({
        "scan_min": valid[ranked[0]]["combined"], "scan_max": valid[ranked[-1]]["combined"],
        "high_pushed": high["final_combined_validated"].get("combined"),
        "low_pushed": low["final_combined_validated"].get("combined"),
        "high_elems": high["final_elems"], "low_elems": low["final_elems"],
    }, open(f"{OUT}/summary.json", "w"), indent=2)
    open(f"{OUT}/status.txt", "w").write(f"BINNING DONE {time.strftime('%FT%TZ')}\n")
    log("ALL DONE")

if __name__ == "__main__":
    main()

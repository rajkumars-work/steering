#!/usr/bin/env python3
"""C-E13: does frontier coverage scale with the NUMBER of properties (2 -> 3)? Extends C-E12.
Three properties, each a corner (high in that property): gap>=2.5 eV, eps0>=35, BVS-GII>=0.5.
coverage = min(fracA_gap, fracB_eps, fracC_bvs) — the batch must hold ALL three corner kinds.
Methods (N=120 x 3 seeds): telling(single richest gap chem) / conditioning(broad + bg-vhigh) /
showing(equal mix of the per-corner specialist chemistries).

HONEST CAVEAT (pre-registered): the data is ~2-D (gap is anti-correlated with a polarizable
cluster {eps0, density, BVS}); among available cheap scorers no third axis is anti-correlated with
BOTH gap and eps0 (BVS is the least-correlated-with-eps0 at +0.35). So the 3rd corner partially
overlaps the eps0 corner -> this n=3 result is a LOWER BOUND on the true N-scaling effect; a clean
test needs a formation-energy / bulk-modulus surrogate on a genuine 3rd axis (not available).
gap & eps0 are XGBoost composition surrogates; BVS-GII is structural (pymatgen). No MACE.
"""
import sys, os, json, time, csv, random, importlib.util, math
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
                "/home/ubuntu/packages/lemat-genbench/src"]
import numpy as np, xgboost as xgb

CKPT = "/opt/dlami/nvme/recast/train/mix_ep120_ckpt"; VERSION = "d15_binrho_k7"; DEVICE = "cuda"
TRAIN = os.environ.get("CUES_TRAIN_CSV", "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv")
PD = "/home/ubuntu/code/py/dielectric/pipeline/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce13_coverage.json")
N_C, SEEDS, MIN_MEMBERS, N_AUDIT, M_AUDIT = 120, [0, 1, 2], 5, 600, 6
GAP_HI, EPS_HI, BVS_HI = 2.5, 35.0, 0.5     # pre-registered corner thresholds (high in each property)

_sp = importlib.util.spec_from_file_location("e", "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
_e = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e); ccf = _e.compute_composition_features
from dielectric_data.reader import parse_target
_bg = xgb.Booster(); _bg.load_model(f"{PD}/xgb_composition_dft_band_gap.json")
_e0 = xgb.Booster(); _e0.load_model(f"{PD}/xgb_composition_dft_eps_0.json")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def bvsgii(a):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
        oxi = BVAnalyzer().get_oxi_state_decorated_structure(AseAtomsAdaptor.get_structure(a))
        dv = [calculate_bv_sum(s, oxi.get_neighbors(s, 4.0)) - s.specie.oxi_state for s in oxi if oxi.get_neighbors(s, 4.0)]
        return float(np.sqrt(np.mean(np.square(dv)))) if dv else None
    except Exception:
        return None

def gap_eps(a):
    try:
        f = ccf(a).reshape(1, -1)
        return max(0.0, float(_bg.predict(xgb.DMatrix(f))[0])), max(1.0, float(_e0.predict(xgb.DMatrix(f))[0]))
    except Exception:
        return None, None

def corners(g, e, b):
    return {"gap": g is not None and g >= GAP_HI, "eps": e is not None and e >= EPS_HI,
            "bvs": b is not None and b >= BVS_HI}

def wilson(k, n, z=1.96):
    if n == 0: return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    from eval.screening import load_model, generate_one
    bins = defaultdict(list)
    for row in csv.reader(open(TRAIN)):
        if not row or row[0] == "source": continue
        s = row[0].split("|")
        if len(s) >= 2: bins[(s[0].strip(), s[1].strip())].append(row)
    elig = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    audit = random.Random(7).sample(elig, min(N_AUDIT, len(elig)))
    log(f"auditing {len(audit)} chemistries for specialist pools ...")
    gap_pool, eps_pool, bvs_pool, gap_means = [], [], [], []
    for (els, nat) in audit:
        gs, es, bs = [], [], []
        for r in random.Random(7).sample(bins[(els, nat)], min(M_AUDIT, len(bins[(els, nat)]))):
            try: a = parse_target(r[1], VERSION, r[3])
            except Exception: a = None
            if a is None: continue
            g, e = gap_eps(a); b = bvsgii(a)
            if g is not None: gs.append(g); es.append(e)
            if b is not None: bs.append(b)
        if len(gs) >= 3:
            mg, me = float(np.mean(gs)), float(np.mean(es)); mb = float(np.mean(bs)) if bs else None
            cnt = len(bins[(els, nat)]); key = ((els, nat), cnt)     # uniform: (chem, count)
            if mg >= GAP_HI: gap_pool.append(key); gap_means.append(mg)
            if me >= EPS_HI: eps_pool.append(key)
            if mb is not None and mb >= BVS_HI: bvs_pool.append(key)
    log(f"specialist pools — gap:{len(gap_pool)} eps:{len(eps_pool)} bvs:{len(bvs_pool)}")
    telling_chem = gap_pool[int(np.argmax(gap_means))][0] if gap_pool else audit[0]   # single richest gap chem

    model, sp = load_model(CKPT, DEVICE)

    def prompts(method, seed):
        rng = random.Random(1300 + seed)
        if method == "telling":
            return [f"{telling_chem[0]} | {telling_chem[1]} | "] * N_C
        if method == "conditioning":
            ks = audit; w = [len(bins[k]) for k in ks]
            return [f"{e} | {n} | bg-vhigh " for (e, n) in rng.choices(ks, weights=w, k=N_C)]
        if method == "showing":
            def draw(pool, k):                       # pool entries are ((els,nat), cnt)
                if not pool or k <= 0: return []
                kk = [p[0] for p in pool]; ww = [p[1] for p in pool]
                return [f"{e} | {nn} | " for (e, nn) in rng.choices(kk, weights=ww, k=k)]
            third = N_C // 3
            return draw(gap_pool, third) + draw(eps_pool, third) + draw(bvs_pool, N_C - 2 * third)

    res = {"properties": ["gap", "eps0", "bvs_gii"], "thresholds": {"gap>=": GAP_HI, "eps0>=": EPS_HI, "bvs_gii>=": BVS_HI},
           "n_per_method": N_C, "seeds": SEEDS,
           "n_specialists": {"gap": len(gap_pool), "eps": len(eps_pool), "bvs": len(bvs_pool)}, "methods": {}}
    for method in ["telling", "conditioning", "showing"]:
        allrec, per_seed = [], []
        for seed in SEEDS:
            recs = []
            for src in prompts(method, seed):
                try:
                    o = generate_one(model, sp, src, "alex", VERSION, DEVICE, top_k=10)
                    a = getattr(o, "atoms", None)
                    if a is None: continue
                except Exception:
                    continue
                g, e = gap_eps(a); b = bvsgii(a)
                if g is None: continue
                c = corners(g, e, b)
                recs.append({"gap": round(g, 3), "eps0": round(e, 2), "bvs": round(b, 3) if b is not None else None,
                             "in_gap": c["gap"], "in_eps": c["eps"], "in_bvs": c["bvs"]})
            n = len(recs)
            fr = {k: sum(r["in_" + k] for r in recs) / n if n else 0 for k in ["gap", "eps", "bvs"]}
            cov = min(fr.values()) if n else 0
            per_seed.append({"seed": seed, "n": n, **{f"frac_{k}": round(fr[k], 3) for k in fr}, "coverage": round(cov, 3)})
            allrec += recs
            log(f"  {method} seed{seed}: gap={fr['gap']:.2f} eps={fr['eps']:.2f} bvs={fr['bvs']:.2f} coverage={cov:.3f} (n={n})")
        N = len(allrec)
        fr = {k: sum(r["in_" + k] for r in allrec) / N if N else 0 for k in ["gap", "eps", "bvs"]}
        cov = min(fr.values()) if N else 0
        rngb = np.random.default_rng(0); arr = {k: np.array([r["in_" + k] for r in allrec]) for k in fr}; boot = []
        for _ in range(4000):
            idx = rngb.integers(0, N, N)
            boot.append(min(arr[k][idx].mean() for k in fr))
        res["methods"][method] = {"pooled_n": N, **{f"frac_{k}": round(fr[k], 3) for k in fr}, "coverage": round(cov, 3),
                                  "coverage_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                                  "per_seed": per_seed, "scatter": allrec}
        json.dump(res, open(OUT, "w"), indent=2)
        log(f"  == {method}: coverage(3-prop)={cov:.3f} CI {res['methods'][method]['coverage_ci95']}")
    log("=== C-E13 3-property coverage = min(frac gap, eps, bvs) ===")
    for m in ["telling", "conditioning", "showing"]:
        c = res["methods"][m]; log(f"  {m:13s} coverage {c['coverage']} CI {c['coverage_ci95']} (gap {c['frac_gap']}/eps {c['frac_eps']}/bvs {c['frac_bvs']})")
    print("CE13-COVERAGE-DONE")


if __name__ == "__main__":
    main()

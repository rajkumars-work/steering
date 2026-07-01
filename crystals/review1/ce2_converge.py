#!/usr/bin/env python3
"""C-E2: converge the J->G carry-over ladder (density -> BVS-GII -> stability) with CIs.

Re-runs the difficulty-ladder audit at a larger sample size and over >=3 generation seeds, and
sweeps samples/chemistry so we can show the per-chemistry-agreement fit (r^2) stabilizing rather
than being noise-limited. Saves EVERY per-structure measurement so the ladder CIs (and the
Claim-4 density diagonal CI) can be bootstrapped.

Same measure() as claim1_survival_spectrum.py: one measure per property for data and model.
Output: review1/ce2_converge.json (incremental) + review1/ce2_peritem.json.
Detached, multi-hour (MACE per structure).
"""
import sys, os, json, time, csv, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import CKPT, VERSION, DEVICE, require_train_csv, OUTDIR   # repo-relative; also sets sys.path
import numpy as np

TRAIN_CSV = require_train_csv()
HERE      = OUTDIR
OUT       = os.path.join(HERE, "ce2_converge.json")
PERITEM   = os.path.join(HERE, "ce2_peritem.json")

SEEDS       = [0, 1, 2]
K_BINS      = 24          # chemistries (fixed set across seeds, chosen with rng(0))
MIN_MEMBERS = 15          # data structures available per chemistry
M_DATA      = 15
M_GEN       = 30          # generated per (chemistry, seed); convergence read off at M_GRID
M_GRID      = [5, 10, 20, 30]
EAH_MAX     = 5.0
AMU         = 1.6605390666
CONTINUOUS  = ["density", "bvs_gii", "e_above_hull"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_ase(s):
    if s is None: return None
    if hasattr(s, "get_volume"): return s
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor.get_atoms(s)
    except Exception:
        return None


def bvs_gii(a):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
        struct = a if hasattr(a, "sites") else AseAtomsAdaptor.get_structure(a)
        oxi = BVAnalyzer().get_oxi_state_decorated_structure(struct)
        devs = []
        for site in oxi:
            nn = oxi.get_neighbors(site, 4.0)
            if nn:
                devs.append(calculate_bv_sum(site, nn) - site.specie.oxi_state)
        return float(np.sqrt(np.mean(np.square(devs)))) if devs else None
    except Exception:
        return None


def measure(s, calc, check_validity, compute_eah):
    out = dict.fromkeys(CONTINUOUS, None)
    a = to_ase(s)
    if a is None: return out
    try:
        v = float(a.get_volume())
        if v > 0: out["density"] = float(np.sum(a.get_masses())) / v * AMU
    except Exception: pass
    out["bvs_gii"] = bvs_gii(a)
    try:
        e = compute_eah(a, calc=calc, timeout=120).get("e_above_hull")
        if e is not None and np.isfinite(e) and abs(e) <= EAH_MAX:
            out["e_above_hull"] = float(e)
    except Exception: pass
    return out


def row_to_atoms(row):
    from dielectric_data.reader import parse_target
    try: return parse_target(row[1], VERSION, row[3])
    except Exception: return None


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main():
    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from chem.validity import check_validity

    bins = defaultdict(list)
    with open(TRAIN_CSV) as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            if len(row) < 6: continue
            src = row[0].split("|")
            if len(src) >= 2: bins[(src[0].strip(), src[1].strip())].append(row)
    eligible = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    chosen = random.Random(0).sample(eligible, min(K_BINS, len(eligible)))   # fixed chemistry set
    log(f"{len(eligible)} eligible bins (>= {MIN_MEMBERS}); testing {len(chosen)} chemistries x {len(SEEDS)} seeds")

    model, sp = load_model(CKPT, DEVICE)
    calc = load_stability_calc(device=DEVICE)

    # per-item store: peritem[seed][chem_idx] = {"data": [{prop:val}...], "model": [...]}
    peritem = {}
    for seed in SEEDS:
        peritem[seed] = {}
        rng = random.Random(1000 + seed)
        for ci_, (els, nat) in enumerate(chosen):
            prompt = f"{els} | {nat} | "
            rows = rng.sample(bins[(els, nat)], min(M_DATA, len(bins[(els, nat)])))
            data = [measure(row_to_atoms(r), calc, check_validity, compute_e_above_hull) for r in rows]
            modl = []
            for _ in range(M_GEN):
                try:
                    o = generate_one(model, sp, prompt, "alex", VERSION, DEVICE, top_k=10)
                    a = getattr(o, "atoms", None)
                    if a is not None:
                        modl.append(measure(a, calc, check_validity, compute_e_above_hull))
                except Exception:
                    pass
            peritem[seed][ci_] = {"elements": els, "natoms": nat, "data": data, "model": modl}
            log(f"  seed {seed} [{ci_+1}/{len(chosen)}] {els}|{nat}: data {len(data)} model {len(modl)}")
            json.dump({"chosen": chosen, "peritem": peritem}, open(PERITEM, "w"))

    # ---- analysis: convergence sweep + converged ladder with CIs ----
    def per_chem_means(seed, prop, m_gen):
        xs, ys = [], []
        for ci_ in peritem[seed]:
            dvals = [r[prop] for r in peritem[seed][ci_]["data"] if r.get(prop) is not None]
            mvals = [r[prop] for r in peritem[seed][ci_]["model"][:m_gen] if r.get(prop) is not None]
            if len(dvals) >= 3 and len(mvals) >= 3:
                xs.append(np.mean(dvals)); ys.append(np.mean(mvals))
        return np.array(xs), np.array(ys)

    res = {"ckpt": CKPT, "seeds": SEEDS, "k_chem": len(chosen), "M_data": M_DATA, "M_gen": M_GEN,
           "M_grid": M_GRID, "convergence": {}, "converged_ladder": {}}
    rngb = np.random.default_rng(0)
    for prop in CONTINUOUS:
        res["convergence"][prop] = {}
        for m in M_GRID:
            seed_r2 = [r2(*per_chem_means(s, prop, m)) for s in SEEDS]
            seed_r2 = [v for v in seed_r2 if not np.isnan(v)]
            res["convergence"][prop][m] = {"seed_mean": round(float(np.mean(seed_r2)), 4),
                                           "seed_std": round(float(np.std(seed_r2)), 4),
                                           "per_seed": [round(v, 4) for v in seed_r2]}
        # converged (M_GEN) ladder CI: bootstrap over chemistries, pooled across seeds
        boot = []
        for _ in range(5000):
            s = rngb.choice(SEEDS)
            x, y = per_chem_means(s, prop, M_GEN)
            if len(x) >= 3:
                idx = rngb.integers(0, len(x), len(x))
                boot.append(r2(x[idx], y[idx]))
        boot = [b for b in boot if not np.isnan(b)]
        pt = float(np.mean([r2(*per_chem_means(s, prop, M_GEN)) for s in SEEDS]))
        res["converged_ladder"][prop] = {"r2_point": round(pt, 4),
                                         "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                                  round(float(np.percentile(boot, 97.5)), 4)]}
    json.dump(res, open(OUT, "w"), indent=2)
    log("=== converged ladder (r^2, M_gen=%d, %d seeds) ===" % (M_GEN, len(SEEDS)))
    for prop in CONTINUOUS:
        c = res["converged_ladder"][prop]
        log(f"  {prop:14s} r2={c['r2_point']:.3f}  CI95 {c['ci95']}")
    print("CE2-CONVERGE-DONE")


if __name__ == "__main__":
    main()

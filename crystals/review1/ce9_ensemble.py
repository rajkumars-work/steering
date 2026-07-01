#!/usr/bin/env python3
"""C-E9 (part 2, GPU): lower-noise stability label via an MLIP ENSEMBLE, then recompute the
stability carry-over r². Disentangles "stability doesn't carry" from "single-MACE label is noisy".

Same 24 chemistries as C-E2 (seed 0 set), regenerate 15 data + 30 model structures per chemistry,
and score e_above_hull with an ENSEMBLE of MACE foundation models (each its own self-consistent
hull); the ensemble label = MEDIAN over models (robust to a single model's divergences). We score
EVERY structure with the single (default) MACE too, so single-vs-ensemble r² is compared on the
SAME structures (no confound). Report stability carry-over r² ± bootstrap CI: single vs ensemble,
vs the C-E2 value 0.49.

NOTE: the ensemble is MACE-family (partially correlated errors), so it cuts single-model noise only
partially; the orb+mace+uma 3-MLIP stack (in .venv_genbench) would decorrelate further — flagged.
Output: review1/ce9_ensemble.json (incremental). Detached, multi-hour.
"""
import sys, os, json, time, csv, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import CKPT, VERSION, DEVICE, require_train_csv, OUTDIR   # repo-relative; also sets sys.path
import numpy as np

TRAIN_CSV = require_train_csv()
HERE = OUTDIR
OUT = os.path.join(HERE, "ce9_ensemble.json")
K_BINS, MIN_MEMBERS, M_DATA, M_GEN, SEED = 24, 15, 15, 30, 0
EAH_MAX = 5.0
MACE_VARIANTS = ["medium", "small"]   # extra foundation models beyond the default 0b3-medium


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from mace.calculators import mace_mp

    log("loading generator + MACE ensemble ...")
    model, sp = load_model(CKPT, DEVICE)
    calcs = [("mace0b3", load_stability_calc(device=DEVICE))]              # the C-E2 default
    for v in MACE_VARIANTS:
        try:
            calcs.append((f"mace_{v}", mace_mp(model=v, device=DEVICE, default_dtype="float32")))
            log(f"  loaded ensemble member mace_{v}")
        except Exception as e:
            log(f"  WARN could not load mace_{v}: {e}")
    log(f"ensemble has {len(calcs)} MACE members: {[n for n,_ in calcs]}")

    def eah_each(atoms):
        """list of finite e_above_hull from each model (|e|<=EAH_MAX)."""
        out = []
        for _, c in calcs:
            try:
                e = compute_e_above_hull(atoms, calc=c, timeout=120).get("e_above_hull")
                out.append(e if (e is not None and np.isfinite(e) and abs(e) <= EAH_MAX) else None)
            except Exception:
                out.append(None)
        return out

    bins = defaultdict(list)
    with open(TRAIN_CSV) as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            if len(row) < 6: continue
            s = row[0].split("|")
            if len(s) >= 2: bins[(s[0].strip(), s[1].strip())].append(row)
    eligible = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    chosen = random.Random(SEED).sample(eligible, min(K_BINS, len(eligible)))   # same set as C-E2

    from dielectric_data.reader import parse_target
    def row_to_atoms(r):
        try: return parse_target(r[1], VERSION, r[3])
        except Exception: return None

    def med(vals):
        v = [x for x in vals if x is not None]
        return float(np.median(v)) if v else None

    per = {}   # chem -> {data:[{single,ens}], model:[...]}
    rng = random.Random(1000)
    for bi, (els, nat) in enumerate(chosen):
        prompt = f"{els} | {nat} | "
        rows = rng.sample(bins[(els, nat)], min(M_DATA, len(bins[(els, nat)])))
        drec, mrec = [], []
        for r in rows:
            a = row_to_atoms(r)
            if a is None: continue
            ev = eah_each(a)
            if ev[0] is not None or med(ev) is not None:
                drec.append({"single": ev[0], "ens": med(ev)})
        for _ in range(M_GEN):
            try:
                o = generate_one(model, sp, prompt, "alex", VERSION, DEVICE, top_k=10)
                a = getattr(o, "atoms", None)
                if a is None: continue
                ev = eah_each(a)
                if ev[0] is not None or med(ev) is not None:
                    mrec.append({"single": ev[0], "ens": med(ev)})
            except Exception:
                pass
        per[f"{els}|{nat}"] = {"data": drec, "model": mrec}
        log(f"  [{bi+1}/{len(chosen)}] {els}|{nat}: data {len(drec)} model {len(mrec)}")
        json.dump({"members": [n for n, _ in calcs], "per": per}, open(OUT, "w"))

    # carry-over r2: per-chem mean of (single) and (ens); compare on the SAME chemistries
    def r2_ci(which):
        xs, ys = [], []
        for k, v in per.items():
            d = [r[which] for r in v["data"] if r[which] is not None]
            m = [r[which] for r in v["model"] if r[which] is not None]
            if len(d) >= 3 and len(m) >= 3:
                xs.append(np.mean(d)); ys.append(np.mean(m))
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 3: return {"n": int(len(xs)), "r2": None}
        r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
        rngb = np.random.default_rng(0); boot = []
        for _ in range(5000):
            idx = rngb.integers(0, len(xs), len(xs))
            if np.std(xs[idx]) > 0 and np.std(ys[idx]) > 0: boot.append(np.corrcoef(xs[idx], ys[idx])[0, 1] ** 2)
        return {"n": int(len(xs)), "r2": round(r2, 3),
                "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}

    res = {"members": [n for n, _ in calcs], "n_chem": len(per),
           "stability_carryover": {"single_mace": r2_ci("single"), "ensemble_median": r2_ci("ens")},
           "per": per}
    json.dump(res, open(OUT, "w"), indent=2)
    log("=== stability carry-over r^2 (same structures) ===")
    log(f"  single-MACE : {res['stability_carryover']['single_mace']}")
    log(f"  ensemble    : {res['stability_carryover']['ensemble_median']}")
    print("CE9-ENSEMBLE-DONE")


if __name__ == "__main__":
    main()

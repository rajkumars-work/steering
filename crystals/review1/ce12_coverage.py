#!/usr/bin/env python3
"""C-E12: coverage of an anti-correlated trade-off frontier (band gap ^ dielectric eps_0).
The real Claim-3 crystal experiment: can SHOWING (mixing chemistries) cover BOTH corners of the
frontier, where telling/conditioning (a single request) reaches only one?

Corners (pre-registered, ce12_prediction.md):
  A = gap>=3.0 eV AND eps0<=25  (high-gap, low-dielectric)
  B = gap<=1.0 eV AND eps0>=33  (low-gap, high-dielectric)
Methods (N=120 x 3 seeds): telling(single richest A chem) / conditioning(broad + bg-vhigh tag) /
showing(50/50 mix of A-chems + B-chems). Metric: coverage = min(fracA, fracB); also fracMid + a
2-D (gap,eps0) scatter (structures saved per method). Gap & eps0 are XGBoost composition surrogates
(CPU) — NO MACE — so this is generation-bound. Detached.
"""
import sys, os, json, time, csv, random, importlib.util, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import CKPT, VERSION, DEVICE, require_train_csv, BG_MODEL, EPS0_SURROGATE, EPS0_PATH  # repo-relative
import numpy as np, xgboost as xgb

TRAIN = require_train_csv()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce12_coverage.json")
N_C, SEEDS = 120, [0, 1, 2]
MIN_MEMBERS, N_AUDIT, M_AUDIT = 5, 600, 6   # >=12 gives only 61 chems (corner-A too sparse); >=5 -> 1576
# pre-registered corners
A_GAP, A_EPS = 3.0, 25.0      # high gap, low diel
B_GAP, B_EPS = 1.0, 33.0      # low gap, high diel

_sp = importlib.util.spec_from_file_location("e", EPS0_PATH)
_e = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e); ccf = _e.compute_composition_features
from dielectric_data.reader import parse_target
_bg = xgb.Booster(); _bg.load_model(BG_MODEL)
_ep = xgb.Booster(); _ep.load_model(EPS0_SURROGATE)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def props(a):
    try:
        f = ccf(a).reshape(1, -1)
        return max(0.0, float(_bg.predict(xgb.DMatrix(f))[0])), max(1.0, float(_ep.predict(xgb.DMatrix(f))[0]))
    except Exception:
        return None, None

def corner(g, e, ag=A_GAP, ae=A_EPS, bg=B_GAP, be=B_EPS):
    if g is None or e is None: return "skip"
    if g >= ag and e <= ae: return "A"
    if g <= bg and e >= be: return "B"
    return "mid"

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
    log(f"auditing {len(audit)} chemistries for corner pools ...")
    chem_mean, A_pool, B_pool = {}, [], []
    for (els, nat) in audit:
        gs, es = [], []
        for r in random.Random(7).sample(bins[(els, nat)], min(M_AUDIT, len(bins[(els, nat)]))):
            try: a = parse_target(r[1], VERSION, r[3])
            except Exception: a = None
            if a is None: continue
            g, e = props(a)
            if g is not None: gs.append(g); es.append(e)
        if len(gs) >= 3:
            mg, me = float(np.mean(gs)), float(np.mean(es)); chem_mean[(els, nat)] = (mg, me)
            cnt = len(bins[(els, nat)])
            if mg >= A_GAP and me <= A_EPS: A_pool.append(((els, nat), cnt, mg, me))
            if mg <= B_GAP and me >= B_EPS: B_pool.append(((els, nat), cnt, mg, me))
    log(f"corner-A chems: {len(A_pool)} | corner-B chems: {len(B_pool)} | audited-with-props: {len(chem_mean)}")
    if not A_pool or not B_pool:
        log("WARN a corner pool is empty — coverage undefined; widen thresholds.")
    telling_chem = max(A_pool, key=lambda x: x[2])[0] if A_pool else audit[0]   # single richest corner-A
    log(f"telling single chemistry = {telling_chem}")

    model, sp = load_model(CKPT, DEVICE)

    def sample_prompts(method, seed):
        rng = random.Random(900 + seed)
        if method == "telling":
            return [f"{telling_chem[0]} | {telling_chem[1]} | "] * N_C
        if method == "conditioning":
            ks = [k for k in audit]; w = [len(bins[k]) for k in ks]
            return [f"{e} | {n} | bg-vhigh " for (e, n) in rng.choices(ks, weights=w, k=N_C)]
        if method == "showing":
            def draw(pool, k):
                ks = [p[0] for p in pool]; w = [p[1] for p in pool]
                return [f"{e} | {nn} | " for (e, nn) in rng.choices(ks, weights=w, k=k)]
            return draw(A_pool, N_C // 2) + draw(B_pool, N_C - N_C // 2)

    res = {"pair": "band_gap ^ eps_0", "corner_A": [A_GAP, A_EPS], "corner_B": [B_GAP, B_EPS],
           "n_per_method": N_C, "seeds": SEEDS, "n_corner_A_chems": len(A_pool), "n_corner_B_chems": len(B_pool),
           "methods": {}}
    for method in ["telling", "conditioning", "showing"]:
        per_seed, allrec = [], []
        for seed in SEEDS:
            cnt = {"A": 0, "B": 0, "mid": 0}; recs = []
            for src in sample_prompts(method, seed):
                try:
                    o = generate_one(model, sp, src, "alex", VERSION, DEVICE, top_k=10)
                    a = getattr(o, "atoms", None)
                    if a is None: continue
                except Exception:
                    continue
                g, e = props(a)
                if g is None: continue
                c = corner(g, e); cnt[c] = cnt.get(c, 0) + 1
                recs.append({"gap": round(g, 3), "eps0": round(e, 2), "corner": c})
            n = cnt["A"] + cnt["B"] + cnt["mid"]
            fa, fb = (cnt["A"] / n if n else 0), (cnt["B"] / n if n else 0)
            per_seed.append({"seed": seed, "n": n, "fracA": round(fa, 3), "fracB": round(fb, 3),
                             "fracMid": round(cnt["mid"] / n if n else 0, 3), "coverage": round(min(fa, fb), 3)})
            allrec += recs
            log(f"  {method} seed{seed}: A={fa:.2f} B={fb:.2f} mid={cnt['mid']/max(n,1):.2f} coverage={min(fa,fb):.3f} (n={n})")
        # pooled
        A = sum(r["corner"] == "A" for r in allrec); B = sum(r["corner"] == "B" for r in allrec); N = len(allrec)
        cov = min(A / N, B / N) if N else 0
        # bootstrap CI on coverage over structures
        rngb = np.random.default_rng(0); cors = np.array([r["corner"] for r in allrec]); boot = []
        for _ in range(4000):
            idx = rngb.integers(0, N, N); s = cors[idx]
            boot.append(min((s == "A").mean(), (s == "B").mean()))
        res["methods"][method] = {"pooled_n": N, "fracA": round(A / N, 3), "fracB": round(B / N, 3),
                                  "fracMid": round(sum(r["corner"] == "mid" for r in allrec) / N, 3),
                                  "coverage": round(cov, 3),
                                  "coverage_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                                  "fracA_wilson": wilson(A, N), "fracB_wilson": wilson(B, N),
                                  "per_seed": per_seed, "scatter": allrec}
        json.dump(res, open(OUT, "w"), indent=2)
        log(f"  == {method}: coverage={cov:.3f} CI {res['methods'][method]['coverage_ci95']} (A={A/N:.2f} B={B/N:.2f})")
    log("=== C-E12 coverage = min(fracA, fracB) ===")
    for m in ["telling", "conditioning", "showing"]:
        c = res["methods"][m]; log(f"  {m:13s} coverage {c['coverage']} CI {c['coverage_ci95']}  (A {c['fracA']} / B {c['fracB']} / mid {c['fracMid']})")
    print("CE12-COVERAGE-DONE")


if __name__ == "__main__":
    main()

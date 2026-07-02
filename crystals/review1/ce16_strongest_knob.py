#!/usr/bin/env python3
"""C-E16 — strongest / audit-calibrated knob on the bins-vs-knobs panel.

Give TELLING its strongest within-chemistry shot with audit parity: best-of-N test-time selection by
the property scorer, at FIXED chemistry (so it can only exploit the within-chemistry budget T, never the
between-chemistry E). Analytic telling effort chi2_tell(N) = N^2/(2N-1) - 1 (max-order-statistic
reweighting). Sweep N, measure Delta_tell(N), verify Delta_tell <= sqrt(chi2_tell * T), report the
equal-effort floor sqrt(E/T), and recompute the bin/knob ratio vs the STRONGEST knob (best-of-M).

One tag-free candidate pool per chemistry, generated once, scored for all three properties, so best-of-k
is an order statistic over the pool. gap = composition-XGBoost surrogate (eV); density = geometry (g/cc);
stability = single-MACE metastable rate (fraction e_above_hull <= 0.1), matching the C-E14 stability
re-run. Showing side (bin_shift) reused from C-E14. Output: review1/ce16_strongest_knob.json. Detached.
"""
import sys, os, json, time, csv, random, importlib.util
from collections import defaultdict
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
                "/home/ubuntu/packages/lemat-genbench/src"]
import numpy as np, xgboost as xgb

CKPT = "/opt/dlami/nvme/recast/train/mix_ep120_ckpt"; VERSION = "d15_binrho_k7"; DEVICE = "cuda"
TRAIN = os.environ.get("CUES_TRAIN_CSV", "/opt/dlami/nvme/recast/train/alex_nolemat_lowhull_dataset_train.csv")
PD = "/home/ubuntu/code/py/dielectric/pipeline/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ce16_strongest_knob.json")
SEEDS = [int(x) for x in os.environ.get("CE16_SEEDS", "0,1,2").split(",")]
N_BROAD = int(os.environ.get("CE16_NBROAD", "30"))   # chemistries per seed
M = int(os.environ.get("CE16_M", "12"))              # candidates per chemistry
MIN_MEMBERS, N_AUDIT = 5, 500
KSWEEP = [k for k in (1, 2, 4, 8, 12) if k <= M]
EAH_MAX, META_THR = 5.0, 0.1
AMU = 1.6605390666
BOOT = 5000
# showing side reused from C-E14 (steering/crystals/results/ce14_panel.json)
BIN_SHIFT = {"gap": 2.7093, "density": 8.4037, "stability": 0.3849}
BIN_ET = {"gap": 0.979, "density": 0.955, "stability": 0.573}

_sp = importlib.util.spec_from_file_location("e", "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
_e = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_e); ccf = _e.compute_composition_features
_bg = xgb.Booster(); _bg.load_model(f"{PD}/xgb_composition_dft_band_gap.json")
_calc = {"c": None}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def gap_val(a):
    try: return max(0.0, float(_bg.predict(xgb.DMatrix(ccf(a).reshape(1, -1)))[0]))
    except Exception: return None

def density_val(a):
    try:
        v = float(a.get_volume()); return float(np.sum(a.get_masses())) / v * AMU if v > 0 else None
    except Exception: return None

def eah_val(a):
    try:
        from chem.stability import compute_e_above_hull
        e = compute_e_above_hull(a, calc=_calc["c"], timeout=120).get("e_above_hull")
        return e if (e is not None and np.isfinite(e) and abs(e) <= EAH_MAX) else None
    except Exception: return None

def chi2_tell(N):
    return N * N / (2 * N - 1) - 1.0

def bestofk_stat(vals, k, maximize, rng, draws=400):
    """Mean over random k-subsets of the subset extremum (max if maximize else min)."""
    v = np.asarray(vals, float); m = len(v)
    if m == 0: return None
    if k >= m: return float(v.max() if maximize else v.min())
    idx = rng.integers(0, m, size=(draws, k))          # with-replacement approx of k-subset extremum
    sub = v[idx]; ext = sub.max(1) if maximize else sub.min(1)
    return float(ext.mean())

def bestofk_meta(eahs, k, rng, draws=400):
    """metastable rate of best-of-k selected by MIN e_above_hull."""
    e = np.asarray(eahs, float); m = len(e)
    if m == 0: return None
    if k >= m: return float(e.min() <= META_THR)
    idx = rng.integers(0, m, size=(draws, k)); sub = e[idx]
    sel = sub.min(1)                                    # selected candidate's eah
    return float((sel <= META_THR).mean())


def main():
    from eval.screening import load_model, generate_one
    from chem.stability import load_stability_calc

    bins = defaultdict(list)
    for row in csv.reader(open(TRAIN)):
        if not row or row[0] == "source": continue
        s = row[0].split("|")
        if len(s) >= 2: bins[(s[0].strip(), s[1].strip())].append(row)
    elig = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    audit = random.Random(7).sample(elig, min(N_AUDIT, len(elig)))
    log(f"eligible chems {len(elig)}; audit pool {len(audit)}; seeds {SEEDS} x {N_BROAD} chems x M={M}")

    # ---- load model FIRST, then MACE (dtype footgun is fixed at source, but keep the safe order) ----
    log("loading model, then single-MACE ...")
    model, sp = load_model(CKPT, DEVICE)
    _calc["c"] = load_stability_calc(device=DEVICE)

    # preflight: scorer + generator
    from ase.build import bulk
    from chem.stability import compute_e_above_hull
    nacl = compute_e_above_hull(bulk("NaCl", "rocksalt", a=5.64), calc=_calc["c"], timeout=120).get("e_above_hull")
    log(f"preflight scorer: NaCl e_above_hull={nacl}")
    pf = 0
    for (e, n) in random.Random(3).choices(audit, k=6):
        o = generate_one(model, sp, f"{e} | {n} | ", "alex", VERSION, DEVICE, top_k=10)
        pf += getattr(o, "atoms", None) is not None
    log(f"preflight generator: {pf}/6 atoms")
    if pf == 0:
        print("CE16-PREFLIGHT-FAIL"); sys.exit(75)

    # ---- generate the tag-free candidate pools; score all three properties ----
    # pools[seed][chem] = list of (gap, density, eah) for realized candidates
    pools = {name: defaultdict(list) for name in ("gap", "density", "stability")}
    for seed in SEEDS:
        rng = random.Random(1600 + seed)
        chems = rng.choices(audit, weights=[len(bins[k]) for k in audit], k=N_BROAD)
        for ci, (els, nat) in enumerate(chems):
            key = (seed, ci, els, nat)                 # unique per draw (a chem can recur)
            for _ in range(M):
                try:
                    o = generate_one(model, sp, f"{els} | {nat} | ", "alex", VERSION, DEVICE, top_k=10)
                    a = getattr(o, "atoms", None)
                except Exception:
                    a = None
                if a is None: continue
                g, d, e = gap_val(a), density_val(a), eah_val(a)
                if g is not None: pools["gap"][key].append(g)
                if d is not None: pools["density"][key].append(d)
                if e is not None: pools["stability"][key].append(e)
        n_g = sum(len(v) for v in pools["gap"].values())
        log(f"  seed{seed}: pooled candidates gap={n_g} "
            f"dens={sum(len(v) for v in pools['density'].values())} "
            f"stab={sum(len(v) for v in pools['stability'].values())}")

    # ---- per-property budget + best-of-k sweep ----
    rng = np.random.default_rng(0)
    res = {"design": {"seeds": SEEDS, "n_broad": N_BROAD, "M": M, "ksweep": KSWEEP,
                      "chi2_formula": "N^2/(2N-1)-1", "strong_knob": "best-of-N within-chemistry"},
           "panel": {}}
    for name in ("gap", "density", "stability"):
        chem_vals = [v for v in pools[name].values() if len(v) >= 2]     # per-chemistry candidate lists
        allv = np.concatenate([np.asarray(v, float) for v in chem_vals])
        is_stab = name == "stability"
        maximize = not is_stab                                          # stability: minimize eah
        # T (within-chem) and E (between-chem) in the C-E14 metric
        if is_stab:
            p_c = [float((np.asarray(v) <= META_THR).mean()) for v in chem_vals]
            T = float(np.mean([p * (1 - p) for p in p_c]))
            E = float(np.var(p_c))
            base = float((allv <= META_THR).mean())
        else:
            T = float(np.mean([np.var(v) for v in chem_vals]))
            E = float(np.var([np.mean(v) for v in chem_vals]))
            base = float(allv.mean())
        et = E / (E + T) if (E + T) else None
        sqrt_e_over_t = float(np.sqrt(E / T)) if T > 0 else None

        # best-of-k: per chemistry stat, averaged over chemistries; bootstrap CI over chemistries
        sweep = []
        for k in KSWEEP:
            per_chem = []
            for v in chem_vals:
                s = bestofk_meta(v, k, rng) if is_stab else bestofk_stat(v, k, maximize, rng)
                if s is not None: per_chem.append(s)
            per_chem = np.asarray(per_chem, float)
            delta = abs(float(per_chem.mean()) - base)
            # bootstrap over chemistries
            bs = [abs(per_chem[rng.integers(0, len(per_chem), len(per_chem))].mean() - base) for _ in range(BOOT)]
            ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
            c2 = chi2_tell(k); ceil = float(np.sqrt(c2 * T)) if c2 > 0 else 0.0
            sweep.append({"k": k, "chi2_tell": round(c2, 4), "delta_tell": round(delta, 4),
                          "delta_ci95": ci, "ceiling_sqrt_chi2_T": round(ceil, 4),
                          "within_ceiling": bool(delta <= ceil + 1e-9)})
        strong = sweep[-1]                                              # best-of-M
        bin_shift = BIN_SHIFT[name]
        ratio_strong = round(bin_shift / strong["delta_tell"], 2) if strong["delta_tell"] > 1e-9 else None
        res["panel"][name] = {
            "base_mean": round(base, 4), "T_within": round(T, 6), "E_between": round(E, 6),
            "E_over_EplusT": round(et, 3) if et is not None else None,
            "E_over_EplusT_ce14": BIN_ET[name],
            "sqrt_E_over_T_equal_effort_ratio": round(sqrt_e_over_t, 2) if sqrt_e_over_t else None,
            "bin_shift_showing_ce14": bin_shift,
            "tag_knob_shift_ce14": None,       # filled from ce14_panel below if available
            "strong_knob_delta": strong["delta_tell"], "strong_knob_chi2": strong["chi2_tell"],
            "bin_over_STRONG_knob": ratio_strong,
            "budget_ceiling_holds_all_k": all(s["within_ceiling"] for s in sweep),
            "sweep": sweep}
        log(f"  == {name}: E/(E+T)={et:.3f} sqrt(E/T)={sqrt_e_over_t:.2f} | "
            f"strong-knob Delta(k={M})={strong['delta_tell']} chi2={strong['chi2_tell']} "
            f"bin/STRONG={ratio_strong} | ceiling holds={res['panel'][name]['budget_ceiling_holds_all_k']}")
        json.dump(res, open(OUT, "w"), indent=2)

    # attach the C-E14 tag ("weak") knob_shift for reference
    try:
        p14 = json.load(open(os.path.join(HERE, "ce14_panel.json")))["panel"]
        for name in res["panel"]:
            res["panel"][name]["tag_knob_shift_ce14"] = p14[name]["knob_shift"]
    except Exception as ex:
        log(f"(could not read ce14_panel for tag knob: {ex})")
    json.dump(res, open(OUT, "w"), indent=2)

    log("=== C-E16 summary: strongest within-chem knob vs chemistry selection ===")
    for name in ("gap", "density", "stability"):
        r = res["panel"][name]
        log(f"  {name:10s} E/(E+T)={r['E_over_EplusT']}  sqrt(E/T)={r['sqrt_E_over_T_equal_effort_ratio']}  "
            f"tag-knob={r['tag_knob_shift_ce14']}  strong-knob Delta={r['strong_knob_delta']}  "
            f"bin/STRONG={r['bin_over_STRONG_knob']}  ceiling_holds={r['budget_ceiling_holds_all_k']}")
    print("CE16-DONE")


if __name__ == "__main__":
    main()

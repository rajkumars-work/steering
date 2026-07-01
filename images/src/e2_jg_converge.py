"""E2 — converge the J->G fit (review P4, P5).

claims234.py fit Claim 4.1 at m=8 ("noise-limited"). Here we re-run the per-class
J->G bridge at m>=50 over the 50-class spread (classes = range(0,1000,20)), and:
  - report per-target Pearson r2, MAD, mean offset delta, and Spearman rho,
    each with a bootstrap-over-classes 95% CI;
  - report realization rho = sqrt(mean_b T(G)_b / mean_b v_b);
  - emit an m-sweep curve (r2 at m' = 4,8,16,32,50) showing the fit tightening
    as per-class generated means de-noise.

Generate m images per class ONCE, then subsample the first m' for the m-sweep so
the curve is apples-to-apples (same images, more averaging). Bootstrap over the
50 classes is the CI that matters for the fit (the fit is a 50-point regression).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
CLASSES = list(range(0, 1000, 20))   # 50-class spread
M = 50                               # images per class
MSWEEP = [4, 8, 16, 32, 50]
CFG = 1.0                            # raw conditional, matches Claim 4.1


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_stats(x, y):
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {"r2": r * r, "r": r, "rho_spearman": spearman(x, y),
            "MAD": float(np.mean(np.abs(x - y))), "delta_offset": float(np.mean(y - x))}


def boot_ci_classes(x, y, fn, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x); vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(x[idx], y[idx]))
        except Exception:
            pass
    vals = np.asarray(vals, float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    g = {t: d[f"g_{t}"] for t in Labeler.TARGETS}
    v = {t: d[f"v_{t}"] for t in Labeler.TARGETS}
    pipe = G.load_pipe(); L = Labeler()

    # per-class labels: lab_by_class[c][target] = array length M
    lab_by_class = {}
    for c in CLASSES:
        imgs = G.generate(pipe, [c] * M, guidance_scale=CFG, seed=1000 + c)
        lab_by_class[c] = {t: np.asarray(L.labels(imgs)[t], float) for t in Labeler.TARGETS}
        print(f"class {c} done ({len(imgs)} imgs)", flush=True)

    res = {"classes": CLASSES, "m": M, "cfg": CFG, "msweep": MSWEEP, "targets": {}}
    audit_g = {t: np.array([g[t][c] for c in CLASSES]) for t in Labeler.TARGETS}

    for t in Labeler.TARGETS:
        genmean = np.array([lab_by_class[c][t][:M].mean() for c in CLASSES])
        x, y = audit_g[t], genmean
        stats = fit_stats(x, y)
        # bootstrap CIs over classes
        r2_lo, r2_hi = boot_ci_classes(x, y, lambda a, b: fit_stats(a, b)["r2"], seed=1)
        mad_lo, mad_hi = boot_ci_classes(x, y, lambda a, b: float(np.mean(np.abs(a - b))), seed=2)
        d_lo, d_hi = boot_ci_classes(x, y, lambda a, b: float(np.mean(b - a)), seed=3)
        rho_lo, rho_hi = boot_ci_classes(x, y, lambda a, b: spearman(a, b), seed=4)
        # m-sweep r2
        sweep = {}
        for mp in MSWEEP:
            ym = np.array([lab_by_class[c][t][:mp].mean() for c in CLASSES])
            rr = float(np.corrcoef(x, ym)[0, 1]) if x.std() > 0 and ym.std() > 0 else float("nan")
            sweep[mp] = rr * rr
        # realization rho
        TG = float(np.mean([lab_by_class[c][t][:M].var() for c in CLASSES]))
        Td = float(np.mean([v[t][c] for c in CLASSES]))
        res["targets"][t] = {
            **stats,
            "r2_CI": [r2_lo, r2_hi], "MAD_CI": [mad_lo, mad_hi],
            "delta_CI": [d_lo, d_hi], "rho_spearman_CI": [rho_lo, rho_hi],
            "msweep_r2": sweep,
            "realization_rho": (TG / Td) ** 0.5 if Td > 0 else float("nan"),
            "T_G": TG, "T_data": Td}
        print(f"{t:16s} r2={stats['r2']:.3f} CI[{r2_lo:.3f},{r2_hi:.3f}]  "
              f"MAD={stats['MAD']:.4g}  delta={stats['delta_offset']:+.4g}  "
              f"rho={stats['rho_spearman']:.3f}  msweep={ {k: round(v,3) for k,v in sweep.items()} }",
              flush=True)

    json.dump(res, open(f"{OUT}/e2_jg_converge.json", "w"), indent=2)
    print("E2-JG-CONVERGE-DONE")


if __name__ == "__main__":
    main()

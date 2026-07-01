"""E11 REVISION — primary metric = NUMBER OF CORNERS COVERED (round-5 revision).

The first E11 used coverage = min(corner fractions), which is dominated by the scarcest corner, so
showing's number "fell" 0.30->0.013 going 2->3 corners even though the structural gap widened. The
main session asked to switch the headline to:

  number of corners covered = count of corners whose fraction's 95% CI (bootstrap over images)
  excludes 0, at a fixed per-corner tertile threshold.

Expected: telling = 1, compositional = 0, showing = N -> structural gap (showing - telling) = N-1
WIDENS with N by construction. Keep min(fractions) as a secondary density number only.

Pure post-processing of the per-image scores already saved in the coverage JSONs (no generation):
  P=2  <- out/e10_coverage_vehicle_food.json   (keys 'animal'=vehicle, 'nature'=food)
  P=3  <- out/e11_nscale_coverage_vehicle_food_brightness.json
"""
import json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
HI_PCT, LO_PCT = 66.667, 33.333


def corner_frac(scores, axes, focus, thi, tlo, idx=None):
    """fraction in the corner 'focus' = high on focus AND low on all other axes."""
    if idx is None:
        idx = np.arange(len(scores[axes[0]]))
    m = scores[focus][idx] >= thi[focus]
    for a in axes:
        if a != focus:
            m = m & (scores[a][idx] <= tlo[a])
    return float(m.mean())


def boot_ci(scores, axes, focus, thi, tlo, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores[axes[0]])
    vals = [corner_frac(scores, axes, focus, thi, tlo, rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyze(path, axes_order, label):
    d = json.load(open(path))
    methods = list(d["methods"].keys())
    sc = {m: {a: np.asarray(d["scores"][m][a], float) for a in axes_order} for m in methods}
    pooled = {a: np.concatenate([sc[m][a] for m in methods]) for a in axes_order}
    thi = {a: float(np.percentile(pooled[a], HI_PCT)) for a in axes_order}
    tlo = {a: float(np.percentile(pooled[a], LO_PCT)) for a in axes_order}
    out = {"label": label, "P": len(axes_order), "axes": axes_order, "methods": {}}
    print(f"\n=== {label}  (P={len(axes_order)}, corners={axes_order}) ===")
    for m in methods:
        per_corner = {}
        ncov = 0
        for a in axes_order:
            fr = corner_frac(sc[m], axes_order, a, thi, tlo)
            lo, hi = boot_ci(sc[m], axes_order, a, thi, tlo, seed=1)
            covered = lo > 0.0
            ncov += int(covered)
            per_corner[a] = {"frac": fr, "CI95": [lo, hi], "covered": covered}
        mn = min(per_corner[a]["frac"] for a in axes_order)
        out["methods"][m] = {"n_corners_covered": ncov, "per_corner": per_corner, "min_fraction": mn}
        print(f"  {m:14s} corners_covered={ncov}/{len(axes_order)}  min_frac={mn:.3f}  " +
              "  ".join(f"{a.split('_')[-1]}={per_corner[a]['frac']:.3f}"
                        f"[{per_corner[a]['CI95'][0]:.3f},{per_corner[a]['CI95'][1]:.3f}]{'*' if per_corner[a]['covered'] else ''}"
                        for a in axes_order))
    return out


def main():
    res = {"metric": "n_corners_covered (bootstrap CI excludes 0, tertile thresholds)", "runs": {}}
    # P=2: E10b vehicle/food (scores stored under generic keys 'animal'(=vehicle) / 'nature'(=food))
    p2 = f"{OUT}/e10_coverage_vehicle_food.json"
    if os.path.exists(p2):
        res["runs"]["P2_vehicle_food"] = analyze(p2, ["animal", "nature"], "P=2 vehicle/food (E10b)")
    # P=3: E11b vehicle/food/brightness
    p3 = f"{OUT}/e11_nscale_coverage_vehicle_food_brightness.json"
    if os.path.exists(p3):
        d3 = json.load(open(p3))
        res["runs"]["P3_vehicle_food_brightness"] = analyze(p3, d3["axes"], "P=3 vehicle/food/brightness (E11b)")

    # structural gap summary
    print("\n=== structural gap: n_corners_covered ===")
    print(f"{'run':30s} {'telling':>8s} {'compositional':>14s} {'showing':>8s} {'gap(show-tell)':>15s}")
    for k, r in res["runs"].items():
        m = r["methods"]
        def nc(name):
            for key in m:
                if name in key:
                    return m[key]["n_corners_covered"]
            return None
        t, c, s = nc("telling"), nc("composition"), nc("showing")
        gap = (s - t) if (s is not None and t is not None) else None
        print(f"{k:30s} {str(t):>8s} {str(c):>14s} {str(s):>8s} {str(gap):>15s}")

    json.dump(res, open(f"{OUT}/e11_corners_metric.json", "w"), indent=2)
    print("\nE11-CORNERS-METRIC-DONE")


if __name__ == "__main__":
    main()

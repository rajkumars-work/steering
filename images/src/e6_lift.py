"""E6 — lift predicts which joints are reachable by showing (win Claim 3; surfaces B.6).

For several candidate joints (animal&nature, animal&food, food&nature, vehicle&nature,
animal&bright, food&bright), compute each joint's *lift* from the shadow alone (no scorer):
rank classes by min(z_A, z_B), take the top-30 "joint-rich" bins, and score the joint by how
far that pool sits above average on BOTH axes — lift = mean_{top30} min(z_A, z_B). Rank the
joints by this lift-implied reachability. THEN generate the showing pool (top-30 bins) for each
joint and measure the realized joint hit-rate (hit = A>=gbar_A AND B>=gbar_B, both "above
average"). If lift predicts reachability, the lift ranking and the realized-hit ranking agree
(Spearman rho), and high-lift joints clear a much higher hit-rate than low-lift ones.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
N = 160
CFG = 4.0
JOINTS = [("sim_animal", "sim_nature"), ("sim_animal", "sim_food"),
          ("sim_food", "sim_nature"), ("sim_vehicle", "sim_nature"),
          ("sim_animal", "brightness"), ("sim_food", "brightness")]


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def boot_ci(mask, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(mask, float)
    m = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    g = {t: d[f"g_{t}"] for t in Labeler.TARGETS}
    gbar = {t: float((d["count"] / d["count"].sum() * g[t]).sum()) for t in Labeler.TARGETS}
    z = {t: zscore(g[t]) for t in Labeler.TARGETS}

    pipe = G.load_pipe(); L = Labeler()
    rows = []
    for A, B in JOINTS:
        joint_classes = np.argsort(np.minimum(z[A], z[B]))[::-1][:30]
        lift = float(np.mean(np.minimum(z[A], z[B])[joint_classes]))     # shadow-only lift score
        ids = G.sample_mu(G.uniform_pool(joint_classes), N, seed=21)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=21))
        a = np.asarray(l[A], float); b = np.asarray(l[B], float)
        hit = (a >= gbar[A]) & (b >= gbar[B])
        hit_pct = float(hit.mean()) * 100
        lo, hi = boot_ci(hit, seed=3)
        rows.append({"joint": f"{A}&{B}", "lift_score": lift,
                     "showing_classes": sorted(joint_classes.tolist()),
                     "realized_hit_pct": hit_pct, "hit_CI95": [lo, hi],
                     "mean_A": float(a.mean()), "mean_B": float(b.mean()),
                     "gbar_A": gbar[A], "gbar_B": gbar[B]})
        print(f"{A}&{B:14s} lift={lift:+.3f}  hit={hit_pct:5.1f}% CI[{lo:.1f},{hi:.1f}]", flush=True)

    lifts = [r["lift_score"] for r in rows]
    hits = [r["realized_hit_pct"] for r in rows]
    rho = spearman(np.array(lifts), np.array(hits))
    pear = float(np.corrcoef(lifts, hits)[0, 1])
    res = {"N": N, "cfg": CFG, "joints": rows,
           "spearman_lift_vs_hit": rho, "pearson_lift_vs_hit": pear,
           "ranking_by_lift": [r["joint"] for r in sorted(rows, key=lambda r: -r["lift_score"])],
           "ranking_by_hit": [r["joint"] for r in sorted(rows, key=lambda r: -r["realized_hit_pct"])]}
    print(f"\nSpearman(lift, hit) = {rho:.3f} ; Pearson = {pear:.3f}")
    print("rank by lift:", res["ranking_by_lift"])
    print("rank by hit :", res["ranking_by_hit"])
    json.dump(res, open(f"{OUT}/e6_lift.json", "w"), indent=2)
    print("E6-LIFT-DONE")


if __name__ == "__main__":
    main()

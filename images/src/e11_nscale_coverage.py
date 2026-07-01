"""E11 — coverage scales with the number of properties (the N-scaling test). ROUND 5, TOP PRIORITY.

Theory now states the multi-property case: covering a Pareto front of P properties needs a mixture
of SPECIALIST bins, unless one bin's within-bin spread already reaches every corner — and this
tilts toward showing as P grows. E10 gave the P=2 evidence (vehicle/food). E11 extends to P=3
mutually anti-correlated concepts and reports the *trend* 2->3.

Axes: vehicle / food / nature (pairwise corr -0.135 / +0.178 / +0.095; the most anti-correlated
3-concept set, corners are clean: V=654 F=959 N=972, each high on self ~0.20 and low on the others
~0.10-0.14). A genuinely all-negative alternative is vehicle/food/brightness (-0.135/-0.061/-0.116)
— noted as a flag; we use the 3 semantic concepts for interpretability.

Clean corner per property = high on that one AND low on BOTH others:
  corner_X = score_X >= tau_hi(X) AND score_Y <= tau_lo(Y) AND score_Z <= tau_lo(Z)
coverage = min over the P corner-fractions (you only score if the batch holds ALL corner kinds).

Methods (matched N, >=5 seeds, CIs):
  telling        single best corner bin (cleanest vehicle); also report its variance-reach
                 (fraction of its own batch landing in each corner -> the mechanism)
  compositional  Composable Diffusion of the P property conditionals (generate_composed_multi)
  showing        equal mix of the P specialist corner bins

PRE-REGISTERED PREDICTION: a single bin may cover <=1 clean corner here (its within-bin spread
does not reach the other corners) -> coverage 0; compositional collapses to the middle -> coverage
0; showing covers all P corners -> coverage > 0. The single-bin/compositional gap vs showing WIDENS
from P=2 to P=3 (read alongside E10b). Report whatever happens.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G
from compose import generate_composed_multi

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
# axes via CLI: python e11_nscale_coverage.py sim_vehicle sim_food brightness ; default v/f/nature
AXES = sys.argv[1:] if len(sys.argv) > 1 else ["sim_vehicle", "sim_food", "sim_nature"]
SUF = "_" + "_".join(a.replace("sim_", "") for a in AXES)
N = 128
CFG = 4.0
COMPOSE_SCALE = 4.0
SEEDS = [11, 23, 37, 53, 67]
HI_PCT, LO_PCT = 66.667, 33.333
SENS = [(75.0, 25.0), (60.0, 40.0)]


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def corner_masks(scores, thi, tlo):
    """scores: dict axis->array. thi/tlo: dict axis->threshold. Returns dict axis->bool mask
    (high on that axis AND low on all others), plus a 'middle' mask."""
    P = list(scores.keys())
    masks = {}
    for a in P:
        m = scores[a] >= thi[a]
        for b in P:
            if b != a:
                m = m & (scores[b] <= tlo[b])
        masks[a] = m
    any_corner = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        any_corner = any_corner | m
    masks["middle"] = ~any_corner
    return masks


def coverage_from(scores, thi, tlo, axes):
    m = corner_masks(scores, thi, tlo)
    fr = {a: float(m[a].mean()) for a in axes}
    return min(fr.values()), fr, float(m["middle"].mean())


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def boot_cov_ci(scores, thi, tlo, axes, reps=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores[axes[0]]); vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        s = {a: scores[a][idx] for a in axes}
        vals.append(coverage_from(s, thi, tlo, axes)[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    g = {a: d[f"g_{a}"] for a in AXES}
    z = {a: zscore(g[a]) for a in AXES}
    # cleanest corner bin per axis: high self, low both others
    corner_bin = {}
    for a in AXES:
        others = [o for o in AXES if o != a]
        corner_bin[a] = int(np.argmax(z[a] - 0.5 * sum(z[o] for o in others)))
    print("corner bins:", corner_bin, flush=True)
    specialist = [corner_bin[a] for a in AXES]

    pipe = G.load_pipe(); L = Labeler()

    def gen_std(classes, seed):
        ids = G.sample_mu(G.uniform_pool(classes), N, seed=seed)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=seed))
        return {a: np.asarray(l[a], float) for a in AXES}

    methods = ["telling", "compositional", "showing"]
    data = {m: {"per_seed": []} for m in methods}
    for s in SEEDS:
        data["telling"]["per_seed"].append(gen_std([corner_bin[AXES[0]]], s))            # single best vehicle bin
        data["showing"]["per_seed"].append(gen_std(specialist, s + 1))                   # equal mix of P specialists
        cls_lists = [[corner_bin[a]] * N for a in AXES]
        l = L.labels(generate_composed_multi(pipe, cls_lists, scale=COMPOSE_SCALE, seed=s + 2))
        data["compositional"]["per_seed"].append({a: np.asarray(l[a], float) for a in AXES})
        print(f"seed {s} done", flush=True)

    for m in methods:
        data[m]["scores"] = {a: np.concatenate([ps[a] for ps in data[m]["per_seed"]]) for a in AXES}

    # pre-registered thresholds: per-axis tertiles of the pooled batch (all methods)
    pooled = {a: np.concatenate([data[m]["scores"][a] for m in methods]) for a in AXES}

    def thr(hi, lo):
        return ({a: float(np.percentile(pooled[a], hi)) for a in AXES},
                {a: float(np.percentile(pooled[a], lo)) for a in AXES})

    thi, tlo = thr(HI_PCT, LO_PCT)
    res = {"P": len(AXES), "axes": AXES, "N": N, "cfg": CFG, "seeds": SEEDS,
           "corner_bins": corner_bin,
           "pairwise_corr": {f"{a}|{b}": float(np.corrcoef(g[a], g[b])[0, 1])
                             for i, a in enumerate(AXES) for b in AXES[i+1:]},
           "thresholds": {"hi": thi, "lo": tlo, "hi_pct": HI_PCT, "lo_pct": LO_PCT},
           "methods": {}, "sensitivity": {}, "scores": {}}
    print(f"\n{'method':14s} " + " ".join(f"f_{a.split('_')[-1]:>8s}" for a in AXES) + f" {'mid':>7s} {'coverage':>9s}  CI")
    for m in methods:
        sc = data[m]["scores"]
        cov, fr, mid = coverage_from(sc, thi, tlo, AXES)
        per_seed = [coverage_from(ps, thi, tlo, AXES)[0] for ps in data[m]["per_seed"]]
        cm, clo, chi = across_seed_ci(per_seed)
        blo, bhi = boot_cov_ci(sc, thi, tlo, AXES, seed=7)
        res["methods"][m] = {"corner_fracs": fr, "frac_middle": mid, "coverage": cov,
                             "coverage_per_seed": per_seed, "coverage_across_seed_CI95": [clo, chi],
                             "coverage_bootstrap_CI95": [blo, bhi]}
        res["scores"][m] = {a: sc[a].tolist() for a in AXES}
        print(f"{m:14s} " + " ".join(f"{fr[a]:9.3f}" for a in AXES) +
              f" {mid:7.3f} {cov:9.3f}  seed[{clo:.3f},{chi:.3f}] boot[{blo:.3f},{bhi:.3f}]", flush=True)

    # variance-reach of the single best bin: corner fractions of telling's own batch
    res["telling_variance_reach"] = res["methods"]["telling"]["corner_fracs"]
    print("variance-reach (telling single-bin corner fracs):", res["telling_variance_reach"])

    for hi, lo in SENS:
        th, tl = thr(hi, lo)
        res["sensitivity"][f"{hi:.0f}/{lo:.0f}"] = {m: coverage_from(data[m]["scores"], th, tl, AXES)[0]
                                                    for m in methods}
        print(f"sensitivity {hi:.0f}/{lo:.0f}: " +
              " ".join(f"{m}={coverage_from(data[m]['scores'],th,tl,AXES)[0]:.3f}" for m in methods))

    json.dump(res, open(f"{OUT}/e11_nscale_coverage{SUF}.json", "w"), indent=2)

    # figure: pairwise 2-D projections (3 axis pairs) x 3 methods
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for r, m in enumerate(methods):
        sc = data[m]["scores"]
        for c, (i, j) in enumerate(pairs):
            ax = axes[r][c]
            ax.scatter(sc[AXES[i]], sc[AXES[j]], s=6, alpha=0.4, c="0.5")
            ax.axvline(thi[AXES[i]], ls="--", c="k", lw=0.5); ax.axhline(thi[AXES[j]], ls="--", c="k", lw=0.5)
            ax.set_xlabel(AXES[i].split("_")[-1]); ax.set_ylabel(AXES[j].split("_")[-1])
            if c == 0:
                ax.set_title(f"{m}  (coverage={res['methods'][m]['coverage']:.2f})", loc="left", fontsize=11)
    fig.suptitle("E11 — 3-property coverage (vehicle/food/nature): pairwise projections per method", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig_e11_nscale_coverage{SUF}.png", dpi=120)
    print("saved figure -> fig_e11_nscale_coverage{SUF}.png")
    print("E11-NSCALE-DONE")


if __name__ == "__main__":
    main()

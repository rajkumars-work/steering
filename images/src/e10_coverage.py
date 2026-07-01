"""E10 — coverage of a DISJUNCTION (the real Claim 3). ROUND 4, TOP PRIORITY.

The round-1/E9 "animal AND nature" joint was the wrong test: a conjunction is satisfiable by a
single output (an otter is both), so one bin wins it — that is a Claim-2 concentration result,
not Claim 3. Showing's UNIQUE power is expressing a DISJUNCTION as a spread across a batch:
"give me both A and B" = a batch that covers both clean corners, which no single bin can make,
and which compositional guidance (the practitioner's tool) cannot either — our hypothesis is it
collapses to the mushy MIDDLE (half-animal-half-nature chimeras), covering neither clean corner.

Two CLIP axes: animal, nature. Clean corners (raw CLIP scores; thresholds pre-registered as
per-axis tertiles of the POOLED batch over all methods, sensitivity reported):
  corner A = animal >= tau_hi(animal) AND nature <= tau_lo(nature)   (clean animal, not nature)
  corner B = nature >= tau_hi(nature) AND animal <= tau_lo(animal)   (clean nature, not animal)
  middle   = everything else (both-high otters AND mushy chimeras)

Methods (matched N, >=5 seeds, CIs):
  1. telling_best   point mass on the single audit-cleanest corner-A class (expect ~all in A -> cov 0)
  2. compositional  Composable Diffusion conjunction of a clean-A and a clean-B class (expect middle)
  3. showing_2bin   50% clean-A class + 50% clean-B class (expect clean A's AND B's -> high cov)
  4. showing_broad  uniform over top-5 A + top-5 B classes (a broader spread)

PRIMARY METRIC coverage = min(frac in A, frac in B). Also frac in middle and the A/B/middle
breakdown. Killer figure: 2-D (animal, nature) scatter, one panel per method.

PRE-REGISTERED PREDICTION: telling -> one corner (cov 0); compositional -> middle (covers neither,
cov ~0); showing -> both corners (cov high). We report whatever happens.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G
from compose import generate_composed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
N = 128
CFG = 4.0
COMPOSE_SCALE = 4.0
SEEDS = [11, 23, 37, 53, 67]               # 5 seeds
HI_PCT, LO_PCT = 66.667, 33.333            # pre-registered tertile thresholds
SENS = [(75.0, 25.0), (60.0, 40.0)]        # sensitivity threshold pairs

# Axis pair (override on CLI: python e10_coverage.py sim_vehicle sim_food). Default animal/nature.
AX_A = sys.argv[1] if len(sys.argv) > 1 else "sim_animal"
AX_B = sys.argv[2] if len(sys.argv) > 2 else "sim_nature"
SUF = f"_{AX_A.replace('sim_','')}_{AX_B.replace('sim_','')}"   # output suffix per pair


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def classify(an, na, thi_a, tlo_a, thi_n, tlo_n):
    A = (an >= thi_a) & (na <= tlo_n)
    B = (na >= thi_n) & (an <= tlo_a)
    mid = ~(A | B)
    return A, B, mid


def cov_breakdown(an, na, thi_a, tlo_a, thi_n, tlo_n):
    A, B, mid = classify(an, na, thi_a, tlo_a, thi_n, tlo_n)
    fa, fb, fm = float(A.mean()), float(B.mean()), float(mid.mean())
    return {"fracA": fa, "fracB": fb, "fracMid": fm, "coverage": min(fa, fb)}


def boot_cov_ci(an, na, thr, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(an); vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        c = cov_breakdown(an[idx], na[idx], *thr)
        vals.append(c["coverage"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    ga, gn = d[f"g_{AX_A}"], d[f"g_{AX_B}"]
    corr = float(np.corrcoef(ga, gn)[0, 1])
    print(f"axes: A={AX_A} B={AX_B}", flush=True)
    za, zn = zscore(ga), zscore(gn)
    A_rank = np.argsort(za - zn)[::-1]        # clean animal-not-nature
    B_rank = np.argsort(zn - za)[::-1]        # clean nature-not-animal
    A_best, B_best = int(A_rank[0]), int(B_rank[0])
    A_top5, B_top5 = A_rank[:5].tolist(), B_rank[:5].tolist()
    print(f"corr(g_animal,g_nature)={corr:.3f}  A_best={A_best} B_best={B_best}", flush=True)

    pipe = G.load_pipe(); L = Labeler()
    methods = ["telling_best", "compositional", "showing_2bin", "showing_broad"]
    data = {m: {"an": [], "na": [], "per_seed": []} for m in methods}

    def gen_std(classes, seed):
        ids = G.sample_mu(G.uniform_pool(classes), N, seed=seed)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=seed))
        return np.asarray(l[AX_A], float), np.asarray(l[AX_B], float)

    for s in SEEDS:
        rng = np.random.default_rng(s)
        # telling: single best corner-A bin
        an, na = gen_std([A_best], s)
        data["telling_best"]["per_seed"].append((an, na))
        # showing 2-bin: 50/50 A_best + B_best
        an, na = gen_std([A_best, B_best], s + 1)
        data["showing_2bin"]["per_seed"].append((an, na))
        # showing broad: top5 A + top5 B
        an, na = gen_std(A_top5 + B_top5, s + 2)
        data["showing_broad"]["per_seed"].append((an, na))
        # compositional: conjunction of A_best and B_best (the practitioner's "both" tool)
        l = L.labels(generate_composed(pipe, [A_best] * N, [B_best] * N, scale=COMPOSE_SCALE, seed=s + 3))
        data["compositional"]["per_seed"].append((np.asarray(l[AX_A], float),
                                                  np.asarray(l[AX_B], float)))
        print(f"seed {s} done", flush=True)

    for m in methods:
        data[m]["an"] = np.concatenate([a for a, _ in data[m]["per_seed"]])
        data[m]["na"] = np.concatenate([n for _, n in data[m]["per_seed"]])

    # pre-registered thresholds: per-axis tertiles of the POOLED batch (all methods)
    all_an = np.concatenate([data[m]["an"] for m in methods])
    all_na = np.concatenate([data[m]["na"] for m in methods])

    def thresholds(hi, lo):
        return (np.percentile(all_an, hi), np.percentile(all_an, lo),
                np.percentile(all_na, hi), np.percentile(all_na, lo))

    thi_a, tlo_a, thi_n, tlo_n = thresholds(HI_PCT, LO_PCT)
    thr = (thi_a, tlo_a, thi_n, tlo_n)

    res = {"N": N, "cfg": CFG, "compose_scale": COMPOSE_SCALE, "seeds": SEEDS,
           "axis_A": AX_A, "axis_B": AX_B, "corr_axes_classes": corr,
           "A_best": A_best, "B_best": B_best, "A_top5": A_top5, "B_top5": B_top5,
           "thresholds_primary": {"tau_hi_animal": float(thi_a), "tau_lo_animal": float(tlo_a),
                                  "tau_hi_nature": float(thi_n), "tau_lo_nature": float(tlo_n),
                                  "hi_pct": HI_PCT, "lo_pct": LO_PCT},
           "methods": {}, "sensitivity": {}, "scores": {}}

    print(f"\nthresholds: animal hi={thi_a:.3f} lo={tlo_a:.3f} | nature hi={thi_n:.3f} lo={tlo_n:.3f}")
    print(f"{'method':14s} {'fracA':>7s} {'fracB':>7s} {'mid':>7s} {'coverage':>9s}  across-seed / bootstrap CI")
    for m in methods:
        an, na = data[m]["an"], data[m]["na"]
        bd = cov_breakdown(an, na, *thr)
        per_seed_cov = [cov_breakdown(a, n, *thr)["coverage"] for a, n in data[m]["per_seed"]]
        cm, clo, chi = across_seed_ci(per_seed_cov)
        blo, bhi = boot_cov_ci(an, na, thr, seed=7)
        res["methods"][m] = {**bd, "coverage_per_seed": per_seed_cov,
                             "coverage_across_seed_mean": cm, "coverage_across_seed_CI95": [clo, chi],
                             "coverage_bootstrap_CI95": [blo, bhi]}
        res["scores"][m] = {"animal": an.tolist(), "nature": na.tolist()}   # for the figure / reproducibility
        print(f"{m:14s} {bd['fracA']:7.3f} {bd['fracB']:7.3f} {bd['fracMid']:7.3f} {bd['coverage']:9.3f}  "
              f"seed[{clo:.3f},{chi:.3f}] boot[{blo:.3f},{bhi:.3f}]", flush=True)

    # sensitivity
    for hi, lo in SENS:
        th = thresholds(hi, lo)
        res["sensitivity"][f"{hi:.0f}/{lo:.0f}"] = {m: cov_breakdown(data[m]["an"], data[m]["na"], *th)
                                                    for m in methods}
        print(f"\nsensitivity {hi:.0f}/{lo:.0f}: " +
              " ".join(f"{m}={cov_breakdown(data[m]['an'],data[m]['na'],*th)['coverage']:.3f}" for m in methods))

    json.dump(res, open(f"{OUT}/e10_coverage{SUF}.json", "w"), indent=2)

    # ---- killer figure: 2-D (animal, nature) scatter, one panel per method ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"telling_best": "telling (1 bin)", "compositional": "compositional\n(Composable Diffusion)",
              "showing_2bin": "showing (2-bin mix)", "showing_broad": "showing (broad mix)"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes, methods):
        an, na = data[m]["an"], data[m]["na"]
        A, B, mid = classify(an, na, *thr)
        ax.scatter(an[mid], na[mid], s=8, c="0.7", alpha=0.5, label="middle")
        ax.scatter(an[A], na[A], s=10, c="#1f77b4", alpha=0.7, label="corner A")
        ax.scatter(an[B], na[B], s=10, c="#2ca02c", alpha=0.7, label="corner B")
        ax.axvline(thi_a, ls="--", c="k", lw=0.6); ax.axvline(tlo_a, ls=":", c="k", lw=0.6)
        ax.axhline(thi_n, ls="--", c="k", lw=0.6); ax.axhline(tlo_n, ls=":", c="k", lw=0.6)
        cov = res["methods"][m]["coverage"]
        ax.set_title(f"{titles[m]}\ncoverage={cov:.2f}  mid={res['methods'][m]['fracMid']:.2f}", fontsize=10)
        ax.set_xlabel(f"{AX_A} score")
    axes[0].set_ylabel(f"{AX_B} score"); axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"E10 — disjunction coverage ({AX_A} vs {AX_B}): showing spans both corners; compositional collapses to the middle", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig_e10_coverage_scatter{SUF}.png", dpi=130)
    print(f"saved figure -> fig_e10_coverage_scatter{SUF}.png")
    print("E10-COVERAGE-DONE")


if __name__ == "__main__":
    main()

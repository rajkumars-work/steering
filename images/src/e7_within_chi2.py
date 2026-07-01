"""E7 — realized within-bin chi^2 of telling (review.3 C1/Q1; the Prop-1 fix). HIGHEST PRIORITY.

Prop 1 (corrected): telling's within-bin mean-shift is bounded by sqrt(chi2_within * T), not
sqrt(T). To show the asymmetry is an *empirical regularity* (telling spends modest chi2; showing,
from E1's k-sweep, spends chi2 up to ~10^2), we measure how much within-bin chi2 per-prompt
guidance actually realizes.

Fix a class (bin). Baseline p = the within-bin target distribution at CFG=1 (the data
conditional). Turn the telling knob up: CFG sweep {1,2,3,5,7,10,15}. At each scale s the within-
bin distribution is q_s. Per scale, per target, report:
  - realized within-bin chi2(q_s||p) = sum_i (q_i-p_i)^2 / p_i   (histogram of L within the class)
  - a NOISE FLOOR chi2 from two independent baseline sub-samples (finite-sample histogram chi2 is
    positively biased; the floor is the chi2 you'd measure from sampling noise alone) and the
    floor-subtracted signal
  - realized mean-shift Delta = mean(q_s) - mean(p), and the within-bin ceiling sqrt(chi2 * v_b)
Aggregated over >=5 classes x 3 seeds with bootstrap + across-class 95% CIs. One generation set
per (class,scale) serves all targets. Headline: the within-bin chi2 of the hardest realistic
guidance — does it stay modest (~1-few) or blow up?
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
CLASSES = [207, 388, 817, 933, 973]      # retriever, panda, sports car, cheeseburger, coral reef
SCALES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
SEEDS = [0, 1, 2]
M = 64                                    # images per (class, scale, seed) -> 192 per (class,scale)
BINS = 12
TARGETS = ["aesthetic", "brightness", "sim_animal"]   # within-bin (high-T) + a concept contrast


def chi2_hist(samp_q, samp_p, edges, alpha=0.5):
    """Pearson chi2(q||p) on a shared histogram; p pseudocounted to avoid div-by-zero."""
    cq, _ = np.histogram(samp_q, bins=edges)
    cp, _ = np.histogram(samp_p, bins=edges)
    p = (cp + alpha) / (cp.sum() + alpha * len(cp))
    q = (cq + alpha) / (cq.sum() + alpha * len(cq))
    return float((((q - p) ** 2) / p).sum())


def boot_chi2_ci(samp_q, samp_p, edges, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        qi = samp_q[rng.integers(0, len(samp_q), len(samp_q))]
        vals.append(chi2_hist(qi, samp_p, edges))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def across_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def main():
    pipe = G.load_pipe(); L = Labeler()

    # samples[class][scale][target] = np.array of M*len(SEEDS) labels
    samples = {c: {} for c in CLASSES}
    for c in CLASSES:
        for s in SCALES:
            per_t = {t: [] for t in TARGETS}
            for sd in SEEDS:
                imgs = G.generate(pipe, [c] * M, guidance_scale=s, seed=1000 * sd + int(s))
                lab = L.labels(imgs)
                for t in TARGETS:
                    per_t[t].append(np.asarray(lab[t], float))
            samples[c][s] = {t: np.concatenate(per_t[t]) for t in TARGETS}
            print(f"class {c} scale {s} done", flush=True)

    res = {"classes": CLASSES, "scales": SCALES, "seeds": SEEDS, "m_per_seed": M,
           "bins": BINS, "targets": TARGETS, "per_target": {}}

    for t in TARGETS:
        per_scale = {}
        # per class: build shared bin edges over all scales' samples; baseline = scale 1.0
        chi2_by_class = {s: [] for s in SCALES}
        delta_by_class = {s: [] for s in SCALES}
        ceil_by_class = {s: [] for s in SCALES}
        floor_by_class = []
        boot = {s: [] for s in SCALES}
        for c in CLASSES:
            alls = np.concatenate([samples[c][s][t] for s in SCALES])
            lo, hi = float(alls.min()), float(alls.max())
            if hi <= lo:
                hi = lo + 1e-6
            edges = np.linspace(lo, hi, BINS + 1)
            base = samples[c][1.0][t]
            vb = float(base.var())
            base_mean = float(base.mean())
            # noise floor: split baseline in half, chi2 between halves
            h = len(base) // 2
            floor_by_class.append(chi2_hist(base[:h], base[h:], edges))
            for s in SCALES:
                q = samples[c][s][t]
                x2 = chi2_hist(q, base, edges)
                chi2_by_class[s].append(x2)
                delta_by_class[s].append(float(q.mean()) - base_mean)
                ceil_by_class[s].append((x2 * vb) ** 0.5)
                loci, hici = boot_chi2_ci(q, base, edges, seed=7)
                boot[s].append((loci, hici))
        floor_m, floor_lo, floor_hi = across_ci(floor_by_class)
        for s in SCALES:
            c2m, c2lo, c2hi = across_ci(chi2_by_class[s])
            dm, dlo, dhi = across_ci(delta_by_class[s])
            cem, celo, cehi = across_ci(ceil_by_class[s])
            per_scale[s] = {
                "chi2_within_mean": c2m, "chi2_within_CI95": [c2lo, c2hi],
                "chi2_minus_floor": max(0.0, c2m - floor_m),
                "delta_mean": dm, "delta_CI95": [dlo, dhi],
                "ceiling_sqrt_chi2_v_mean": cem, "ceiling_CI95": [celo, cehi],
                "per_class_chi2": chi2_by_class[s]}
        res["per_target"][t] = {"noise_floor_chi2": floor_m, "noise_floor_CI95": [floor_lo, floor_hi],
                                "per_scale": per_scale}
        print(f"\n=== {t} (noise floor chi2={floor_m:.3f}) ===", flush=True)
        print(f"{'CFG':>5s} {'chi2_within':>12s} {'chi2-floor':>11s} {'Delta':>10s} {'ceiling':>10s}")
        for s in SCALES:
            ps = per_scale[s]
            print(f"{s:5.0f} {ps['chi2_within_mean']:12.3f} {ps['chi2_minus_floor']:11.3f} "
                  f"{ps['delta_mean']:10.4f} {ps['ceiling_sqrt_chi2_v_mean']:10.4f}", flush=True)

    json.dump(res, open(f"{OUT}/e7_within_chi2.json", "w"), indent=2)
    print("E7-WITHIN-CHI2-DONE")


if __name__ == "__main__":
    main()

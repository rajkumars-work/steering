"""E1 — uncertainty on the Claim 2 (chi2 sweep) realized shifts (review P4).

claim2_chi2.py reports a single-seed realized shift per (target, k). Here we wrap it in
>=3 seeds and report the realized shift as mean +/- 95% CI across seeds, plus a bootstrap
CI over the pooled generated images. (The J->G fit CIs are delivered by e2_jg_converge.py;
the Claim 3 joint-hit CIs by e3_baseline.py — this covers the remaining headline numbers.)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
TARGETS = ["sim_animal", "brightness"]
KS = [400, 100, 25, 8]
N = 64
N_BASE = 80
CFG = 1.0
SEEDS = [7, 19, 31]


def boot_ci(x, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    cnt = d["count"].astype(float); JP = cnt / cnt.sum()
    sup = np.where(JP > 0)[0]
    pipe = G.load_pipe(); L = Labeler()

    res = {"cfg": CFG, "N": N, "N_base": N_BASE, "ks": KS, "seeds": SEEDS, "targets": {}}
    for t in TARGETS:
        g = d[f"g_{t}"]; gbar = float((JP * g).sum())
        E = float((JP * (g - gbar) ** 2).sum())
        levels = []
        for k in KS:
            topk = sup[np.argsort(g[sup])[::-1][:k]]
            mu = np.zeros(len(JP)); mu[topk] = 1.0 / len(topk)
            chi2 = float(((mu[JP > 0] - JP[JP > 0]) ** 2 / JP[JP > 0]).sum())
            pred = float((mu * g).sum() - gbar)
            ceiling = float((chi2 * E) ** 0.5)
            per_seed = []; pooled = []
            for s in SEEDS:
                base = np.asarray(L.labels(G.generate(pipe, G.sample_mu(JP, N_BASE, seed=s), guidance_scale=CFG, seed=s))[t], float)
                lab = np.asarray(L.labels(G.generate(pipe, G.sample_mu(mu, N, seed=1000 + s + k), guidance_scale=CFG, seed=1000 + s + k))[t], float)
                per_seed.append(float(lab.mean() - base.mean()))
                pooled.append(lab - base.mean())   # center each seed by its own baseline
            m, lo_s, hi_s = across_seed_ci(per_seed)
            lo_b, hi_b = boot_ci(np.concatenate(pooled), seed=5)
            levels.append({"k": k, "chi2": chi2, "pred_shift": pred, "ceiling": ceiling,
                           "realized_per_seed": per_seed,
                           "realized_mean": m, "across_seed_CI95": [lo_s, hi_s],
                           "bootstrap_CI95": [lo_b, hi_b],
                           "within_ceiling": bool(hi_s <= ceiling + 1e-9)})
            print(f"{t:11s} k={k:4d} chi2={chi2:7.1f} ceiling={ceiling:.3f} "
                  f"realized={m:.4f} seedCI[{lo_s:.4f},{hi_s:.4f}] bootCI[{lo_b:.4f},{hi_b:.4f}]", flush=True)
        res["targets"][t] = {"gbar": gbar, "E": E, "levels": levels}

    json.dump(res, open(f"{OUT}/e1_chi2_seeds.json", "w"), indent=2)
    print("E1-CHI2-SEEDS-DONE")


if __name__ == "__main__":
    main()

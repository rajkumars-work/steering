"""
Claim 2, the part we hadn't tested: showing's reach is bounded by sqrt(chi2 * E), and chi2 — how
hard the recipe mu_P departs from the data's bin mix J_P — is the dial. Concentrate mu_P more
(raise chi2), reach further, up to that ceiling.

For a target, build recipes at increasing concentration (uniform over the top-k bins by per-bin
mean g_b, for k = 400, 100, 25, 8 — smaller k = larger chi2). For each:
  data-side prediction: chi2(mu_P || J_P), predicted shift = sum mu_P g_b - gbar, ceiling sqrt(chi2 E)
  model-side realized:  generate under mu_P, measure the target's mean, shift vs an unsteered baseline
Expectation: the realized shift grows with chi2, tracks the predicted shift, and stays under the
ceiling — except where a heavy tail makes the ceiling loose (brightness), the honest edge.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
TARGETS = ["sim_animal", "brightness"]      # clean (high-E) and the heavy-tail edge
KS = [400, 100, 25, 8]                       # increasing concentration -> increasing chi2
N = 80                                       # images per recipe
N_BASE = 100                                 # unsteered baseline images
CFG = 1.0                                    # raw conditional


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    cnt = d["count"].astype(float); JP = cnt / cnt.sum()
    sup = np.where(JP > 0)[0]
    pipe = G.load_pipe(); L = Labeler()

    base_ids = G.sample_mu(JP, N_BASE, seed=7)
    base_lab = L.labels(G.generate(pipe, base_ids, guidance_scale=CFG, seed=7))

    res = {"cfg": CFG, "N": N, "ks": KS, "targets": {}}
    for t in TARGETS:
        g = d[f"g_{t}"]; gbar = float((JP * g).sum())
        E = float((JP * (g - gbar) ** 2).sum())
        base_mean = float(np.mean(base_lab[t]))
        levels = []
        for k in KS:
            topk = sup[np.argsort(g[sup])[::-1][:k]]
            mu = np.zeros(len(JP)); mu[topk] = 1.0 / len(topk)
            chi2 = float(((mu[JP > 0] - JP[JP > 0]) ** 2 / JP[JP > 0]).sum())
            pred = float((mu * g).sum() - gbar)
            ceiling = float((chi2 * E) ** 0.5)
            lab = L.labels(G.generate(pipe, G.sample_mu(mu, N, seed=100 + k), guidance_scale=CFG, seed=100 + k))
            realized = float(np.mean(lab[t])) - base_mean
            levels.append({"k": k, "chi2": chi2, "pred_shift": pred,
                           "ceiling": ceiling, "realized_shift": realized,
                           "within_ceiling": bool(abs(realized) <= ceiling + 1e-9)})
            print(f"{t:11s} k={k:4d}  chi2={chi2:7.1f}  ceiling={ceiling:.3f}  "
                  f"pred={pred:.3f}  realized={realized:.3f}  within={abs(realized)<=ceiling}", flush=True)
        res["targets"][t] = {"gbar": gbar, "E": E, "base_mean": base_mean, "levels": levels}

    json.dump(res, open(f"{OUT}/claim2_chi2.json", "w"), indent=2)
    print("CLAIM2-CHI2-DONE")


if __name__ == "__main__":
    main()

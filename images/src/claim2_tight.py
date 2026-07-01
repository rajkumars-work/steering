"""E5 — a regime where showing's bound is TIGHT and predictive (review P2).

The min-chi^2 recipe  muP(b) = JP(b) * [1 + Delta*(g_b - gbar)/E]  is the Cauchy-Schwarz-tight
direction: its data-side predicted shift equals the ceiling sqrt(chi2*E) (both equal Delta,
since chi2 = Delta^2 / E for this recipe). So if the model carries the target, the *realized*
shift should land right on the predicted ceiling — the bound predicting, not merely capping.

Contrast with claim2_chi2.py, where the top-k *uniform* recipe at high chi2 lands far below its
ceiling (the loose regime). Together: the bound is tight in the min-chi^2 regime, loose in the
concentrated-uniform regime, exactly as Cauchy-Schwarz says.

Outputs distributions/claim2_tight.json with bootstrap 95% CIs on every realized shift.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
TARGET = "sim_animal"          # clean, high-E target that carries well
DELTAS = [0.02, 0.04, 0.06]    # requested mean shifts (= predicted = ceiling, by construction)
N = 128                        # images per recipe
CFG = 1.0
SEED = 11


def boot_ci(x, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    cnt = d["count"].astype(float); JP = cnt / cnt.sum()
    g = d[f"g_{TARGET}"]; gbar = float((JP * g).sum())
    E = float((JP * (g - gbar) ** 2).sum())
    pipe = G.load_pipe(); L = Labeler()

    base = L.labels(G.generate(pipe, G.sample_mu(JP, N, seed=SEED), guidance_scale=CFG, seed=SEED))[TARGET]
    base_mean = float(np.mean(base))
    blo, bhi = boot_ci(base, seed=1)
    print(f"baseline mean {base_mean:.4f}  CI[{blo:.4f},{bhi:.4f}]", flush=True)

    levels = []
    for Delta in DELTAS:
        mu = JP * (1.0 + Delta * (g - gbar) / E)
        mu = np.clip(mu, 0, None); mu = mu / mu.sum()
        nz = JP > 0
        chi2 = float(((mu[nz] - JP[nz]) ** 2 / JP[nz]).sum())
        pred = float((mu * g).sum() - gbar)
        ceiling = float((chi2 * E) ** 0.5)
        s = SEED + int(round(Delta * 1000))
        lab = L.labels(G.generate(pipe, G.sample_mu(mu, N, seed=s), guidance_scale=CFG, seed=s))[TARGET]
        realized = float(np.mean(lab)) - base_mean
        lo, hi = boot_ci(lab, seed=2)
        rec = {"Delta_requested": Delta, "chi2": chi2, "pred_shift": pred, "ceiling": ceiling,
               "realized_shift": realized,
               "realized_CI": [lo - base_mean, hi - base_mean],
               "ceiling_in_CI": bool(lo - base_mean <= ceiling <= hi - base_mean)}
        levels.append(rec)
        print(f"Delta={Delta:.2f}  chi2={chi2:.3f}  pred=ceiling={ceiling:.4f}  "
              f"realized={realized:.4f} CI[{rec['realized_CI'][0]:.4f},{rec['realized_CI'][1]:.4f}]",
              flush=True)

    res = {"target": TARGET, "N": N, "cfg": CFG, "gbar": gbar, "E": E,
           "base_mean": base_mean, "base_CI": [blo, bhi], "levels": levels}
    json.dump(res, open(f"{OUT}/claim2_tight.json", "w"), indent=2)
    print("CLAIM2-TIGHT-DONE")


if __name__ == "__main__":
    main()

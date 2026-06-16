import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""
Generation-side reconstructions, consuming the audit's per-class stats (claim1_perclass.npz).

  Claim 2  (bounds): showing's data-side predicted shift = sum_c mu_P(c) g_b(c) - gbar
            must satisfy |shift| <= sqrt(chi2 * E); plus a model-side realization for one
            target (generate under the designed mu_P, measure the actual shift).
  Claim 4.1 (J->G bridge): generate at CFG=1 across a class spread; per-class generated
            means vs the audit's g_b -> r^2, MAD (structural targets track; aesthetic drifts).
  Claim 4.3 (soft Boltzmann): CFG sweep {1,3,7,12} at a fixed class -> monotonic target drift.
  Realization rho = sqrt(mean_b T(G)_b / mean_b v_b): within-bin spread of G vs the data.

NOTE: the bright-AND-animal joint (Claim 3) is intentionally NOT done here — in a purely
class-conditional DiT, "animal" is the selectable key and "brightness" an independent dial,
so the joint is the *independently-controllable* (telling-easy) case discussed for Application
§3; reconstructing it faithfully needs a careful telling/showing definition. Flagged for review.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

OUT = _os.path.join(_ROOT, "distributions")
STRUCTURAL = ["brightness", "filesize_per_mp", "sim_animal", "sim_vehicle", "sim_food", "sim_nature"]


def load_audit():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    w = d["w"]                      # J_P over 1000 classes
    g = {t: d[f"g_{t}"] for t in Labeler.TARGETS}
    v = {t: d[f"v_{t}"] for t in Labeler.TARGETS}
    return w, g, v


def chi2(muP, JP):
    nz = JP > 0
    return float((((muP[nz] - JP[nz]) ** 2) / JP[nz]).sum())


def main():
    w, g, v = load_audit()
    JP = w / w.sum()
    pipe = G.load_pipe()
    L = Labeler()
    res = {}

    # ---------- Claim 2: data-side bound (all targets) + one realization ----------
    c2 = []
    for t in Labeler.TARGETS:
        gb = g[t]; gbar = float((JP * gb).sum())
        E = float((JP * (gb - gbar) ** 2).sum())
        # top-k=50 classes by g_b among supported classes -> uniform muP
        supported = np.where(JP > 0)[0]
        topk = supported[np.argsort(gb[supported])[::-1][:50]]
        muP = np.zeros(1000); muP[topk] = 1.0 / len(topk)
        pred = float((muP * gb).sum() - gbar)
        bound = (chi2(muP, JP) * E) ** 0.5
        c2.append({"target": t, "E": E, "pred_shift": pred, "bound_sqrt_chi2E": bound,
                   "within_bound": abs(pred) <= bound + 1e-9})
    res["claim2_dataside"] = c2

    # realize brightness: generate under its top-k muP, measure actual shift vs audit gbar
    t = "brightness"; gb = g[t]; gbar = float((JP * gb).sum())
    supported = np.where(JP > 0)[0]; topk = supported[np.argsort(gb[supported])[::-1][:50]]
    muP = np.zeros(1000); muP[topk] = 1.0 / len(topk)
    ids = G.sample_mu(muP, 96, seed=1)
    imgs = G.generate(pipe, ids, guidance_scale=1.0, seed=1)
    realized = float(np.mean(L.labels(imgs)[t]) - gbar)
    res["claim2_realized_brightness"] = {"realized_shift": realized,
                                         "bound": (chi2(muP, JP) * float((JP*(gb-gbar)**2).sum()))**0.5,
                                         "gbar_audit": gbar}

    # ---------- Claim 4.1: DiT (CFG=1) per-class means vs audit g_b ----------
    classes = list(range(0, 1000, 20))   # 50-class spread
    m = 8
    genmeans = {t: [] for t in Labeler.TARGETS}
    genvars = {t: [] for t in Labeler.TARGETS}
    audit_g = {t: [] for t in Labeler.TARGETS}
    for c in classes:
        imgs = G.generate(pipe, [c] * m, guidance_scale=1.0, seed=100 + c)
        lab = L.labels(imgs)
        for t in Labeler.TARGETS:
            genmeans[t].append(float(np.mean(lab[t])))
            genvars[t].append(float(np.var(lab[t])))
            audit_g[t].append(float(g[t][c]))
    c41 = {}
    for t in Labeler.TARGETS:
        x = np.array(audit_g[t]); y = np.array(genmeans[t])
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        c41[t] = {"r2": r * r, "MAD": float(np.mean(np.abs(x - y))), "mean_offset": float(np.mean(y - x))}
    res["claim41_dit_vs_audit"] = c41

    # Realization rho per target (within-bin spread of G vs audit v_b), over the 50 classes
    rho = {}
    for t in Labeler.TARGETS:
        TG = float(np.mean(genvars[t]))
        Td = float(np.mean([v[t][c] for c in classes]))
        rho[t] = {"rho": (TG / Td) ** 0.5 if Td > 0 else float("nan"), "T_G": TG, "T_data": Td}
    res["realization_rho"] = rho

    # ---------- Claim 4.3: CFG sweep drift at a fixed class ----------
    sweep = {}
    for cfg in [1.0, 3.0, 7.0, 12.0]:
        imgs = G.generate(pipe, [207] * 32, guidance_scale=cfg, seed=7)
        lab = L.labels(imgs)
        sweep[cfg] = {t: float(np.mean(lab[t])) for t in ["aesthetic", "brightness", "sim_animal"]}
    res["claim43_cfg_sweep_class207"] = sweep

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/claims234.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("CLAIMS234-DONE")


if __name__ == "__main__":
    main()

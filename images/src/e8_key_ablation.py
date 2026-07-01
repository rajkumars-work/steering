"""E8 — key-sensitivity ablation (review.2 C3, both reviewers Q5). DATA-SIDE ONLY.

The budget T/E is defined relative to the key pi (the partition into bins). Reviewers ask how
robust the split and the reaches are to that choice. We recompute T, E, E/(E+T) under several
keys, all from the shadow alone via the law of total variance (no generation, no relabeling of
images needed — any key that is a *coarsening* of the 1000 ImageNet classes is computable from
the per-class count/mean/within-var):

  key A  1000 ImageNet classes (current pi)
  key B  k-means on per-class target-profiles -> 50 and 100 semantic bins (a *structured* coarse key)
  key C  random partition of the 1000 classes into 50 bins (a control: a key with no structure)

For a coarse bin B = union of fine classes c (weights w_c, means g_c, within-var v_c):
  W_B = sum w_c ; G_B = sum w_c g_c / W_B ; V_B = sum w_c (v_c + (g_c-G_B)^2) / W_B   (total variance)
  T(key) = sum p_B V_B ;  E(key) = sum p_B (G_B - gbar)^2 ;  with p_B = W_B / sum W.
T+E is invariant to the key (asserted). The question: does E/(E+T) — the room for showing vs
telling — stay qualitatively the same? Expectation: structured keys (1000, k-means) keep the
high-E targets (class-aligned concepts) high-E; a random key collapses E (no between-bin
structure), confirming the budget tracks a *meaningful* key, not an artifact of cardinality.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
TARGETS = Labeler.TARGETS


def kmeans(F, k, seed, iters=100):
    rng = np.random.default_rng(seed)
    cent = F[rng.choice(len(F), k, replace=False)].copy()
    for _ in range(iters):
        d = ((F[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([F[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab


def split_for_key(labels, w, g, v):
    """labels: bin id per fine class. Returns T, E, gbar via law of total variance."""
    p = w / w.sum()
    gbar = float((p * g).sum())
    T = E = 0.0
    for b in np.unique(labels):
        m = labels == b
        W = w[m].sum()
        if W == 0:
            continue
        pB = W / w.sum()
        GB = float((w[m] * g[m]).sum() / W)
        VB = float((w[m] * (v[m] + (g[m] - GB) ** 2)).sum() / W)
        T += pB * VB
        E += pB * (GB - gbar) ** 2
    return T, E, gbar


def showing_reach(labels, w, g):
    """Top-k(=up to 50) bins by bin-mean -> uniform muP over bins; predicted shift and sqrt(chi2*E)."""
    p = w / w.sum()
    gbar = float((p * g).sum())
    bins = np.unique(labels)
    GB = np.array([(w[labels == b] * g[labels == b]).sum() / w[labels == b].sum() for b in bins])
    WB = np.array([w[labels == b].sum() for b in bins]); JB = WB / WB.sum()
    E = float((JB * (GB - gbar) ** 2).sum())
    k = min(50, len(bins))
    top = np.argsort(GB)[::-1][:k]
    mu = np.zeros(len(bins)); mu[top] = 1.0 / k
    chi2 = float(((mu - JB) ** 2 / JB).sum())
    pred = float((mu * GB).sum() - gbar)
    return {"pred_shift": pred, "ceiling_sqrt_chi2E": (chi2 * E) ** 0.5, "chi2": chi2, "nbins": len(bins)}


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    w = d["count"].astype(float)
    G = {t: d[f"g_{t}"].astype(float) for t in TARGETS}
    V = {t: d[f"v_{t}"].astype(float) for t in TARGETS}

    # feature matrix for the structured (k-means) key: standardized per-class target profiles
    F = np.stack([(G[t] - G[t].mean()) / (G[t].std() + 1e-12) for t in TARGETS], axis=1)

    keys = {}
    keys["imagenet_1000"] = np.arange(1000)
    keys["kmeans_100"] = kmeans(F, 100, seed=0)
    keys["kmeans_50"] = kmeans(F, 50, seed=0)
    keys["random_50"] = np.random.default_rng(1).integers(0, 50, size=1000)

    res = {"keys": {}, "targets": TARGETS}
    print(f"{'target':16s} " + " ".join(f"{k:>16s}" for k in keys))
    rows = {t: {} for t in TARGETS}
    for kname, lab in keys.items():
        res["keys"][kname] = {"nbins": int(len(np.unique(lab))), "split": {}, "reach": {}}
        for t in TARGETS:
            T, E, gbar = split_for_key(lab, w, G[t], V[t])
            frac = E / (E + T) if (E + T) > 0 else float("nan")
            res["keys"][kname]["split"][t] = {"T": T, "E": E, "E_frac": frac, "TplusE": T + E}
            res["keys"][kname]["reach"][t] = showing_reach(lab, w, G[t])
            rows[t][kname] = frac
    # invariance check: T+E constant across keys
    inv_ok = {}
    for t in TARGETS:
        tot = [res["keys"][k]["split"][t]["TplusE"] for k in keys]
        inv_ok[t] = float(np.std(tot) / (np.mean(tot) + 1e-30))
    res["TplusE_rel_spread_across_keys"] = inv_ok

    print("\n=== E/(E+T) per target per key ===")
    print(f"{'target':16s} " + " ".join(f"{k:>14s}" for k in keys))
    for t in TARGETS:
        print(f"{t:16s} " + " ".join(f"{rows[t][k]:14.3f}" for k in keys))
    print("\nT+E rel-spread across keys (should be ~0, invariance):",
          {t: round(v, 8) for t, v in inv_ok.items()})

    json.dump(res, open(f"{OUT}/e8_key_ablation.json", "w"), indent=2)
    print("E8-KEY-ABLATION-DONE")


if __name__ == "__main__":
    main()

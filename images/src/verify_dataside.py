import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""
Reader-facing verification of the DATA-SIDE claims, from the (P,L) SHADOW alone.

Inputs (tiny, license-clean — derived statistics, not images):
  claim1_perclass.npz : per-class count, w_b, g_b[target], v_b[target]  (~130 KB)
  claim1.json         : the global Var(L) per target (one scalar each) + N

Reproduces, with NO ImageNet, NO models, NO GPU, in well under a second:
  Claim 1a : T + E reconstructed from the per-class shadow equals the global Var(L)
  Claim 1  : the E/(E+T) spectrum (class-aligned concepts high-E, pixel targets high-T)
  Claim 1b : coarsening 1000->50 bins (pooled) moves variance E -> T
  Claim 2  : showing's data-side bound  |sum_c mu_P(c) g_b(c) - gbar| <= sqrt(chi2 * E)

This is the framework's own thesis made checkable: you only ever need the shadow.
"""
import json, os
import numpy as np

OUT = _os.path.join(_ROOT, "distributions")
TARGETS = ["aesthetic", "brightness", "filesize_per_mp",
           "sim_animal", "sim_vehicle", "sim_food", "sim_nature"]


def pool(count, g, v, group):
    """Merge fine bins into superbins by `group` labels. Returns superbin count,g,v."""
    G = group.max() + 1
    n = np.bincount(group, weights=count, minlength=G)
    sg = np.bincount(group, weights=count * g, minlength=G)
    sgg = np.bincount(group, weights=count * (v + g * g), minlength=G)   # n*E[L^2]
    nz = n > 0
    gs = np.zeros(G); gs[nz] = sg[nz] / n[nz]
    vs = np.zeros(G); vs[nz] = sgg[nz] / n[nz] - gs[nz] ** 2
    return n, gs, vs


def TE(count, g, v):
    N = count.sum(); w = count / N
    gbar = float((w * g).sum())
    T = float((w * v).sum())
    E = float((w * (g - gbar) ** 2).sum())
    return T, E, gbar


def main():
    d = np.load(os.path.join(OUT, "claim1_perclass.npz"))
    j = json.load(open(os.path.join(OUT, "claim1.json")))
    var = {r["target"]: r["Var"] for r in j["split"]}
    count = d["count"]

    print(f"Loaded shadow: {int(count.sum())} samples over {int((count>0).sum())} bins, "
          f"{len(TARGETS)} targets.  (no images, no models)\n")

    print("=== Claim 1a: per-class shadow reconstructs Var(L) ===")
    print(f"{'target':16s} {'T':>10s} {'E':>10s} {'T+E':>12s} {'Var(shipped)':>14s} {'|resid|':>10s}")
    ok = True
    for t in TARGETS:
        T, E, _ = TE(count, d[f"g_{t}"], d[f"v_{t}"])
        resid = abs((T + E) - var[t]); rel = resid / (var[t] + 1e-30)
        ok &= rel < 1e-6
        print(f"{t:16s} {T:10.4g} {E:10.4g} {T+E:12.6g} {var[t]:14.6g} {resid:10.2e}")
    print(f"  -> identity holds (relative resid < 1e-6): {ok}\n")

    print("=== Claim 1: spectrum  E/(E+T)  (concepts high, pixels low) ===")
    spec = []
    for t in TARGETS:
        T, E, _ = TE(count, d[f"g_{t}"], d[f"v_{t}"]); spec.append((t, E / (E + T)))
    for t, f in sorted(spec, key=lambda x: -x[1]):
        print(f"  {t:16s} E/(E+T) = {f:5.3f}")

    print("\n=== Claim 1b: coarsen 1000 -> 50 bins (E should fall, T rise) ===")
    rng = np.random.default_rng(0); group = rng.integers(0, 50, size=len(count))
    print(f"{'target':16s} {'E_fine':>10s} {'E_coarse':>10s} {'T_fine':>10s} {'T_coarse':>10s}")
    for t in TARGETS:
        Tf, Ef, _ = TE(count, d[f"g_{t}"], d[f"v_{t}"])
        nc, gc, vc = pool(count, d[f"g_{t}"], d[f"v_{t}"], group)
        Tc, Ec, _ = TE(nc, gc, vc)
        print(f"{t:16s} {Ef:10.4g} {Ec:10.4g} {Tf:10.4g} {Tc:10.4g}")

    print("\n=== Claim 2: showing data-side bound |shift| <= sqrt(chi2 * E)  (top-50 muP) ===")
    N = count.sum(); JP = count / N
    print(f"{'target':16s} {'pred_shift':>11s} {'bound':>11s} {'within?':>8s}")
    for t in TARGETS:
        g = d[f"g_{t}"]; T, E, gbar = TE(count, g, v=d[f"v_{t}"])
        sup = np.where(JP > 0)[0]; topk = sup[np.argsort(g[sup])[::-1][:50]]
        muP = np.zeros(len(count)); muP[topk] = 1.0 / len(topk)
        chi2 = float((((muP[JP > 0] - JP[JP > 0]) ** 2) / JP[JP > 0]).sum())
        shift = float((muP * g).sum() - gbar); bound = (chi2 * E) ** 0.5
        print(f"{t:16s} {shift:11.4g} {bound:11.4g} {str(abs(shift) <= bound + 1e-9):>8s}")

    print("\nVERIFY-DATASIDE-DONE (shadow only)")


if __name__ == "__main__":
    main()

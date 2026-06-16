import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""
Claim 1 — the LoTV budget split is exact, and moves with the binning.

Audits an ImageNet subset (streamed val split; ground-truth class = the key pi):
per-class (w_b, g_b, v_b) for each of the 7 targets, then
    T = sum_b w_b v_b   (within-bin),   E = sum_b w_b (g_b - gbar)^2  (between-bin)
and checks  T + E == Var(L)  to machine precision.

Also reports:
  - the E/(E+T) spectrum across the 7 targets (class-aligned high-E vs pixel-level high-T)
  - coarsening: merge the 1000 classes into B' random superbins -> E should fall, T rise.

Usage: python audit_claim1.py --per_class 25
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler

OUT = _os.path.join(_ROOT, "distributions")


def lotv(values, bins, nbins):
    """values: [N] float, bins: [N] int in [0,nbins). Returns T, E, Var (population)."""
    N = len(values)
    gbar = values.mean()
    var = values.var()  # population
    # per-bin counts, sums, sumsqs
    cnt = np.bincount(bins, minlength=nbins).astype(np.float64)
    s = np.bincount(bins, weights=values, minlength=nbins)
    s2 = np.bincount(bins, weights=values * values, minlength=nbins)
    nz = cnt > 0
    w = cnt / N
    g = np.zeros(nbins); g[nz] = s[nz] / cnt[nz]
    v = np.zeros(nbins); v[nz] = s2[nz] / cnt[nz] - g[nz] ** 2
    T = float((w * v).sum())
    E = float((w * (g - gbar) ** 2).sum())
    return T, E, float(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from datasets import load_dataset
    # Canonical ImageNet-1k (gate accepted on this account); standard 0..999 labels.
    tok = open(os.path.expanduser("~/.ssh/hf-read.key")).read().strip()
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=tok)

    L = Labeler()
    targets = Labeler.TARGETS
    # collect per-image: label value arrays + class id
    vals = {t: [] for t in targets}
    keys = []
    per_class = np.zeros(1000, int)
    batch_imgs, batch_cls = [], []
    seen = 0

    def flush():
        nonlocal batch_imgs, batch_cls
        if not batch_imgs:
            return
        lab = L.labels(batch_imgs)
        for t in targets:
            vals[t].extend(lab[t].tolist())
        keys.extend(batch_cls)
        batch_imgs, batch_cls = [], []

    for ex in ds:
        c = int(ex["label"])
        if per_class[c] >= args.per_class:
            continue
        per_class[c] += 1
        batch_imgs.append(ex["image"].convert("RGB"))
        batch_cls.append(c)
        seen += 1
        if len(batch_imgs) >= args.batch:
            flush()
        if seen % 2000 == 0:
            print(f"  ... labeled {seen} images ({(per_class>0).sum()} classes seen)", flush=True)
        if per_class.min() >= args.per_class and (per_class >= args.per_class).all():
            break
    flush()

    keys = np.array(keys)
    N = len(keys)
    print(f"audited N={N} images across {len(np.unique(keys))} classes "
          f"(target {args.per_class}/class)", flush=True)

    # --- Claim 1a: split exact, per target ---
    rows = []
    for t in targets:
        v = np.array(vals[t], np.float64)
        T, E, Var = lotv(v, keys, 1000)
        rows.append({"target": t, "T": T, "E": E, "T+E": T + E, "Var": Var,
                     "abs_resid": abs((T + E) - Var), "E_frac": E / (T + E) if (T + E) else 0.0})

    # --- Claim 1b: coarsening (1000 -> 50 random superbins) ---
    rng = np.random.default_rng(0)
    super_map = rng.integers(0, 50, size=1000)
    coarse_keys = super_map[keys]
    coarse = []
    for t in targets:
        v = np.array(vals[t], np.float64)
        Tf, Ef, _ = lotv(v, keys, 1000)            # fine
        Tc, Ec, _ = lotv(v, coarse_keys, 50)        # coarse
        coarse.append({"target": t, "E_fine": Ef, "E_coarse": Ec, "T_fine": Tf, "T_coarse": Tc})

    res = {"N": N, "per_class": args.per_class, "split": rows, "coarsen": coarse}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/claim1.json", "w") as f:
        json.dump(res, f, indent=2)

    # Per-class (w_b, g_b, v_b) for every target, for downstream mu_P design (claims 2-4)
    cnt = np.bincount(keys, minlength=1000).astype(np.float64)
    pc = {"count": cnt, "w": cnt / N}
    for t in targets:
        v = np.array(vals[t], np.float64)
        s = np.bincount(keys, weights=v, minlength=1000)
        s2 = np.bincount(keys, weights=v * v, minlength=1000)
        nz = cnt > 0
        g = np.zeros(1000); g[nz] = s[nz] / cnt[nz]
        vv = np.zeros(1000); vv[nz] = s2[nz] / cnt[nz] - g[nz] ** 2
        pc[f"g_{t}"] = g
        pc[f"v_{t}"] = vv
    np.savez(f"{OUT}/claim1_perclass.npz", **pc)

    print("\n=== Claim 1a: T + E = Var(L) (population) ===")
    print(f"{'target':18s} {'T':>10s} {'E':>10s} {'T+E':>10s} {'Var':>10s} {'|resid|':>10s} {'E/(E+T)':>8s}")
    for r in rows:
        print(f"{r['target']:18s} {r['T']:10.5f} {r['E']:10.5f} {r['T+E']:10.5f} "
              f"{r['Var']:10.5f} {r['abs_resid']:10.2e} {r['E_frac']:8.3f}")
    print("\n=== Claim 1b: coarsen 1000->50 bins (E should fall, T rise) ===")
    print(f"{'target':18s} {'E_fine':>9s} {'E_coarse':>9s} {'T_fine':>9s} {'T_coarse':>9s}")
    for c in coarse:
        print(f"{c['target']:18s} {c['E_fine']:9.5f} {c['E_coarse']:9.5f} {c['T_fine']:9.5f} {c['T_coarse']:9.5f}")
    print("\nCLAIM1-DONE ->", f"{OUT}/claim1.json")


if __name__ == "__main__":
    main()

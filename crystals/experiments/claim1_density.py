#!/usr/bin/env python3
"""
Claim 1 (crystals), DENSITY-ONLY fast view: does the data-audit budget split survive the model?

The easy end of the difficulty ladder (see claim1_survival_spectrum.py), run on its own so it
finishes in minutes with NO scorer to load. Density is the fair, brightness-like analog: every
generated structure has one, the model produces it reliably, it is NOT a curation criterion, and
it is pure geometry (mass / cell volume) — so no MACE, no hull, no inf.

  target : DENSITY (g/cm^3)        bins : chemistry = (element set, atom count)

We read the within/between split off the DATA, GENERATE from CUES, bin the model's own outputs
the same way, and recompute the split. One measure (density, g/cm^3) for BOTH sides.

    CUES_TRAIN_CSV=/path/to/alex_nolemat_lowhull_dataset_train.csv \\
        python experiments/claim1_density.py     # -> experiments/out/claim1_survival_density.json

NOTE: needs the training rows (real structures) for the DATA side, NOT in this data-light repo —
point CUES_TRAIN_CSV at the dataset from the comprehensive repo.
"""
import os, json, time, csv, random
from collections import defaultdict
import numpy as np
from _repo import CKPT, VERSION, DEVICE, OUTDIR

TRAIN_CSV = os.environ.get("CUES_TRAIN_CSV", "")
OUT = os.path.join(OUTDIR, "claim1_survival_density.json")
K_BINS      = 30     # chemistry bins to test (a spread); cheap now, so more than the MACE run
MIN_MEMBERS = 12     # data structures required in a bin to include it
M_DATA      = 15     # data structures measured per bin
M_GEN       = 20     # structures generated per bin
SEED        = 0
AMU_PER_A3_TO_G_CC = 1.6605390666


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def split(means, vars):
    """uniform-weight within (T) and between (E) from per-bin means and variances."""
    return float(np.mean(vars)), float(np.var(means))


def density(s):
    """g/cm^3, the same way for a data structure or a generated one. Handles a pymatgen
    Structure (.density) or an ASE Atoms (mass / volume)."""
    if s is None:
        return None
    try:
        if hasattr(s, "density"):                       # pymatgen Structure -> already g/cc
            return float(s.density)
        m = float(np.sum(s.get_masses()))               # ASE: amu
        v = float(s.get_volume())                       # Å^3
        if v <= 0:
            return None
        return m / v * AMU_PER_A3_TO_G_CC
    except Exception:
        return None


def row_to_atoms(row):
    """Decode a training row's structure with the SAME decoder generate_one uses (parse_target),
    so the data side is measured on the identical density scale.
    row layout: source, target, id, origin, label, e_above_hull, in_lemat."""
    from dielectric_data.reader import parse_target
    try:
        return parse_target(row[1], VERSION, row[3])
    except Exception:
        return None


def main():
    if not TRAIN_CSV or not os.path.isfile(TRAIN_CSV):
        raise SystemExit(
            "Set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV (from the comprehensive "
            f"repo). Got: {TRAIN_CSV!r}")

    from eval.screening import load_model, generate_one

    # ---- 1. read the data, group rows by chemistry bin -----------------------------------
    bins = defaultdict(list)
    with open(TRAIN_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) < 6:
                continue
            src = row[0].split("|")
            if len(src) < 2:
                continue
            key = (src[0].strip(), src[1].strip())          # (elements, natoms)
            bins[key].append(row)

    rng = random.Random(SEED)
    eligible = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    log(f"{len(eligible)} chemistry bins have >= {MIN_MEMBERS} members")
    chosen = rng.sample(eligible, min(K_BINS, len(eligible)))

    model, sp = load_model(CKPT, DEVICE)

    data_means, data_vars, model_means, model_vars, used = [], [], [], [], []
    for bi, key in enumerate(chosen):
        els, nat = key
        prompt = f"{els} | {nat} | "

        # ---- 2. DATA side: density of real structures ----------------------------------
        rows = rng.sample(bins[key], min(M_DATA, len(bins[key])))
        d = [x for x in (density(row_to_atoms(row)) for row in rows) if x is not None]

        # ---- 3. MODEL side: generate from the same chemistry, density the same way -----
        g = []
        for _ in range(M_GEN):
            try:
                out = generate_one(model, sp, prompt, "alex", VERSION, DEVICE, top_k=10)
                a = getattr(out, "atoms", None)
                x = density(a)
                if x is not None:
                    g.append(x)
            except Exception:
                pass

        if len(d) >= 3 and len(g) >= 3:
            data_means.append(np.mean(d)); data_vars.append(np.var(d))
            model_means.append(np.mean(g)); model_vars.append(np.var(g))
            used.append({"elements": els, "natoms": nat, "n_data": len(d), "n_gen": len(g),
                         "data_mean": float(np.mean(d)), "model_mean": float(np.mean(g))})
        log(f"  [{bi+1}/{len(chosen)}] {els} | {nat}: data n={len(d)} mean={np.mean(d) if d else float('nan'):.3f}"
            f"  model n={len(g)} mean={np.mean(g) if g else float('nan'):.3f}")

    # ---- 4. within/between split on each side, same scale --------------------------------
    dm, dv = np.array(data_means), np.array(data_vars)
    mm, mv = np.array(model_means), np.array(model_vars)
    T_data, E_data = split(dm, dv)
    T_model, E_naive = split(mm, mv)
    E_model = E_naive - T_model / M_GEN                 # remove sampling-noise inflation
    r = float(np.corrcoef(dm, mm)[0, 1]) if len(dm) > 2 else float("nan")

    res = {"ckpt": CKPT, "k_bins_used": len(used), "M_data": M_DATA, "M_gen": M_GEN,
           "target": "density_g_per_cc", "measure": "geometry (mass/volume), same for both sides",
           "data":  {"T": T_data, "E": E_data, "V": T_data + E_data},
           "model": {"T": T_model, "E": E_model, "E_naive": E_naive, "V": T_model + E_model},
           "ratio": {"T_model_over_data": T_model / T_data if T_data else None,
                     "E_model_over_data": E_model / E_data if E_data else None},
           "per_bin_mean_r2": r * r, "bins": used}
    print("\n            within(T)   between(E)   total(V)")
    print(f"  DATA    {T_data:10.5f} {E_data:11.5f} {T_data+E_data:10.5f}")
    print(f"  MODEL   {T_model:10.5f} {E_model:11.5f} {T_model+E_model:10.5f}")
    print(f"  per-bin mean alignment r^2 = {r*r:.3f}")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    log(f"wrote {OUT}")
    print("CLAIM1-DENSITY-DONE")


if __name__ == "__main__":
    main()

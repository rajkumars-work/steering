#!/usr/bin/env python3
"""
Claim 1 (crystals): how far does the data-audit budget survive the model — as a function of
how hard the property is for the model to get right?

The crystal mirror of the ImageNet experiment, across a DIFFICULTY LADDER, on the SAME
generated structures, so we watch the statistics degrade as the property gets harder:

  density        geometry (mass/volume)                      EASIEST  — model nails it
  SMACT          composition charge-balance / electroneg.    easy
  BVS            bond-valence self-consistency (geometry)    medium
  validity       SMACT + BVS + min-distance all pass         hard
  e_above_hull   stability vs the convex hull                HARDEST  — what models are worst at

bins = chemistry (element set, atom count). For each bin we read the statistic off the DATA
and recompute it on the MODEL's own outputs.

Thesis, made precise: the data-side budget is preserved only to the extent the model has
CONVERGED on the property. Two views, both written to JSON:
  (A) CONVERGENCE LADDER — pass rate, data vs model, SMACT -> BVS -> validity -> (meta)stable.
  (B) BUDGET SURVIVAL — within (T) / between (E) split + per-chemistry mean r^2, for the
      CONTINUOUS properties density -> BVS GII -> e_above_hull.

    CUES_TRAIN_CSV=/path/to/alex_nolemat_lowhull_dataset_train.csv \\
        python experiments/claim1_survival_spectrum.py    # -> experiments/out/claim1_survival_spectrum.json

NOTE: the DATA side needs the training rows (real structures), NOT in this data-light repo —
point CUES_TRAIN_CSV at the dataset from the comprehensive repo. Both sides decode through the
SAME parse_target decoder and use one measure per property. Cost: MACE loads once, multi-hour.
"""
import os, json, time, csv, random
from collections import defaultdict
import numpy as np
from _repo import CKPT, VERSION, DEVICE, OUTDIR

TRAIN_CSV = os.environ.get("CUES_TRAIN_CSV", "")
OUT = os.path.join(OUTDIR, "claim1_survival_spectrum.json")
K_BINS, MIN_MEMBERS, M_DATA, M_GEN, SEED = 24, 10, 10, 12, 0
EAH_MAX = 5.0
AMU_PER_A3_TO_G_CC = 1.6605390666
CONTINUOUS = ["density", "bvs_gii", "e_above_hull"]                 # easy -> hard (survival)
LADDER     = ["smact_pass", "bvs_pass", "validity_pass", "metastable"]   # easy -> hard (pass rate)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_ase(s):
    if s is None: return None
    if hasattr(s, "get_volume"): return s                          # already ASE
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor.get_atoms(s)
    except Exception:
        return None


def bvs_gii(a):
    """Global Instability Index — the CONTINUOUS bond-valence self-consistency metric:
    RMS over sites of (bond-valence sum from bond lengths  -  formal oxidation state), in
    valence units. 0 = perfectly self-consistent; larger = strained/unphysical bonding.

    Computed identically for data and generated structures. None when oxidation states can't
    be assigned (e.g. pure intermetallics with no anion) or the structure is unusable. (This
    replaces check_validity's `bvs_max_deviation`, which is |valence - round(valence)| on the
    INTEGER assigned valences and so is structurally ~0 — useless as a continuous signal.)"""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
        struct = a if hasattr(a, "sites") else AseAtomsAdaptor.get_structure(a)
        oxi = BVAnalyzer().get_oxi_state_decorated_structure(struct)
        devs = []
        for site in oxi:
            nn = oxi.get_neighbors(site, 4.0)
            if not nn:
                continue
            devs.append(calculate_bv_sum(site, nn) - site.specie.oxi_state)
        return float(np.sqrt(np.mean(np.square(devs)))) if devs else None
    except Exception:
        return None


def measure(s, calc, check_validity, compute_eah):
    """All properties of one structure; None where undefined."""
    out = dict.fromkeys(CONTINUOUS + LADDER, None)
    a = to_ase(s)
    if a is None: return out
    try:
        v = float(a.get_volume())
        if v > 0: out["density"] = float(np.sum(a.get_masses())) / v * AMU_PER_A3_TO_G_CC
    except Exception: pass
    try:
        r = check_validity(a)
        out["smact_pass"] = bool(r.get("smact_pass")) if r.get("smact_pass") is not None else None
        out["bvs_pass"] = bool(r.get("bvs_pass")) if r.get("bvs_pass") is not None else None
        out["validity_pass"] = bool(r.get("validity_pass")) if r.get("validity_pass") is not None else None
    except Exception: pass
    out["bvs_gii"] = bvs_gii(a)                                    # continuous BVS self-consistency (GII)
    try:
        e = compute_eah(a, calc=calc, timeout=120).get("e_above_hull")
        if e is not None and np.isfinite(e) and abs(e) <= EAH_MAX:
            out["e_above_hull"] = float(e)
            out["metastable"] = bool(e <= 0.1)
    except Exception: pass
    return out


def row_to_atoms(row):
    from dielectric_data.reader import parse_target
    try: return parse_target(row[1], VERSION, row[3])
    except Exception: return None


def main():
    if not TRAIN_CSV or not os.path.isfile(TRAIN_CSV):
        raise SystemExit(
            "Set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV (from the comprehensive "
            f"repo). Got: {TRAIN_CSV!r}")

    from eval.screening import load_model, generate_one
    from chem.stability import compute_e_above_hull, load_stability_calc
    from chem.validity import check_validity

    bins = defaultdict(list)
    with open(TRAIN_CSV) as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            if len(row) < 6: continue
            src = row[0].split("|")
            if len(src) >= 2: bins[(src[0].strip(), src[1].strip())].append(row)
    rng = random.Random(SEED)
    eligible = [k for k, v in bins.items() if len(v) >= MIN_MEMBERS]
    chosen = rng.sample(eligible, min(K_BINS, len(eligible)))
    log(f"{len(eligible)} eligible bins; testing {len(chosen)}")

    model, sp = load_model(CKPT, DEVICE)
    calc = load_stability_calc(device=DEVICE)

    cont = {p: {"dm": [], "dv": [], "mm": [], "mv": []} for p in CONTINUOUS}
    ladder = {p: {"d": [], "m": []} for p in LADDER}                # per-bin pass rates
    for bi, (els, nat) in enumerate(chosen):
        prompt = f"{els} | {nat} | "
        rows = rng.sample(bins[(els, nat)], min(M_DATA, len(bins[(els, nat)])))
        dvals = {p: [] for p in CONTINUOUS + LADDER}
        mvals = {p: [] for p in CONTINUOUS + LADDER}
        for row in rows:
            for p, x in measure(row_to_atoms(row), calc, check_validity, compute_e_above_hull).items():
                if x is not None: dvals[p].append(float(x))
        for _ in range(M_GEN):
            try:
                o = generate_one(model, sp, prompt, "alex", VERSION, DEVICE, top_k=10)
                a = getattr(o, "atoms", None)
                if a is None: continue
                for p, x in measure(a, calc, check_validity, compute_e_above_hull).items():
                    if x is not None: mvals[p].append(float(x))
            except Exception: pass
        for p in CONTINUOUS:
            if len(dvals[p]) >= 3 and len(mvals[p]) >= 3:
                cont[p]["dm"].append(np.mean(dvals[p])); cont[p]["dv"].append(np.var(dvals[p]))
                cont[p]["mm"].append(np.mean(mvals[p])); cont[p]["mv"].append(np.var(mvals[p]))
        for p in LADDER:
            if dvals[p]: ladder[p]["d"].append(np.mean(dvals[p]))
            if mvals[p]: ladder[p]["m"].append(np.mean(mvals[p]))
        log(f"  [{bi+1}/{len(chosen)}] {els}|{nat} done")

    res = {"ckpt": CKPT, "k_bins": len(chosen), "M_data": M_DATA, "M_gen": M_GEN,
           "ladder": {}, "survival": {}}
    print("\n(A) CONVERGENCE LADDER — pass rate (data vs model), easy -> hard")
    print(f"{'check':14s} {'data':>7s} {'model':>7s}")
    for p in LADDER:
        d = float(np.mean(ladder[p]["d"])) if ladder[p]["d"] else float("nan")
        m = float(np.mean(ladder[p]["m"])) if ladder[p]["m"] else float("nan")
        res["ladder"][p] = {"data_rate": d, "model_rate": m}
        print(f"{p:14s} {d:7.2f} {m:7.2f}")

    print("\n(B) BUDGET SURVIVAL — within/between + r^2 (continuous), easy -> hard")
    print(f"{'property':14s} {'bins':>4s} {'T_data':>9s} {'E_data':>9s} {'T_mdl':>9s} {'E_mdl':>9s} {'r^2':>6s}")
    for p in CONTINUOUS:
        c = cont[p]; dm = np.array(c["dm"]); mm = np.array(c["mm"])
        if len(dm) < 3:
            res["survival"][p] = {"n_bins": int(len(dm))}; print(f"{p:14s} too few bins"); continue
        Td, Ed = float(np.mean(c["dv"])), float(np.var(dm))
        Tm, Em = float(np.mean(c["mv"])), float(np.var(mm))
        r2 = float(np.corrcoef(dm, mm)[0, 1] ** 2)
        res["survival"][p] = {"n_bins": int(len(dm)), "T_data": Td, "E_data": Ed,
                              "T_model": Tm, "E_model": Em, "per_bin_mean_r2": r2}
        print(f"{p:14s} {len(dm):4d} {Td:9.4g} {Ed:9.4g} {Tm:9.4g} {Em:9.4g} {r2:6.3f}")
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    log(f"wrote {OUT}")
    print("CLAIM1-SPECTRUM-DONE")


if __name__ == "__main__":
    main()

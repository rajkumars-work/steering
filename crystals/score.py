#!/usr/bin/env python3
"""Compute the Combined (SUN + MSUN) score of generated crystals.

    python score.py generated.extxyz

Combined = SUN + MSUN, over all structures (no validity gate):
  SUN  = stable (e_above_hull <= 0)        & novel (vs LeMat-Bulk) & unique
  MSUN = metastable (0 < e_above_hull <=0.1) & novel & unique

Stability: single MACE single-point energy vs a Materials-Project reference hull
  (references re-evaluated with the same MACE -> self-consistent; cached per chemical
  system). Needs an MP API key (free): export MP_API_KEY=...   and `pip install emmet-core`.
Novelty + uniqueness: lemat-genbench SUNMetric vs LeMat-Bulk (auto-downloads; needs HF token).

This is the single-MACE scorer used for all the steering numbers in the paper. A 3-MLIP
(orb+mace+uma) "lemat-raw" variant is in scripts/verify_combined.py (heavier; UMA is gated).
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "lemat/code/py/dielectric"), os.path.join(HERE, "lemat/code/py/ed"),
          os.path.join(HERE, "lemat/code/py/lemat-genbench/src"),
          "/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
          "/home/ubuntu/packages/lemat-genbench/src"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structures", help="path to an .extxyz of generated crystals")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    from chem.stability import compute_e_above_hull, load_stability_calc
    from lemat_genbench.metrics.sun_metric import SUNMetric

    atoms = read(args.structures, index=":")
    print(f"scoring {len(atoms)} structures (single-MACE + SUNMetric) ...", flush=True)
    calc = load_stability_calc(device=args.device)
    sun = SUNMetric()
    structs = []
    for a in atoms:
        try:
            eah = compute_e_above_hull(a, calc=calc, timeout=120).get("e_above_hull")
        except Exception:
            eah = None
        s = AseAtomsAdaptor.get_structure(a)
        if eah is not None:
            s.properties["e_above_hull_mean"] = float(eah)
        structs.append(s)
    m = sun.compute(structs).metrics
    print("=" * 56)
    print(f"  structures : {len(structs)}")
    print(f"  stable     : {m.get('stable_count')}   (SUN  {m.get('sun_rate'):.3f})")
    print(f"  metastable : {m.get('metastable_count')}   (MSUN {m.get('msun_rate'):.3f})")
    print(f"  COMBINED   : {m.get('combined_sun_msun_rate'):.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the Combined (SUN+MSUN) score of generated crystal structures using the
3-MLIP lemat-genbench hull (orb + mace + uma), the exact scorer behind the published
nxn / bin-pool Combined numbers.

  Combined = SUN + MSUN
    SUN  = stable (e_above_hull <= 0)      & novel (vs LeMat-Bulk) & unique
    MSUN = metastable (0 < e_above_hull<=0.1) & novel & unique
  e_above_hull = mean over the chosen MLIPs (each vs its own reference hull).

Usage:
    HF_TOKEN=<tok> python verify_combined.py <structures.extxyz> [--mlips orb,mace,uma]

Requirements:
  - The bundled `lemat-genbench` installed (see README) — it carries the orb-0.7.0 fix.
  - HF_TOKEN with: (a) read access (LeMat-Bulk novelty reference auto-downloads),
    (b) accepted access to the gated `facebook/UMA` model if using uma
    (request at https://huggingface.co/facebook/UMA). Drop uma from --mlips to skip it.
"""
import argparse
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from lemat_genbench.preprocess.multi_mlip_preprocess import create_multi_mlip_preprocessor
from lemat_genbench.metrics.sun_metric import SUNMetric

# NOTE: orb must be the *_mpa* (MP+Alexandria, PBE-scale) variant — it matches genbench's
# PBE-DFT reference hull. The genbench default `_omat` (OMat24 reference) is mis-calibrated
# here (scores elemental Cu at e_above_hull=0.355 instead of ~0). Verified Cu/NaCl/MgO ≈ 0
# with _mpa. See README "orb calibration".
MLIP_CFG = {
    "orb":  {"model_type": "orb_v3_conservative_inf_mpa", "device": "cuda"},
    "mace": {"model_type": "mp", "device": "cuda"},
    "uma":  {"model_name": "uma-s-1p1", "task": "omat", "device": "cuda"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structures", help="path to an .extxyz of generated structures")
    ap.add_argument("--mlips", default="orb,mace,uma",
                    help="comma-separated subset of orb,mace,uma (default all three)")
    args = ap.parse_args()
    mlips = [m.strip() for m in args.mlips.split(",") if m.strip()]

    atoms = read(args.structures, index=":")
    structs = [AseAtomsAdaptor.get_structure(a) for a in atoms]
    print(f"loaded {len(structs)} structures; scoring with MLIPs={mlips} (n_jobs=1)", flush=True)

    pre = create_multi_mlip_preprocessor(
        mlip_names=mlips, relax_structures=False, n_jobs=1, extract_embeddings=False,
        mlip_configs={m: MLIP_CFG[m] for m in mlips},
    )
    res = pre.run(structs)
    m = SUNMetric().compute(res.processed_structures).metrics
    print("=" * 60)
    print(f"  n scored : {len(res.processed_structures)}")
    print(f"  stable   : {m.get('stable_count')}   (SUN  rate {m.get('sun_rate'):.3f})")
    print(f"  metastab : {m.get('metastable_count')}   (MSUN rate {m.get('msun_rate'):.3f})")
    print(f"  COMBINED : {m.get('combined_sun_msun_rate'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

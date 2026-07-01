"""Repo-relative bootstrap for the review.1 reproduction scripts.

Resolves the checkpoint, distributions, bundled code, and the gap surrogate relative to the repo
root — NO hardcoded working-dir or volatile (/opt/dlami) paths — so a reader can run these from a
fresh clone of `steering/crystals/`. Each driver does e.g.:

    from _repo import CKPT, VERSION, DEVICE, require_train_csv, BG_MODEL, EPS0_PATH, RESULTS, OUTDIR

Prerequisites a reader provides:
  - the checkpoint: run `../fetch_checkpoint.sh` (pulls from HF into checkpoints/alex_nolemat_lowhull/)
  - the training data (DATA side of C-E2/C-E3/C-E4): set CUES_TRAIN_CSV to the alex_nolemat_lowhull
    training CSV (the Alexandria-derived subset; not shipped in this data-light repo).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # steering/crystals/

# bundled code first; live checkout as a fallback (so the original-box scripts still resolve)
for _p in ["lemat/code/py/ed", "lemat/code/py/dielectric", "lemat/code/py/lemat-genbench/src"]:
    _fp = os.path.join(ROOT, _p)
    if os.path.isdir(_fp):
        sys.path.insert(0, _fp)
for _p in ["/home/ubuntu/code/py/ed", "/home/ubuntu/code/py/dielectric",
           "/home/ubuntu/packages/lemat-genbench/src"]:
    if os.path.isdir(_p):
        sys.path.append(_p)


def _first(*ps):
    for p in ps:
        if os.path.exists(p):
            return p
    return ps[0]


CKPT      = os.path.join(ROOT, "checkpoints/alex_nolemat_lowhull")            # via fetch_checkpoint.sh
DIST      = os.path.join(ROOT, "lemat/data/distributions/an_lh.dist.json")
BG_MODEL  = _first(os.path.join(ROOT, "data/xgb_composition_dft_band_gap.json"),
                   "/home/ubuntu/code/py/dielectric/pipeline/data/xgb_composition_dft_band_gap.json")
EPS0_SURROGATE = _first(os.path.join(ROOT, "data/xgb_composition_dft_eps_0.json"),
                       "/home/ubuntu/code/py/dielectric/pipeline/data/xgb_composition_dft_eps_0.json")
EPS0_PATH = _first(os.path.join(ROOT, "lemat/code/py/dielectric/chem/surrogates/eps0.py"),
                   "/home/ubuntu/code/py/dielectric/chem/surrogates/eps0.py")
VERSION   = "d15_binrho_k7"
DEVICE    = os.environ.get("CUES_DEVICE", "cuda")
TRAIN_CSV = os.environ.get("CUES_TRAIN_CSV", "")
RESULTS   = os.path.join(ROOT, "results")
OUTDIR    = HERE


def require_train_csv():
    if not TRAIN_CSV or not os.path.isfile(TRAIN_CSV):
        raise SystemExit(
            "Set CUES_TRAIN_CSV to the alex_nolemat_lowhull training CSV (Alexandria-derived "
            f"subset; not in this data-light repo). Got: {TRAIN_CSV!r}")
    return TRAIN_CSV

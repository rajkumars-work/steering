"""Shared bootstrap for the reproduction experiments.

Resolves every path relative to the repo root and puts the bundled code on sys.path, so the
experiment drivers in this directory run unchanged from a fresh clone. Each driver does:

    from _repo import CKPT, DIST, HIGH_J, LOW_J, VERSION, DEVICE, OUTDIR, out

`import _repo` first inserts the bundled `lemat/code/py/{ed,dielectric,lemat-genbench/src}`
onto sys.path (falling back to a live checkout if you're running on the original box), so the
subsequent `from eval.screening import ...` / `from chem.stability import ...` /
`from lemat_genbench... import ...` all resolve.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # repo root (experiments/ is one level down)

# bundled code first; live checkout as a fallback (so the original-box drivers still work)
for _p in ["lemat/code/py/ed", "lemat/code/py/dielectric", "lemat/code/py/lemat-genbench/src"]:
    _fp = os.path.join(ROOT, _p)
    if os.path.isdir(_fp):
        sys.path.insert(0, _fp)
for _p in ["/home/ubuntu/code/py/ed", "/home/ubuntu/code/py/dielectric",
           "/home/ubuntu/packages/lemat-genbench/src"]:
    if os.path.isdir(_p):
        sys.path.append(_p)

# --- the model, the distribution Counters, and the run config -------------------------------
CKPT    = os.path.join(ROOT, "checkpoints/alex_nolemat_lowhull")            # alex_nolemat_lowhull ep120
DIST    = os.path.join(ROOT, "lemat/data/distributions/an_lh.dist.json")   # full (E,N) Counter (12,917 tuples)
HIGH_J  = os.path.join(ROOT, "lemat/data/distributions/HIGH_rareearth.json")
LOW_J   = os.path.join(ROOT, "lemat/data/distributions/LOW_broad_HPtRh.json")
VERSION = "d15_binrho_k7"
DEVICE  = os.environ.get("CUES_DEVICE", "cuda")
OUTDIR  = os.path.join(ROOT, "experiments/out")
os.makedirs(OUTDIR, exist_ok=True)


def out(*parts):
    """Path under experiments/out/ (created on demand)."""
    p = os.path.join(OUTDIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

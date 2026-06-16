#!/usr/bin/env python
"""Compute training statistics from extxyz files and save to data/training_stats.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase.io import read
from glob import glob


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extxyz",
        default="/data/assets/datasets/dielectric/mp/*.extxyz",
        help="Extxyz file or glob pattern for training statistics.",
    )
    parser.add_argument(
        "--out",
        default="data/training_stats.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    files = sorted(glob(args.extxyz))
    if not files:
        raise SystemExit(f"No extxyz files found for: {args.extxyz}")

    keys = [
        "dft_band_gap",
        "dft_eps_0",
        "mlip_bandgap",
        "surrogate_eps_0",
    ]

    values: Dict[str, List[float]] = {k: [] for k in keys}

    for path in files:
        try:
            atoms_list = read(path, index=":")
        except Exception:
            continue
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for atoms in atoms_list:
            for k in keys:
                v = atoms.info.get(k)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    values[k].append(float(v))

    stats = {k: _summarize(vs) for k, vs in values.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"Wrote stats to {out_path}")
    for k in keys:
        s = stats[k]
        print(
            f"{k}: n={s['n']} mean={s['mean']:.3f} std={s['std']:.3f} "
            f"median={s['median']:.3f} p25={s['p25']:.3f} p75={s['p75']:.3f}"
        )


if __name__ == "__main__":
    main()

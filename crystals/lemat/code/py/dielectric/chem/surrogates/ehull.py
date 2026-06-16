#!/usr/bin/env python3
"""
Compute MLIP energy_above_hull using MLFF energies + a phase diagram.

This uses Atlas PhysicsActor (egip-inf) to get static energies and then
computes energy above hull against a precomputed phase diagram file.

For quick baselines, a SOAP surrogate helper is also provided.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List

import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

from atlas.physics.actor import PhysicsActor
from atlas.physics.config import ConfigEhull
from atlas.physics.ehull import calc_ehull
from atlas.test_utils.ray_test_setup import setup_ray_cluster, shutdown_ray_cluster

from .soap import EXTXYZ_PATH, predict_properties  # noqa: F401


def mlip_energy_above_hull_surrogate(
    structures: Iterable,
    extxyz_path: str = EXTXYZ_PATH,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Predict energy_above_hull for ASE Atoms using the SOAP surrogate model.

    Returns:
        np.ndarray of predicted energy_above_hull (same order as input).
    """
    structures_list: List = list(structures)
    if not structures_list:
        return np.array([], dtype=float)
    preds = predict_properties(
        structures_list,
        model_names=("energy_above_hull",),
        extxyz_path=extxyz_path,
        n_jobs=n_jobs,
    )
    return np.asarray(preds["energy_above_hull"], dtype=float)


def mlip_energy_above_hull(
    structures: Iterable,
    phase_diagram_file: str = "/data/assets/atlas/data/phasediagram.json",
    relax_type: str = "none",
    nrelax: int = 300,
    fmax: float = 1e-3,
    local_gpus: float = 1.0,
    checkpoints_dir: str | None = None,
    shutdown_ray: bool = True,
) -> np.ndarray:
    """
    Compute energy_above_hull using MLFF energies + phase diagram.

    Args:
        structures: Iterable of ASE Atoms.
        phase_diagram_file: Path to precomputed phase diagram JSON/pickle.
        relax_type: "none", "atoms", or "cell".
        nrelax: Max relaxation steps (if relax_type != "none").
        fmax: Force threshold for relaxation.
        local_gpus: GPU fraction for MLFF actor.
        checkpoints_dir: Optional override for ATLAS_CHECKPOINTS_DIR.
        shutdown_ray: Whether to shutdown Ray after calculation.
    """
    structures_list: List = list(structures)
    if not structures_list:
        return np.array([], dtype=float)

    if checkpoints_dir:
        os.environ["ATLAS_CHECKPOINTS_DIR"] = checkpoints_dir

    setup_ray_cluster()
    try:
        actor = PhysicsActor.options(
            num_gpus=local_gpus,
            name="EhullActor",
            namespace="dielectrics",
            get_if_exists=True,
            lifetime="detached",
        ).remote(model_name="egip-inf")

        config = ConfigEhull(
            relax_type=relax_type,
            nrelax=nrelax,
            fmax=fmax,
            phase_diagram_file=phase_diagram_file,
        )

        pmg_structs = [AseAtomsAdaptor.get_structure(a) for a in structures_list]
        results = calc_ehull(pmg_structs, config=config, actor=actor)
        return np.asarray([r.e_above_hull for r in results], dtype=float)
    finally:
        if shutdown_ray:
            shutdown_ray_cluster()


def _load_structures(paths: List[str]) -> List:
    atoms_list: List = []
    for path in paths:
        atoms = ase_read(path, index=":")
        if isinstance(atoms, list):
            atoms_list.extend(atoms)
        else:
            atoms_list.append(atoms)
    return atoms_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict MLIP energy_above_hull for structures"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Structure paths (extxyz)",
    )
    parser.add_argument(
        "--method",
        choices=["mlff", "surrogate"],
        default="mlff",
        help="Energy-above-hull method (default: mlff)",
    )
    parser.add_argument(
        "--extxyz-path",
        default=EXTXYZ_PATH,
        help="Training extxyz path for SOAP species (default: soap.EXTXYZ_PATH)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers (default: all cores)",
    )
    parser.add_argument(
        "--phase-diagram-file",
        default="/data/assets/atlas/data/phasediagram.json",
        help="Phase diagram file for MLFF ehull (default: atlas phasediagram.json)",
    )
    parser.add_argument(
        "--relax-type",
        default="none",
        choices=["none", "atoms", "cell"],
        help="Relaxation mode for MLFF ehull (default: none)",
    )
    parser.add_argument(
        "--nrelax",
        type=int,
        default=300,
        help="Relaxation steps (if relax_type != none)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1e-3,
        help="Relaxation force threshold",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Override ATLAS_CHECKPOINTS_DIR",
    )
    args = parser.parse_args()

    atoms_list = _load_structures(args.paths)
    if args.method == "surrogate":
        preds = mlip_energy_above_hull_surrogate(
            atoms_list, extxyz_path=args.extxyz_path, n_jobs=args.n_jobs
        )
    else:
        preds = mlip_energy_above_hull(
            atoms_list,
            phase_diagram_file=args.phase_diagram_file,
            relax_type=args.relax_type,
            nrelax=args.nrelax,
            fmax=args.fmax,
            checkpoints_dir=args.checkpoints_dir,
        )

    print("path,formula,dft_energy_above_hull,mlip_energy_above_hull")
    idx = 0
    for path in args.paths:
        atoms = ase_read(path, index=":")
        atoms_seq = atoms if isinstance(atoms, list) else [atoms]
        for item in atoms_seq:
            dft_val = item.info.get("energy_above_hull")
            pred_val = float(preds[idx]) if idx < len(preds) else float("nan")
            print(
                f"{path},{item.get_chemical_formula()},{dft_val},{pred_val}"
            )
            idx += 1


if __name__ == "__main__":
    main()

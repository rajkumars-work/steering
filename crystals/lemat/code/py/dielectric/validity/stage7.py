"""Stage 7: physical validity screening of Pareto-front structures."""
from __future__ import annotations

import csv
import os

from ase import Atoms
from ase.io import write as ase_write

from .structural import load_mace, load_pet, run_checks


def run_stage7(
    atoms_list: list[Atoms],
    rows: list[dict],
    output_dir: str,
    device: str = "cuda",
    run_md: bool = False,
    run_cross: bool = True,
) -> tuple[list[Atoms], list[dict]]:
    """Run physical validity checks on relaxed Pareto-front structures.

    Args:
        atoms_list: Relaxed Atoms objects (output of stage6_relax_and_props).
        rows:        Corresponding metadata dicts (one per atoms object).
        output_dir:  Directory for stage7_out.csv and passing structures.
        device:      'cuda' or 'cpu'.
        run_md:      Include NVT-MD stability check (slow, optional).
        run_cross:   Include cross-MLIP energy comparison.

    Returns:
        (passing_atoms, passing_rows) containing only structures that cleared
        all hard-reject checks (uq, ehull, phonon).
    """
    os.makedirs(output_dir, exist_ok=True)
    struct_dir = os.path.join(output_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)

    pet = load_pet(device)
    mace = load_mace(device)
    ehull_cache: dict = {}

    all_rows: list[dict] = []
    passing_atoms: list[Atoms] = []
    passing_rows: list[dict] = []

    for i, (atoms, row) in enumerate(zip(atoms_list, rows)):
        formula = atoms.get_chemical_formula()
        print(f"[stage7] {i + 1}/{len(atoms_list)} {formula}")
        checks = run_checks(atoms, pet, mace, ehull_cache, run_md=run_md, run_cross=run_cross)
        merged = dict(row) | checks
        all_rows.append(merged)
        if checks.get("overall_pass", False):
            passing_atoms.append(atoms)
            passing_rows.append(merged)
            ase_write(os.path.join(struct_dir, f"{i:03d}_{formula}.extxyz"), atoms)

    if all_rows:
        csv_path = os.path.join(output_dir, "stage7_out.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"[stage7] wrote {csv_path}")

    print(f"[stage7] {len(passing_atoms)}/{len(atoms_list)} passed all hard-reject checks")
    return passing_atoms, passing_rows

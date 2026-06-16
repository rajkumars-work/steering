"""Compute primitive-cell atom count from a cached (lattice, atoms_string) pair.

This is the canonical count per polymorph: uniquely defined (via spglib's
standardize_cell with to_primitive=True), robust to supercell / centering
conventions, and distinguishes polymorphs (different structures → different
primitive cells → different atom counts).

Cached atoms_string format (from FieldFormatter.get_atoms_string):
    "Sym Wyckoff_letter x y z Sym Wyckoff_letter x y z ..."
Cached lattice format (from FieldFormatter.get_lattice_string):
    "a b c alpha beta gamma"

Positions are rounded to 2 decimals in the cache; use symprec >= 0.05 so
spglib is robust to that rounding.
"""

from __future__ import annotations

import numpy as np
import spglib
from ase import Atoms


def _atoms_from_cache(lattice_str: str, atoms_str: str) -> Atoms:
    """Reconstruct an ASE Atoms object from cached string fields."""
    lat_vals = [float(x) for x in lattice_str.split()]
    if len(lat_vals) != 6:
        raise ValueError(f"lattice must have 6 values (a b c α β γ), got {len(lat_vals)}")
    a, b, c, al, be, ga = lat_vals

    tokens = atoms_str.split()
    if len(tokens) % 5 != 0:
        raise ValueError(f"atoms_string token count {len(tokens)} not divisible by 5")

    symbols, scaled_positions = [], []
    for i in range(0, len(tokens), 5):
        sym = tokens[i]
        # tokens[i+1] is Wyckoff letter — ignored here
        x, y, z = float(tokens[i + 2]), float(tokens[i + 3]), float(tokens[i + 4])
        symbols.append(sym)
        scaled_positions.append((x, y, z))

    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions,
                  cell=[a, b, c, al, be, ga], pbc=True)
    return atoms


def primitive_natoms(lattice_str: str, atoms_str: str,
                     symprec: float = 0.05) -> int:
    """Return the primitive-cell atom count for a cached structure.

    Uses spglib.standardize_cell(to_primitive=True, no_idealize=True).
    Raises ValueError on unparseable input; RuntimeError if spglib fails.
    """
    atoms = _atoms_from_cache(lattice_str, atoms_str)
    cell = (np.array(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True,
                                   symprec=symprec)
    if prim is None:
        raise RuntimeError("spglib.standardize_cell returned None")
    prim_lattice, prim_positions, prim_numbers = prim
    return len(prim_numbers)


def primitive_natoms_from_atoms(atoms: Atoms, symprec: float = 0.05) -> int:
    """Variant taking an ASE Atoms object directly (e.g., freshly built CIFs)."""
    cell = (np.array(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True,
                                   symprec=symprec)
    if prim is None:
        raise RuntimeError("spglib.standardize_cell returned None")
    return len(prim[2])

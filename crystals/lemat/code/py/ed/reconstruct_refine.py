"""Reconstruct full structures from v3/v4 refine model outputs.

v3 (numbers-only): Model outputs just numbers, we pair them back with
    formula/SG/elements/Wyckoff from the source.
v4 (corrections): Model outputs signed corrections, we apply them to
    the 2-decimal source values.

Usage:
    python reconstruct_refine.py --variant v3 --source refine_full_v3.csv --predictions refine_out_v3.csv --output reconstructed_v3.extxyz
    python reconstruct_refine.py --variant v4 --source refine_full_v4.csv --predictions refine_out_v4.csv --output reconstructed_v4.extxyz
"""
import argparse
import csv
import re
import sys

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import write as ase_write


def parse_source(source_str):
    """Parse a refine source string into components.

    Input: 'refine | Formula | SG N Sym | a b c α β γ | Elem wl x y z ...'
    Returns dict with formula, sg_number, sg_symbol, lattice, atoms list.
    """
    parts = source_str.split(" | ")
    if parts[0] != "refine":
        return None

    formula = parts[1]

    sg_tokens = parts[2].split(None, 2)
    sg_number = int(sg_tokens[1])
    sg_symbol = sg_tokens[2]

    lat_tokens = parts[3].split()
    lattice = [float(x) for x in lat_tokens]  # a b c alpha beta gamma

    atom_tokens = parts[4].split()
    atoms = []
    for j in range(0, len(atom_tokens), 5):
        atoms.append({
            "symbol": atom_tokens[j],
            "wyckoff": atom_tokens[j + 1],
            "x": float(atom_tokens[j + 2]),
            "y": float(atom_tokens[j + 3]),
            "z": float(atom_tokens[j + 4]),
        })

    return {
        "formula": formula,
        "sg_number": sg_number,
        "sg_symbol": sg_symbol,
        "lattice": lattice,
        "atoms": atoms,
    }


def reconstruct_v3(source_info, target_str):
    """Reconstruct from numbers-only output.

    Target: 'a b c α β γ | x1 y1 z1 x2 y2 z2 ...'
    We take numbers from target, elements/Wyckoff from source.
    """
    parts = target_str.strip().split(" | ")
    if len(parts) < 2:
        return None

    lat_tokens = parts[0].split()
    if len(lat_tokens) != 6:
        return None
    lattice = [float(x) for x in lat_tokens]

    coord_tokens = parts[1].split()
    n_atoms = len(source_info["atoms"])
    if len(coord_tokens) != n_atoms * 3:
        return None

    cell = cellpar_to_cell(lattice)
    symbols = []
    frac_coords = []
    wyckoff_letters = []

    for i, atom in enumerate(source_info["atoms"]):
        symbols.append(atom["symbol"])
        wyckoff_letters.append(atom["wyckoff"])
        frac_coords.append([
            float(coord_tokens[i * 3]),
            float(coord_tokens[i * 3 + 1]),
            float(coord_tokens[i * 3 + 2]),
        ])

    atoms_obj = Atoms(
        symbols=symbols,
        scaled_positions=np.array(frac_coords),
        cell=cell,
        pbc=True,
    )
    atoms_obj.info["sg_number"] = source_info["sg_number"]
    atoms_obj.info["sg_symbol"] = source_info["sg_symbol"]
    atoms_obj.arrays["wyckoff_letters"] = np.array(wyckoff_letters)

    # Build target string in standard format for downstream compatibility
    lat_str = " ".join(f"{v:.4f}" if i < 3 else f"{v:.2f}" for i, v in enumerate(lattice))
    atom_parts = []
    for i, atom in enumerate(source_info["atoms"]):
        fx, fy, fz = frac_coords[i]
        atom_parts.append(f"{atom['symbol']} {atom['wyckoff']} {fx:.4f} {fy:.4f} {fz:.4f}")
    target_full = f"{source_info['formula']} | SG {source_info['sg_number']} {source_info['sg_symbol']} | {lat_str} | {' '.join(atom_parts)}"
    atoms_obj.info["target"] = target_full
    atoms_obj.info["source"] = f"refine_v3"

    return atoms_obj


def reconstruct_v4(source_info, target_str):
    """Reconstruct from correction tokens.

    Target: 'Δa Δb Δc Δα Δβ Δγ | Δx1 Δy1 Δz1 Δx2 Δy2 Δz2 ...'
    Corrections are integers: val_4dec = val_2dec*100 + correction
    For lattice lengths: 4dec = round(2dec, 2)*100 + corr, then /10000
    For angles: 2dec = round(1dec, 1)*10 + corr, then /100
    For coords: 4dec = round(2dec, 2)*100 + corr, then /10000
    """
    parts = target_str.strip().split(" | ")
    if len(parts) < 2:
        return None

    lat_corr_tokens = parts[0].split()
    if len(lat_corr_tokens) != 6:
        return None

    coord_corr_tokens = parts[1].split()
    n_atoms = len(source_info["atoms"])
    if len(coord_corr_tokens) != n_atoms * 3:
        return None

    # Parse correction integers
    lat_corrs = [int(t) for t in lat_corr_tokens]
    coord_corrs = [int(t) for t in coord_corr_tokens]

    # Apply corrections to lattice
    lattice = []
    for i in range(3):  # a, b, c
        base_2dec = round(source_info["lattice"][i], 2)
        val_4dec = round(base_2dec * 100) * 100 + lat_corrs[i]
        lattice.append(val_4dec / 10000.0)
    for i in range(3, 6):  # alpha, beta, gamma
        base_1dec = round(source_info["lattice"][i], 1)
        val_2dec = round(base_1dec * 10) * 10 + lat_corrs[i]
        lattice.append(val_2dec / 100.0)

    # Apply corrections to coordinates
    cell = cellpar_to_cell(lattice)
    symbols = []
    frac_coords = []
    wyckoff_letters = []

    for i, atom in enumerate(source_info["atoms"]):
        symbols.append(atom["symbol"])
        wyckoff_letters.append(atom["wyckoff"])
        coords = []
        for j in range(3):
            src_val = [atom["x"], atom["y"], atom["z"]][j]
            base_2dec = round(src_val, 2)
            val_4dec = round(base_2dec * 100) * 100 + coord_corrs[i * 3 + j]
            coords.append(val_4dec / 10000.0)
        frac_coords.append(coords)

    atoms_obj = Atoms(
        symbols=symbols,
        scaled_positions=np.array(frac_coords),
        cell=cell,
        pbc=True,
    )
    atoms_obj.info["sg_number"] = source_info["sg_number"]
    atoms_obj.info["sg_symbol"] = source_info["sg_symbol"]
    atoms_obj.arrays["wyckoff_letters"] = np.array(wyckoff_letters)

    lat_str = " ".join(f"{v:.4f}" if i < 3 else f"{v:.2f}" for i, v in enumerate(lattice))
    atom_parts = []
    for i, atom in enumerate(source_info["atoms"]):
        fx, fy, fz = frac_coords[i]
        atom_parts.append(f"{atom['symbol']} {atom['wyckoff']} {fx:.4f} {fy:.4f} {fz:.4f}")
    target_full = f"{source_info['formula']} | SG {source_info['sg_number']} {source_info['sg_symbol']} | {lat_str} | {' '.join(atom_parts)}"
    atoms_obj.info["target"] = target_full
    atoms_obj.info["source"] = f"refine_v4"

    return atoms_obj


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=["v3", "v4"])
    p.add_argument("--source", required=True, help="Source CSV (from prepare_refine.py)")
    p.add_argument("--predictions", required=True, help="Predictions CSV (from ed_translate.py --csv)")
    p.add_argument("--output", required=True, help="Output extxyz")
    args = p.parse_args()

    # Load sources
    sources = []
    with open(args.source) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["source"])

    # Load predictions
    predictions = []
    with open(args.predictions) as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row["target"])

    if len(sources) != len(predictions):
        print(f"Error: {len(sources)} sources vs {len(predictions)} predictions")
        sys.exit(1)

    reconstruct_fn = reconstruct_v3 if args.variant == "v3" else reconstruct_v4

    atoms_list = []
    n_failed = 0
    n_parse_fail = 0
    for i, (src, pred) in enumerate(zip(sources, predictions)):
        src_info = parse_source(src)
        if src_info is None:
            n_parse_fail += 1
            continue
        atoms = reconstruct_fn(src_info, pred)
        if atoms is not None:
            atoms_list.append(atoms)
        else:
            n_failed += 1

    print(f"Reconstructed {len(atoms_list)}/{len(sources)} structures ({args.variant})")
    if n_failed:
        print(f"  {n_failed} reconstruction failures (wrong token count, etc)")
    if n_parse_fail:
        print(f"  {n_parse_fail} source parse failures")

    ase_write(args.output, atoms_list, format="extxyz")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

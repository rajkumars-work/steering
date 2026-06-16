import json
import os
import re
from pathlib import Path
from ase import Atoms
from ase.geometry import cellpar_to_cell
from .versions import *

def get_dataset_info(csv_path):
    """Identifies the version of a dataset by checking for .json metadata or using heuristics.

    If a `<stem>.binning.json` sidecar sits next to the CSV, its path is
    returned under the `binning_spec_path` key. Consumers (e.g.
    FieldFormatter) can then load it via chem.auto_bin.load_binnings.
    """
    csv_path = Path(csv_path)
    meta_path = csv_path.with_suffix(".json")
    # Auto-bin sidecar: data/<stem>.binning.json
    binning_sidecar = csv_path.with_name(csv_path.stem + ".binning.json")
    binning_path = str(binning_sidecar) if binning_sidecar.exists() else None

    if meta_path.exists():
        with open(meta_path, "r") as f:
            info = json.load(f)
        if binning_path and "binning_spec_path" not in info:
            info["binning_spec_path"] = binning_path
        return info

    # Heuristics based on filename
    stem = csv_path.stem
    for vid in VERSIONS:
        if f"_{vid}" in stem:
            info = {"version_id": vid}
            if binning_path:
                info["binning_spec_path"] = binning_path
            return info

    # Fallback: check header (legacy vs new format)
    import csv
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if "id" in reader.fieldnames and "origin" in reader.fieldnames:
            # Likely d3 or later. Default to d6 if modern but unknown
            info = {"version_id": "d6"}
        else:
            info = {"version_id": "d1"} # Legacy
    if binning_path:
        info["binning_spec_path"] = binning_path
    return info

def parse_target(target_str, version_id, origin="mp"):
    """Parses a target string into an ASE Atoms object using the version specification."""
    spec = get_version(version_id)
    segments = [s.strip() for s in target_str.split("|")]
    
    # Map segment index to field name
    field_map = {i: name for i, name in enumerate(spec.target_segments.get(origin, spec.target_segments["mp"]))}
    
    if len(segments) != len(field_map):
        # Fallback to fuzzy parsing if segment count doesn't match
        return robust_fuzzy_parse(target_str)

    data = {field_map[i]: segments[i] for i in range(len(segments))}
    
    # 1. Extract Lattice
    lattice_str = data.get(T_LATTICE)
    if not lattice_str: return None
    try:
        cell_params = [float(x) for x in lattice_str.split()]
        cell = cellpar_to_cell(cell_params)
    except: return None

    # 2. Extract Atoms
    atoms_str = data.get(T_ATOMS)
    if not atoms_str: return None
    
    tokens = atoms_str.split()
    symbols, coords, wls = [], [], []
    
    # Determine if it has Wyckoff letters (standard in all versions d1-d6)
    # Format: Sym WL x y z (5 tokens per atom)
    has_wl = True 
    step = 5 if has_wl else 4
    
    for j in range(0, len(tokens), step):
        try:
            sym = tokens[j]
            if has_wl:
                wl = tokens[j+1]
                xyz = [float(tokens[j+2]), float(tokens[j+3]), float(tokens[j+4])]
                wls.append(wl)
            else:
                xyz = [float(tokens[j+1]), float(tokens[j+2]), float(tokens[j+3])]
            symbols.append(sym)
            coords.append(xyz)
        except: break

    atoms = Atoms(symbols=symbols, scaled_positions=coords, cell=cell, pbc=True)
    if wls: atoms.info["wyckoff_letters"] = wls

    # 3. Add other info — always store GCD-reduced formula
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms.info["formula"] = AseAtomsAdaptor.get_structure(atoms).composition.reduced_formula
    except Exception:
        if T_FORMULA in data:
            atoms.info["formula"] = data[T_FORMULA]
    if T_EF in data and data[T_EF].startswith("Ef:"):
        try: atoms.info["ef"] = float(data[T_EF][3:])
        except: pass
        
    return atoms

def robust_fuzzy_parse(target_str):
    """Fallback fuzzy parser when version-based parsing fails."""
    # Similar to the one in screen_checkpoint.py but consolidated here
    parts = [p.strip() for p in target_str.split("|")]
    cell = None
    for p in parts:
        tokens = p.split()
        if len(tokens) == 6:
            try:
                vals = [float(t) for t in tokens]
                if all(0.1 < v < 180 for v in vals):
                    cell = cellpar_to_cell(vals)
                    break
            except: continue
    if cell is None: return None
    
    for p in parts:
        tokens = p.split()
        if len(tokens) < 5: continue
        symbols, coords, wls = [], [], []
        for j in range(0, len(tokens) - 4, 5):
            try:
                sym = tokens[j]
                wl = tokens[j+1]
                xyz = [float(tokens[j+2]), float(tokens[j+3]), float(tokens[j+4])]
                symbols.append(sym)
                coords.append(xyz)
                wls.append(wl)
            except: break
        if symbols:
            atoms = Atoms(symbols=symbols, scaled_positions=coords, cell=cell, pbc=True)
            atoms.info["wyckoff_letters"] = wls
            return atoms
    return None

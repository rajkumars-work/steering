#!/usr/bin/env python3
"""Curate DPO preference pairs from crystal structure databases (MP, Alexandria, JARVIS).

Produces a CSV with (source, chosen, rejected, score) columns compatible with
dpo_data.py.  Three pairing strategies:

  1. Stability pairs      — same formula, different e_above_hull bins
  2. Structural quality    — same formula, one passes / one fails structural checks
  3. Cross-database pairs  — same formula across different databases with stability gap

Usage:
    python dpo_curate.py --databases mp alex jarvis \
        --output /data/rkumar/code/py/dielectric/data/dpo_curated.csv \
        --strategies stability structural cross_db \
        --min_ehull_gap 0.025 --max_pairs_per_formula 5 --workers 8
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from multiprocessing import Pool
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list as ase_neighbor_list
from tqdm import tqdm

# Import formatting utilities from make_datasets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dielectric", "scripts"))
from make_datasets import (
    bin_band_gap,
    bin_dielectric,
    bin_refractive,
    bin_stability,
    compute_spg,
    crystal_system,
    format_source_l1,
    format_source_l2,
    format_target,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

MP_GGA_DIR = "/data/assets/atlas/data/MP/GGA"
MP_ELECTRIC_DIR = "/data/assets/atlas/data/MP/electric"
ALEX_DIR = "/data/assets/atlas/data/Alex/data"
JARVIS_PATH = "/data/assets/atlas/jarvis/jdft_3d-12-12-2022.json"

DEFAULT_OUTPUT = "/data/rkumar/code/py/dielectric/data/dpo_curated.csv"


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------

@dataclass
class Record:
    formula: str                        # reduced formula (for grouping)
    e_above_hull: Optional[float]       # eV/atom
    band_gap: Optional[float]           # eV
    eps_0: Optional[float]
    refractive_index: Optional[float]
    source_l2: str                      # full source string (prompt)
    target: str                         # target string (completion)
    database: str                       # "mp", "alex", "jarvis"
    struct_checks: Optional[dict] = field(default=None)  # {check_name: pass/fail}


# ---------------------------------------------------------------------------
# Anonymous formula helper
# ---------------------------------------------------------------------------

def _anonymous_formula(numbers):
    """Compute anonymous formula from atomic numbers (e.g. A2B3C)."""
    count = Counter(numbers)
    sorted_counts = sorted(count.values(), reverse=True)
    g = reduce(gcd, sorted_counts)
    reduced_counts = sorted([c // g for c in sorted_counts], reverse=True)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for i, c in enumerate(reduced_counts):
        if c == 1:
            parts.append(letters[i])
        else:
            parts.append(f"{letters[i]}{c}")
    return "".join(parts)


def _reduced_formula(numbers):
    """Compute reduced formula string from atomic numbers."""
    atoms = Atoms(numbers=numbers)
    return atoms.get_chemical_formula(mode="metal")


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def _run_cheap_checks(atoms):
    """Run fast structural checks (neighbor-list based). Returns dict {name: bool}."""
    checks = {}

    # Min interatomic distance (< 0.5 Å is unphysical)
    try:
        dists = ase_neighbor_list("d", atoms, cutoff=0.5, self_interaction=False)
        checks["min_dist"] = len(dists) == 0
    except Exception:
        checks["min_dist"] = True

    # Isolated atoms (no neighbor within 4 Å)
    try:
        i_list = ase_neighbor_list("i", atoms, cutoff=4.0, self_interaction=False)
        connected = set(i_list)
        checks["isolated_atoms"] = len(connected) >= len(atoms)
    except Exception:
        checks["isolated_atoms"] = True

    return checks


def _cheap_checks_pass(checks):
    """Return True if all cheap structural checks passed."""
    if checks is None:
        return False
    return all(checks.values())


def _cheap_checks_fail(checks):
    """Return True if any cheap structural check failed."""
    if checks is None:
        return False
    return not all(checks.values())


# ---------------------------------------------------------------------------
# MP GGA loader
# ---------------------------------------------------------------------------

def _load_mp_electric():
    """Load MP electric data into dict keyed by material_id."""
    electric = {}
    files = glob.glob(os.path.join(MP_ELECTRIC_DIR, "*.extxyz"))
    for f in tqdm(files, desc="MP electric", unit="file"):
        try:
            atoms = ase_read(f)
            mid = atoms.info.get("material_id", "")
            if mid:
                electric[mid] = {
                    "eps_0": atoms.info.get("eps_0"),
                    "refractive_index": atoms.info.get("refractive_index"),
                    "band_gap": atoms.info.get("band_gap"),
                    "e_above_hull": atoms.info.get("energy_above_hull"),
                }
        except Exception:
            continue
    return electric


# Global for multiprocessing (set before Pool.map)
_mp_electric_data = {}


def _process_one_mp(filepath):
    """Process a single MP GGA extxyz file. Returns Record or None."""
    try:
        atoms = ase_read(filepath)
        info = atoms.info

        material_id = info.get("material_id", "")
        band_gap = info.get("band_gap")
        e_hull = info.get("energy_above_hull")

        numbers = atoms.numbers
        cell_array = atoms.cell.array
        positions = atoms.positions
        scaled_pos = atoms.get_scaled_positions()

        spg_num, spg_sym = compute_spg(cell_array, scaled_pos, numbers)
        csys = crystal_system(spg_num)

        anon_formula = _anonymous_formula(numbers)
        formula = _reduced_formula(numbers)
        elements = sorted(set(chemical_symbols[n] for n in numbers))
        natoms = len(numbers)

        mass_amu = atoms.get_masses().sum()
        vol = atoms.get_volume()
        if vol < 1e-6:
            return None
        density = mass_amu * 1.66054 / vol

        # Merge electric properties
        elec = _mp_electric_data.get(material_id, {})
        eps_0 = elec.get("eps_0")
        refractive_index = elec.get("refractive_index")

        target = format_target(numbers, cell_array, positions)
        source_l1 = format_source_l1(csys, spg_sym, anon_formula, elements, natoms, density)
        source_l2 = format_source_l2(source_l1, band_gap, e_hull, eps_0, refractive_index)

        struct_checks = _run_cheap_checks(atoms)

        return Record(
            formula=formula,
            e_above_hull=float(e_hull) if e_hull is not None else None,
            band_gap=float(band_gap) if band_gap is not None else None,
            eps_0=float(eps_0) if eps_0 is not None else None,
            refractive_index=float(refractive_index) if refractive_index is not None else None,
            source_l2=source_l2,
            target=target,
            database="mp",
            struct_checks=struct_checks,
        )
    except Exception:
        return None


def _init_mp_worker(electric_data):
    """Initializer for MP worker processes — set global electric data."""
    global _mp_electric_data
    _mp_electric_data = electric_data


def load_mp(workers=8, max_files=0):
    """Load MP GGA structures, merging electric properties. Returns list[Record]."""
    electric_data = _load_mp_electric()

    files = sorted(glob.glob(os.path.join(MP_GGA_DIR, "*.extxyz")))
    if max_files > 0:
        files = files[:max_files]
    print(f"MP GGA: processing {len(files):,} files")

    records = []
    with Pool(workers, initializer=_init_mp_worker, initargs=(electric_data,)) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one_mp, files, chunksize=64),
            total=len(files),
            desc="MP GGA",
            unit="file",
        ):
            if result is not None:
                records.append(result)

    print(f"MP GGA: loaded {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# Alexandria loader
# ---------------------------------------------------------------------------

def _process_one_alex(filepath):
    """Process a single Alexandria extxyz file. Returns Record or None."""
    try:
        atoms = ase_read(filepath)
        info = atoms.info

        e_hull = info.get("energy_above_hull")

        numbers = atoms.numbers
        cell_array = atoms.cell.array
        positions = atoms.positions
        scaled_pos = atoms.get_scaled_positions()

        spg_num, spg_sym = compute_spg(cell_array, scaled_pos, numbers)
        csys = crystal_system(spg_num)

        anon_formula = _anonymous_formula(numbers)
        formula = _reduced_formula(numbers)
        elements = sorted(set(chemical_symbols[n] for n in numbers))
        natoms = len(numbers)

        mass_amu = atoms.get_masses().sum()
        vol = atoms.get_volume()
        if vol < 1e-6:
            return None
        density = mass_amu * 1.66054 / vol

        target = format_target(numbers, cell_array, positions)
        source_l1 = format_source_l1(csys, spg_sym, anon_formula, elements, natoms, density)
        # Alexandria has no band_gap, eps_0, refractive_index → only stability label
        source_l2 = format_source_l2(source_l1, None, e_hull, None, None)

        struct_checks = _run_cheap_checks(atoms)

        return Record(
            formula=formula,
            e_above_hull=float(e_hull) if e_hull is not None else None,
            band_gap=None,
            eps_0=None,
            refractive_index=None,
            source_l2=source_l2,
            target=target,
            database="alex",
            struct_checks=struct_checks,
        )
    except Exception:
        return None


def load_alex(workers=8, max_files=0):
    """Load Alexandria structures. Returns list[Record]."""
    files = sorted(glob.glob(os.path.join(ALEX_DIR, "*.extxyz")))
    if max_files > 0:
        files = files[:max_files]
    print(f"Alexandria: processing {len(files):,} files")

    records = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one_alex, files, chunksize=64),
            total=len(files),
            desc="Alexandria",
            unit="file",
        ):
            if result is not None:
                records.append(result)

    print(f"Alexandria: loaded {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# JARVIS loader
# ---------------------------------------------------------------------------

def load_jarvis(max_entries=0):
    """Load JARVIS-DFT 3D structures. Returns list[Record]."""
    print(f"JARVIS: loading {JARVIS_PATH}")
    with open(JARVIS_PATH, "r") as f:
        entries = json.load(f)
    if max_entries > 0:
        entries = entries[:max_entries]
    print(f"JARVIS: processing {len(entries):,} entries")

    records = []
    skipped = 0

    for entry in tqdm(entries, desc="JARVIS", unit="entry"):
        try:
            atoms_dict = entry["atoms"]
            lattice = np.array(atoms_dict["lattice_mat"], dtype=float)
            coords = np.array(atoms_dict["coords"], dtype=float)
            elements = atoms_dict["elements"]
            is_cartesian = atoms_dict.get("cartesian", True)

            numbers = [chemical_symbols.index(e) for e in elements]

            if is_cartesian:
                atoms = Atoms(numbers=numbers, positions=coords, cell=lattice, pbc=True)
            else:
                atoms = Atoms(numbers=numbers, scaled_positions=coords, cell=lattice, pbc=True)

            # Use pre-computed spacegroup from JARVIS (skip spglib)
            spg_num_str = entry.get("spg_number", "1")
            try:
                spg_num = int(spg_num_str)
            except (ValueError, TypeError):
                spg_num = 1
            csys_str = entry.get("crys", "")
            if not csys_str or csys_str == "na":
                csys_str = crystal_system(spg_num)
            spg_sym = entry.get("spg_symbol", "")
            if not spg_sym or spg_sym == "na":
                spg_sym = entry.get("spg", "P1")
                if not spg_sym or spg_sym == "na":
                    spg_sym = "P1"

            anon_formula = _anonymous_formula(numbers)
            formula = _reduced_formula(numbers)
            elem_set = sorted(set(elements))
            natoms = len(numbers)

            mass_amu = atoms.get_masses().sum()
            vol = atoms.get_volume()
            if vol < 1e-6:
                skipped += 1
                continue
            density = mass_amu * 1.66054 / vol

            # Properties
            e_hull_raw = entry.get("ehull", "na")
            e_hull = float(e_hull_raw) if e_hull_raw != "na" and e_hull_raw is not None else None

            bg_raw = entry.get("optb88vdw_bandgap", "na")
            band_gap = float(bg_raw) if bg_raw != "na" and bg_raw is not None else None

            # Dielectric: average of epsx, epsy, epsz
            eps_vals = []
            for k in ("epsx", "epsy", "epsz"):
                v = entry.get(k, "na")
                if v != "na" and v is not None:
                    try:
                        eps_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            eps_0 = float(np.mean(eps_vals)) if eps_vals else None

            refractive_index = np.sqrt(eps_0) if eps_0 is not None and eps_0 > 0 else None

            target = format_target(numbers, lattice, atoms.positions)
            source_l1 = format_source_l1(csys_str, spg_sym, anon_formula, elem_set, natoms, density)
            source_l2 = format_source_l2(source_l1, band_gap, e_hull, eps_0, refractive_index)

            struct_checks = _run_cheap_checks(atoms)

            records.append(Record(
                formula=formula,
                e_above_hull=e_hull,
                band_gap=band_gap,
                eps_0=eps_0,
                refractive_index=refractive_index,
                source_l2=source_l2,
                target=target,
                database="jarvis",
                struct_checks=struct_checks,
            ))
        except Exception:
            skipped += 1
            continue

    print(f"JARVIS: loaded {len(records):,} records, skipped {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Pairing strategies
# ---------------------------------------------------------------------------

def pair_by_stability(records_by_formula, min_gap=0.025, max_per_formula=5):
    """Strategy 1: Pair structures with different stability bins.

    Yields (source, chosen, rejected, score) where chosen is more stable.
    """
    pairs = []

    for formula, recs in records_by_formula.items():
        # Filter to records with valid e_above_hull
        valid = [r for r in recs if r.e_above_hull is not None]
        if len(valid) < 2:
            continue

        # Sort by e_above_hull (most stable first)
        valid.sort(key=lambda r: r.e_above_hull)

        formula_pairs = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                chosen = valid[i]
                rejected = valid[j]

                gap = rejected.e_above_hull - chosen.e_above_hull
                if gap < min_gap:
                    continue

                # Must be in different stability bins
                chosen_bin = bin_stability(chosen.e_above_hull)
                rejected_bin = bin_stability(rejected.e_above_hull)
                if chosen_bin == rejected_bin:
                    continue

                score = f"stability_{rejected_bin}_vs_{chosen_bin}"
                formula_pairs.append((chosen.source_l2, chosen.target, rejected.target, score, gap))

            if len(formula_pairs) >= max_per_formula:
                break

        # Take top pairs by gap size
        formula_pairs.sort(key=lambda x: x[4], reverse=True)
        for src, cho, rej, sc, _ in formula_pairs[:max_per_formula]:
            pairs.append((src, cho, rej, sc))

    return pairs


def pair_by_structural_quality(records_by_formula, max_per_formula=5):
    """Strategy 2: Pair structures where one passes all cheap structural checks and one fails.

    Uses min_dist and isolated_atoms checks (fast neighbor-list based).
    Yields (source, chosen, rejected, score).
    """
    pairs = []

    for formula, recs in records_by_formula.items():
        passing = [r for r in recs if _cheap_checks_pass(r.struct_checks)]
        failing = [r for r in recs if _cheap_checks_fail(r.struct_checks)]

        if not passing or not failing:
            continue

        formula_pairs = []
        for chosen in passing:
            for rejected in failing:
                # Identify which checks failed
                failed = [k for k, v in rejected.struct_checks.items() if not v]
                score = "+".join(sorted(failed))
                formula_pairs.append((chosen.source_l2, chosen.target, rejected.target, score))

                if len(formula_pairs) >= max_per_formula:
                    break
            if len(formula_pairs) >= max_per_formula:
                break

        pairs.extend(formula_pairs[:max_per_formula])

    return pairs


def pair_cross_database(records_by_formula, min_gap=0.025, max_per_formula=5):
    """Strategy 3: Pair structures from different databases with stability gap.

    Yields (source, chosen, rejected, score).
    """
    pairs = []

    for formula, recs in records_by_formula.items():
        # Filter to records with valid e_above_hull from multiple databases
        valid = [r for r in recs if r.e_above_hull is not None]
        databases = set(r.database for r in valid)
        if len(databases) < 2:
            continue

        # Sort by e_above_hull
        valid.sort(key=lambda r: r.e_above_hull)

        formula_pairs = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                chosen = valid[i]
                rejected = valid[j]

                # Must be from different databases
                if chosen.database == rejected.database:
                    continue

                gap = rejected.e_above_hull - chosen.e_above_hull
                if gap < min_gap:
                    continue

                score = f"cross_db_stability_{chosen.database}_vs_{rejected.database}"
                formula_pairs.append((chosen.source_l2, chosen.target, rejected.target, score, gap))

            if len(formula_pairs) >= max_per_formula:
                break

        formula_pairs.sort(key=lambda x: x[4], reverse=True)
        for src, cho, rej, sc, _ in formula_pairs[:max_per_formula]:
            pairs.append((src, cho, rej, sc))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Curate DPO preference pairs from crystal databases")
    parser.add_argument("--databases", nargs="+", default=["mp", "alex", "jarvis"],
                        choices=["mp", "alex", "jarvis"],
                        help="Which databases to load")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output CSV path")
    parser.add_argument("--strategies", nargs="+", default=["stability", "structural", "cross_db"],
                        choices=["stability", "structural", "cross_db"],
                        help="Pairing strategies to use")
    parser.add_argument("--min_ehull_gap", type=float, default=0.025,
                        help="Min e_above_hull gap for stability pairs (eV/atom)")
    parser.add_argument("--max_pairs_per_formula", type=int, default=5,
                        help="Max pairs per formula per strategy")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers for multiprocessing")
    parser.add_argument("-n", type=int, default=0,
                        help="Limit files/entries loaded per database (0 = all, for testing)")
    args = parser.parse_args()

    t0 = time.time()

    # --- Load databases ---
    all_records = []

    if "mp" in args.databases:
        print("\n=== Loading MP GGA ===")
        all_records.extend(load_mp(workers=args.workers, max_files=args.n))

    if "alex" in args.databases:
        print("\n=== Loading Alexandria ===")
        all_records.extend(load_alex(workers=args.workers, max_files=args.n))

    if "jarvis" in args.databases:
        print("\n=== Loading JARVIS ===")
        all_records.extend(load_jarvis(max_entries=args.n))

    print(f"\nTotal records loaded: {len(all_records):,}")

    # --- Group by reduced formula ---
    records_by_formula = defaultdict(list)
    for rec in all_records:
        records_by_formula[rec.formula].append(rec)

    n_formulas = len(records_by_formula)
    n_multi = sum(1 for recs in records_by_formula.values() if len(recs) > 1)
    print(f"Unique formulas: {n_formulas:,}")
    print(f"Formulas with >1 polymorph: {n_multi:,}")

    # --- Run pairing strategies ---
    all_pairs = []

    if "stability" in args.strategies:
        print("\n=== Strategy: Stability pairs ===")
        t1 = time.time()
        pairs = pair_by_stability(records_by_formula, args.min_ehull_gap, args.max_pairs_per_formula)
        print(f"  Stability pairs: {len(pairs):,} ({time.time() - t1:.1f}s)")
        all_pairs.extend(pairs)

    if "structural" in args.strategies:
        print("\n=== Strategy: Structural quality pairs ===")
        t1 = time.time()
        pairs = pair_by_structural_quality(records_by_formula, args.max_pairs_per_formula)
        print(f"  Structural quality pairs: {len(pairs):,} ({time.time() - t1:.1f}s)")
        all_pairs.extend(pairs)

    if "cross_db" in args.strategies:
        print("\n=== Strategy: Cross-database pairs ===")
        t1 = time.time()
        pairs = pair_cross_database(records_by_formula, args.min_ehull_gap, args.max_pairs_per_formula)
        print(f"  Cross-database pairs: {len(pairs):,} ({time.time() - t1:.1f}s)")
        all_pairs.extend(pairs)

    # --- Deduplicate ---
    seen = set()
    unique_pairs = []
    for src, cho, rej, sc in all_pairs:
        key = (cho, rej)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((src, cho, rej, sc))

    print(f"\nTotal pairs before dedup: {len(all_pairs):,}")
    print(f"Total pairs after dedup:  {len(unique_pairs):,}")

    # --- Write output ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "chosen", "rejected", "score"])
        for src, cho, rej, sc in unique_pairs:
            writer.writerow([src, cho, rej, sc])

    # --- Summary ---
    score_counts = Counter(sc for _, _, _, sc in unique_pairs)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total unique pairs: {len(unique_pairs):,}")
    print(f"  Output: {args.output}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"\nScore distribution:")
    for score, count in sorted(score_counts.items(), key=lambda x: -x[1]):
        print(f"  {score}: {count:,}")


if __name__ == "__main__":
    main()

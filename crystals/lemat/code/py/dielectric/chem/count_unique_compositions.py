#!/usr/bin/env python3
"""
Count unique compositions in the dataset.
"""

from pathlib import Path
from collections import defaultdict
import re
from pymatgen.core import Composition

# Configuration
DATA_DIR = "/data/assets/datasets/dielectric/mp"


def extract_composition_from_filename(filename):
    """Extract composition from filename."""
    # Remove .extxyz extension
    name = filename.replace('.extxyz', '')

    # Handle two formats:
    # 1. mp-XXXXX-mp.extxyz - need to extract from metadata
    # 2. Composition-hash-source.extxyz (e.g., Ag1Bi1Mo2O8-61eef0bac6f3fe83b67b1e57-rg.extxyz)

    if name.startswith('mp-'):
        # For mp files, return None to signal we need to read from file
        return None
    else:
        # First part before '-' is the composition
        parts = name.split('-')
        return parts[0] if parts else None


def normalize_composition(comp_str):
    """Normalize composition using pymatgen to handle different orderings."""
    try:
        comp = Composition(comp_str)
        # Get reduced formula (removes common factors)
        return comp.reduced_formula
    except:
        # Fallback: just return as-is if parsing fails
        return comp_str


def extract_composition_from_metadata(file_path):
    """Extract composition from the atoms in the file."""
    try:
        from ase.io import read
        atoms = read(file_path, index=0)
        formula = atoms.get_chemical_formula(mode='hill')
        return formula
    except:
        return None


def count_unique_compositions():
    """Count unique compositions in the dataset."""

    data_path = Path(DATA_DIR)
    extxyz_files = list(data_path.glob("*.extxyz"))

    print(f"Analyzing {len(extxyz_files)} files...")
    print()

    # Dictionary to track compositions and their file counts
    composition_counts = defaultdict(list)
    failed_parsing = []

    for i, file_path in enumerate(extxyz_files, 1):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(extxyz_files)} files...")

        comp_str = extract_composition_from_filename(file_path.name)

        # If filename parsing failed (e.g., mp- files), read from file
        if comp_str is None:
            comp_str = extract_composition_from_metadata(file_path)

        if comp_str:
            normalized_comp = normalize_composition(comp_str)
            composition_counts[normalized_comp].append(file_path.name)
        else:
            failed_parsing.append(file_path.name)

    # Print results
    print("="*70)
    print("UNIQUE COMPOSITION ANALYSIS")
    print("="*70)
    print(f"Total files analyzed:           {len(extxyz_files)}")
    print(f"Unique compositions found:      {len(composition_counts)}")
    print(f"Failed to parse:                {len(failed_parsing)}")
    print("="*70)

    # Show composition distribution
    print("\nComposition frequency distribution:")
    freq_dist = defaultdict(int)
    for comp, files in composition_counts.items():
        freq_dist[len(files)] += 1

    for count in sorted(freq_dist.keys()):
        num_comps = freq_dist[count]
        print(f"  {count} file(s):  {num_comps} compositions")

    # Show top 10 most common compositions
    print("\nTop 10 most common compositions:")
    sorted_comps = sorted(composition_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (comp, files) in enumerate(sorted_comps[:10], 1):
        print(f"  {i:2d}. {comp:20s} - {len(files)} structures")

    # Show some examples of compositions with multiple structures
    print("\nExample: Compositions with multiple structures:")
    multi_structure_comps = [(comp, files) for comp, files in sorted_comps if len(files) > 1][:5]
    for comp, files in multi_structure_comps:
        print(f"\n  {comp} ({len(files)} structures):")
        for fname in files[:3]:  # Show first 3 files
            print(f"    - {fname}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

    print("\n" + "="*70)

    return composition_counts, failed_parsing


if __name__ == "__main__":
    compositions, failed = count_unique_compositions()

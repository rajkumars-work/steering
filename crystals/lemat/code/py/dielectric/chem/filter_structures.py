#!/usr/bin/env python3
"""
Filter dielectric structures based on bandgap and energy_above_hull criteria.
Copies structures from source to destination directory.
"""

import os
import shutil
from pathlib import Path
import re

# Configuration
SOURCE_DIR = "/data/assets/atlas/dielectrics/DATABASE/db-mp_updated/"
DEST_DIR = "/data/assets/datasets/dielectric/mp"
BANDGAP_MIN = 0.1
ENERGY_ABOVE_HULL_MAX = 0.2


def parse_metadata(line):
    """Parse metadata from the second line of an extxyz file."""
    metadata = {}

    # Extract band_gap value
    bandgap_match = re.search(r'band_gap=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
    if bandgap_match:
        metadata['band_gap'] = float(bandgap_match.group(1))

    # Extract energy_above_hull value
    energy_match = re.search(r'energy_above_hull=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
    if energy_match:
        metadata['energy_above_hull'] = float(energy_match.group(1))

    return metadata


def filter_and_copy_structures():
    """Filter structures based on criteria and copy to destination."""

    # Create destination directory
    os.makedirs(DEST_DIR, exist_ok=True)

    # Get all extxyz files
    source_path = Path(SOURCE_DIR)
    extxyz_files = list(source_path.glob("*.extxyz"))

    print(f"Found {len(extxyz_files)} files to process...")

    copied_count = 0
    skipped_count = 0
    error_count = 0

    for i, file_path in enumerate(extxyz_files, 1):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(extxyz_files)} files... (Copied: {copied_count}, Skipped: {skipped_count})")

        try:
            # Read the second line to get metadata
            with open(file_path, 'r') as f:
                f.readline()  # Skip first line (atom count)
                metadata_line = f.readline()

            # Parse metadata
            metadata = parse_metadata(metadata_line)

            # Check if we have the required fields
            if 'band_gap' not in metadata or 'energy_above_hull' not in metadata:
                error_count += 1
                print(f"Warning: Missing metadata in {file_path.name}")
                continue

            bandgap = metadata['band_gap']
            energy_above_hull = metadata['energy_above_hull']

            # Apply filters
            if bandgap > BANDGAP_MIN and energy_above_hull < ENERGY_ABOVE_HULL_MAX:
                # Copy file to destination
                dest_file = Path(DEST_DIR) / file_path.name
                shutil.copy2(file_path, dest_file)
                copied_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error processing {file_path.name}: {str(e)}")

    # Print summary
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"Total files processed: {len(extxyz_files)}")
    print(f"Files copied (meeting criteria): {copied_count}")
    print(f"Files skipped (not meeting criteria): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nCriteria applied:")
    print(f"  - band_gap > {BANDGAP_MIN}")
    print(f"  - energy_above_hull < {ENERGY_ABOVE_HULL_MAX}")
    print(f"\nDestination directory: {DEST_DIR}")
    print("="*60)


if __name__ == "__main__":
    filter_and_copy_structures()

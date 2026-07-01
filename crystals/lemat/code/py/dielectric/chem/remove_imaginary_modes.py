#!/usr/bin/env python3
"""
Remove structures with imaginary phonon modes (has_imaginary_modes=T) from the dataset.
"""

import os
import shutil
from pathlib import Path
import re

# Configuration
DATA_DIR = "/data/assets/datasets/dielectric/mp"
BACKUP_DIR = "/data/assets/datasets/dielectric/mp_backup_imaginary_modes"


def has_imaginary_modes(file_path):
    """Check if a structure has imaginary phonon modes."""
    try:
        with open(file_path, 'r') as f:
            f.readline()  # Skip first line (atom count)
            metadata_line = f.readline()

        # Check for has_imaginary_modes=T
        return "has_imaginary_modes=T" in metadata_line
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def remove_imaginary_mode_structures():
    """Remove structures with imaginary modes from the dataset."""

    # Create backup directory
    os.makedirs(BACKUP_DIR, exist_ok=True)
    print(f"Backup directory created: {BACKUP_DIR}")

    # Get all extxyz files
    data_path = Path(DATA_DIR)
    extxyz_files = list(data_path.glob("*.extxyz"))

    print(f"\nFound {len(extxyz_files)} total files in dataset...")
    print("Scanning for structures with imaginary modes...\n")

    removed_count = 0
    kept_count = 0
    error_count = 0

    for i, file_path in enumerate(extxyz_files, 1):
        if i % 500 == 0:
            print(f"Processed {i}/{len(extxyz_files)} files... (Removed: {removed_count}, Kept: {kept_count})")

        try:
            if has_imaginary_modes(file_path):
                # Move file to backup (safer than deleting)
                backup_file = Path(BACKUP_DIR) / file_path.name
                shutil.move(str(file_path), str(backup_file))
                removed_count += 1
            else:
                kept_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error processing {file_path.name}: {str(e)}")

    # Print summary
    print("\n" + "="*70)
    print("IMAGINARY MODE REMOVAL COMPLETE")
    print("="*70)
    print(f"Total files processed:                    {len(extxyz_files)}")
    print(f"Files removed (has_imaginary_modes=T):    {removed_count}")
    print(f"Files kept (has_imaginary_modes=F):       {kept_count}")
    print(f"Errors encountered:                       {error_count}")
    print(f"\nRemoved files backed up to: {BACKUP_DIR}")
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Remaining structures: {kept_count}")
    print("="*70)


if __name__ == "__main__":
    # Confirm action
    print("="*70)
    print("IMAGINARY PHONON MODE REMOVAL")
    print("="*70)
    print(f"This script will remove structures with has_imaginary_modes=T")
    print(f"from the dataset at: {DATA_DIR}")
    print(f"\nFiles will be moved (not deleted) to: {BACKUP_DIR}")
    print("="*70)

    response = input("\nProceed? (yes/no): ").strip().lower()
    if response == 'yes':
        remove_imaginary_mode_structures()
    else:
        print("Operation cancelled.")

#!/usr/bin/env python3
"""
Create a random sample of structures for testing.
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "/data/assets/datasets/dielectric/mp"
DEST_DIR = "./data"
SAMPLE_SIZE = 100

def sample_structures():
    """Randomly sample structures and copy to data directory."""

    # Create destination directory
    os.makedirs(DEST_DIR, exist_ok=True)

    # Get all extxyz files
    source_path = Path(SOURCE_DIR)
    all_files = list(source_path.glob("*.extxyz"))

    print(f"Found {len(all_files)} files in source directory")

    # Randomly sample files
    sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    print(f"Randomly selected {len(sample_files)} files to copy...")

    # Copy sampled files
    for i, file_path in enumerate(sample_files, 1):
        dest_file = Path(DEST_DIR) / file_path.name
        shutil.copy2(file_path, dest_file)
        if i % 20 == 0:
            print(f"  Copied {i}/{len(sample_files)} files...")

    print(f"\n✓ Successfully copied {len(sample_files)} files to {DEST_DIR}/")
    print(f"  Directory size: {sum(f.stat().st_size for f in Path(DEST_DIR).glob('*.extxyz')) / 1024:.1f} KB")

if __name__ == "__main__":
    sample_structures()

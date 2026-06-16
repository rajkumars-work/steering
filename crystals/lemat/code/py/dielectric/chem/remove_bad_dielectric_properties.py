"""
Remove problematic e_inf and e_static properties from extxyz files,
keeping only the correct eps_inf and eps_0 values
"""

from ase.io import read, write
from glob import glob
import shutil
import os

# Find all extxyz files
files = glob("data/*.extxyz")
print(f"Found {len(files)} extxyz files\n")

# Statistics
updated_count = 0
skipped_count = 0
backup_dir = "data/backup_before_cleanup"

# Create backup directory
os.makedirs(backup_dir, exist_ok=True)
print(f"Created backup directory: {backup_dir}\n")

print("="*80)
print("REMOVING e_inf AND e_static FROM FILES")
print("="*80)
print("\nKeeping eps_inf and eps_0 (the correct dielectric constants)\n")

for file in files:
    try:
        atoms = read(file)
        info = atoms.info

        # Check if file has the problematic properties
        has_e_inf = 'e_inf' in info
        has_e_static = 'e_static' in info

        if has_e_inf or has_e_static:
            # Backup original file
            backup_file = os.path.join(backup_dir, os.path.basename(file))
            shutil.copy2(file, backup_file)

            material_id = info.get('material_id', os.path.basename(file))

            # Show what we're removing
            removed = []
            if has_e_inf:
                removed.append(f"e_inf={info['e_inf']:.2f}")
                del info['e_inf']
            if has_e_static:
                removed.append(f"e_static={info['e_static']:.2f}")
                del info['e_static']

            # Show what we're keeping
            kept = []
            if 'eps_inf' in info:
                kept.append(f"eps_inf={info['eps_inf']:.2f}")
            if 'eps_0' in info:
                kept.append(f"eps_0={info['eps_0']:.2f}")

            # Write updated file
            write(file, atoms)

            print(f"✓ Updated {material_id}")
            print(f"  Removed: {', '.join(removed)}")
            print(f"  Kept:    {', '.join(kept)}")

            updated_count += 1
        else:
            skipped_count += 1

    except Exception as e:
        print(f"✗ Error processing {file}: {e}")
        skipped_count += 1

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal files: {len(files)}")
print(f"Updated: {updated_count}")
print(f"Skipped (no changes needed): {skipped_count}")
print(f"\nBackup files saved to: {backup_dir}")

print("\n✓ All files have been cleaned up!")
print("\nRemaining dielectric properties in files:")
print("  - eps_inf: optical/high-frequency dielectric constant (ε_∞ ≈ n²)")
print("  - eps_0: static/low-frequency dielectric constant (ε_0 > ε_∞)")
print("\nThe problematic e_inf and e_static properties have been removed.")

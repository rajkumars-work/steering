"""
Fix mislabeled e_inf and e_static values in extxyz files by swapping them
"""

from ase.io import read, write
from glob import glob
import shutil
import os

# Find all extxyz files
files = glob("data/*.extxyz")
print(f"Found {len(files)} extxyz files\n")

# Statistics
fixed_count = 0
skipped_count = 0
backup_dir = "data/backup_before_fix"

# Create backup directory
os.makedirs(backup_dir, exist_ok=True)
print(f"Created backup directory: {backup_dir}\n")

print("="*80)
print("FIXING FILES")
print("="*80)

for file in files:
    try:
        atoms = read(file)
        info = atoms.info

        # Check if file has both e_inf and e_static
        if 'e_inf' in info and 'e_static' in info:
            # Backup original file
            backup_file = os.path.join(backup_dir, os.path.basename(file))
            shutil.copy2(file, backup_file)

            # Get current (incorrect) values
            old_e_inf = info['e_inf']
            old_e_static = info['e_static']

            # Swap them (fix the labeling)
            info['e_inf'] = old_e_static  # What was called e_static is actually e_inf
            info['e_static'] = old_e_inf  # What was called e_inf is actually e_static

            # Write corrected file
            write(file, atoms)

            material_id = info.get('material_id', os.path.basename(file))
            print(f"✓ Fixed {material_id}")
            print(f"  OLD: e_inf={old_e_inf:.2f}, e_static={old_e_static:.2f}")
            print(f"  NEW: e_inf={info['e_inf']:.2f}, e_static={info['e_static']:.2f}")

            fixed_count += 1
        else:
            skipped_count += 1

    except Exception as e:
        print(f"✗ Error processing {file}: {e}")
        skipped_count += 1

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal files: {len(files)}")
print(f"Fixed: {fixed_count}")
print(f"Skipped: {skipped_count}")
print(f"\nBackup files saved to: {backup_dir}")
print("\n✓ All files have been corrected!")
print("\nThe labels are now:")
print("  - e_inf: high-frequency/optical dielectric constant")
print("  - e_static: static/low-frequency dielectric constant")

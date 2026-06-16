import argparse
import numpy as np
import sys
from ase.io import read

def is_realistic(atoms, min_dist_threshold=0.8, vol_per_atom_threshold=5.0, max_vol_per_atom=50.0, verbose=True):
    """
    Check if a structure is physically realistic using comprehensive checks.
    
    Checks:
    1. Basic Integrity: Non-empty, no NaNs/Infs in positions/cell.
    2. Cell Geometry: Positive volume, lattice vectors > 1.0 A, reasonable angles.
    3. Atomic Density: Volume per atom within [threshold, max_limit].
    4. Atomic Overlaps: No atoms closer than min_dist_threshold (mic=True).
    """
    # 0. Basic Integrity
    if len(atoms) == 0:
        if verbose: print("  [Realism] No atoms in structure.")
        return False
    
    if not np.all(np.isfinite(atoms.get_positions())):
        if verbose: print("  [Realism] Positions contain NaN or Inf.")
        return False
        
    if atoms.cell is None:
         if verbose: print("  [Realism] Structure has no cell.")
         return False
         
    if not np.all(np.isfinite(atoms.cell.array)):
        if verbose: print("  [Realism] Cell contains NaN or Inf.")
        return False

    # 1. Cell Geometry
    try:
        vol = atoms.get_volume()
    except Exception as e:
        if verbose: print(f"  [Realism] Failed to calculate volume: {e}")
        return False
        
    if vol < 0.1: # Practically zero
        if verbose: print(f"  [Realism] Cell volume near zero: {vol:.4f}")
        return False

    # Check lattice lengths
    lengths = atoms.cell.lengths()
    if np.any(lengths < 1.0): 
        if verbose: print(f"  [Realism] Lattice vector too short (< 1.0 A): {lengths}")
        return False
    
    # Check angles
    angles = atoms.cell.angles()
    if np.any(angles < 10.0) or np.any(angles > 170.0):
        if verbose: print(f"  [Realism] Extreme cell angles (<10 or >170): {angles}")
        return False

    # 2. Atomic Density
    vpa = vol / len(atoms)
    if vpa < vol_per_atom_threshold:
        if verbose: print(f"  [Realism] Density too high (VPA={vpa:.2f} < {vol_per_atom_threshold})")
        return False
    if vpa > max_vol_per_atom:
        if verbose: print(f"  [Realism] Density too low (VPA={vpa:.2f} > {max_vol_per_atom})")
        return False

    # 3. Overlaps / Short Bonds
    if len(atoms) > 1:
        try:
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d < 1e-4:
                if verbose: print(f"  [Realism] Overlapping atoms detected (dist < 1e-4 A).")
                return False
                
            if min_d < min_dist_threshold:
                if verbose: print(f"  [Realism] Atoms too close: {min_d:.2f} A < {min_dist_threshold} A")
                return False
        except Exception as e:
             if verbose: print(f"  [Realism] Distance check failed: {e}")
             return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Check if structures in a file are physically realistic.")
    parser.add_argument("file", help="Path to structure file (e.g., .xyz, .extxyz, .cif)")
    parser.add_argument("--index", default=":", help="Index of structures to read (default: all)")
    
    args = parser.parse_args()

    try:
        atoms_list = read(args.file, index=args.index)
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
    except Exception as e:
        print(f"Error reading file {args.file}: {e}")
        sys.exit(1)

    print(f"Checking {len(atoms_list)} structures from {args.file}...")
    
    valid_count = 0
    for i, atoms in enumerate(atoms_list):
        print(f"--- Structure {i} ---")
        if is_realistic(atoms):
            print("  Result: VALID")
            valid_count += 1
        else:
            print("  Result: INVALID")

    print(f"\nSummary: {valid_count}/{len(atoms_list)} structures are valid.")

if __name__ == "__main__":
    main()

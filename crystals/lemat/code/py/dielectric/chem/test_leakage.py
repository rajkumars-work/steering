"""
Quick test to verify that SOAP and composition features don't leak from atoms.info
"""

import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from mendeleev import element
from glob import glob

# Load a few test structures
EXTXYZ_PATH = "/data/assets/atlas/dielectrics/DATABASE/db-mp_updated/*.extxyz"
files = glob(EXTXYZ_PATH)[:10]  # Just test first 10 files

structures = []
for file in files[:10]:
    atoms_list = read(file, index=":")
    if isinstance(atoms_list, list):
        structures.extend(atoms_list[:1])  # One structure per file
    else:
        structures.append(atoms_list)

print(f"Testing with {len(structures)} structures")

# Define feature functions
def composition_features(atoms):
    Z = atoms.get_atomic_numbers()
    elems, counts = np.unique(Z, return_counts=True)
    frac = counts / counts.sum()

    electroneg = []
    radius = []

    for z in elems:
        e = element(int(z))
        electroneg.append(e.en_pauling or 0)
        radius.append(e.atomic_radius or 0)

    electroneg = np.array(electroneg)
    radius = np.array(radius)

    features = [
        len(elems),
        np.sum(frac * elems),
        np.var(elems),
        np.sum(frac * electroneg),
        np.var(electroneg),
        np.sum(frac * radius),
        np.var(radius),
    ]

    return np.array(features)

# Setup SOAP
species = sorted(list({Z for atoms in structures for Z in atoms.get_atomic_numbers()}))
soap = SOAP(species=species, r_cut=5.0, n_max=6, l_max=4, periodic=True, sparse=False)

def soap_features(atoms):
    s = soap.create(atoms)
    return s.mean(axis=0)

# Test each structure
print("\nTesting for data leakage...")
print("=" * 60)

all_passed = True
for i, atoms in enumerate(structures):
    print(f"\nStructure {i}: {atoms.get_chemical_formula()}")

    # Compute features WITH info
    comp1 = composition_features(atoms)
    soap1 = soap_features(atoms)

    # Save and clear info
    original_info = atoms.info.copy()
    original_arrays = {k: v.copy() for k, v in atoms.arrays.items() if k not in ['positions', 'numbers']}

    print(f"  Original info keys: {list(original_info.keys())[:5]}...")  # Show first 5

    # Clear everything except essential structure data
    atoms.info = {}
    for key in list(atoms.arrays.keys()):
        if key not in ['positions', 'numbers']:
            del atoms.arrays[key]

    # Compute features WITHOUT info
    comp2 = composition_features(atoms)
    soap2 = soap_features(atoms)

    # Restore
    atoms.info = original_info
    for key, val in original_arrays.items():
        atoms.arrays[key] = val

    # Compare
    comp_match = np.allclose(comp1, comp2, rtol=1e-10)
    soap_match = np.allclose(soap1, soap2, rtol=1e-10)

    if comp_match and soap_match:
        print(f"  ✓ PASS: Features identical with/without info")
    else:
        print(f"  ✗ FAIL: Features differ!")
        if not comp_match:
            print(f"    Composition diff: max={np.abs(comp1-comp2).max()}, mean={np.abs(comp1-comp2).mean()}")
        if not soap_match:
            print(f"    SOAP diff: max={np.abs(soap1-soap2).max()}, mean={np.abs(soap1-soap2).mean()}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✓✓✓ ALL TESTS PASSED - No data leakage detected!")
    print("Features do NOT depend on atoms.info or atoms.arrays")
else:
    print("✗✗✗ TESTS FAILED - DATA LEAKAGE DETECTED!")
    print("Features are accessing atoms.info or atoms.arrays!")

print("=" * 60)

"""
Analyze the train/test split to ensure proper separation
"""

import numpy as np
from ase.io import read
from glob import glob
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

# Load structures
EXTXYZ_PATH = "/data/assets/atlas/dielectrics/DATABASE/db-mp_updated/*.extxyz"
PROPERTY_KEY = "bandgap"

print("Loading structures...")
files = glob(EXTXYZ_PATH)
structures = []
for file in files:
    atoms_list = read(file, index=":")
    if isinstance(atoms_list, list):
        structures.extend(atoms_list)
    else:
        structures.append(atoms_list)

# Filter structures with target property
valid_structures = []
y_values = []
for atoms in structures:
    if PROPERTY_KEY in atoms.info:
        valid_structures.append(atoms)
        y_values.append(atoms.info[PROPERTY_KEY])

structures = valid_structures
y = np.array(y_values)

print(f"Loaded {len(structures)} structures with '{PROPERTY_KEY}' property")

# Create composition groups
def composition_string(atoms):
    Z = atoms.get_atomic_numbers()
    elems, counts = np.unique(Z, return_counts=True)
    return "_".join(f"{e}{c}" for e, c in sorted(zip(elems, counts)))

groups = [composition_string(a) for a in structures]

# Perform split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(structures, y, groups))

train_groups = [groups[i] for i in train_idx]
test_groups = [groups[i] for i in test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT ANALYSIS")
print("=" * 60)

print(f"\nDataset size:")
print(f"  Total structures: {len(structures)}")
print(f"  Training: {len(train_idx)} ({100*len(train_idx)/len(structures):.1f}%)")
print(f"  Test: {len(test_idx)} ({100*len(test_idx)/len(structures):.1f}%)")

print(f"\nComposition groups:")
print(f"  Total unique compositions: {len(set(groups))}")
print(f"  Train compositions: {len(set(train_groups))}")
print(f"  Test compositions: {len(set(test_groups))}")

# Check for overlap
overlap = set(train_groups) & set(test_groups)
print(f"  Overlapping compositions: {len(overlap)}")

if len(overlap) > 0:
    print(f"\n⚠ WARNING: {len(overlap)} compositions appear in both train and test!")
    print(f"  This means GroupShuffleSplit is NOT working as expected!")
    print(f"  Examples: {list(overlap)[:5]}")
else:
    print(f"\n✓ No composition overlap between train and test sets")

# Target distribution
print(f"\nTarget ({PROPERTY_KEY}) distribution:")
print(f"  Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}, min={y_train.min():.3f}, max={y_train.max():.3f}")
print(f"  Test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}, min={y_test.min():.3f}, max={y_test.max():.3f}")

# Check composition diversity
print(f"\nComposition frequency:")
comp_counts = Counter(groups)
print(f"  Most common compositions:")
for comp, count in comp_counts.most_common(10):
    in_train = sum(1 for g in train_groups if g == comp)
    in_test = sum(1 for g in test_groups if g == comp)
    print(f"    {comp}: {count} total ({in_train} train, {in_test} test)")

# Check for element overlap
train_elements = set()
test_elements = set()

for idx in train_idx:
    train_elements.update(structures[idx].get_atomic_numbers())

for idx in test_idx:
    test_elements.update(structures[idx].get_atomic_numbers())

print(f"\nElement coverage:")
print(f"  Elements in training: {len(train_elements)}")
print(f"  Elements in test: {len(test_elements)}")
print(f"  Elements only in train: {train_elements - test_elements}")
print(f"  Elements only in test: {test_elements - train_elements}")

print("\n" + "=" * 60)

# Recommendations
print("\nRECOMMENDATIONS:")
print("=" * 60)

if len(overlap) > 0:
    print("❌ ISSUE: Composition groups appear in both train and test")
    print("   This could explain high performance (model seeing similar compositions)")
    print("   GroupShuffleSplit should prevent this - check implementation")
else:
    print("✓ Train/test split is correct (no composition overlap)")

if test_elements - train_elements:
    print(f"\n⚠ WARNING: {len(test_elements - train_elements)} elements only in test set")
    print("   Model may struggle with extrapolation to new elements")

if y_train.max() < y_test.max() or y_train.min() > y_test.min():
    print(f"\n⚠ WARNING: Test set has targets outside training range")
    print("   Model is being asked to extrapolate")

# Analyze performance by whether composition is in training
print("\n" + "=" * 60)
print("Potential explanations for high Spearman (0.880):")
print("=" * 60)
print("\n1. Bandgap is strongly determined by composition")
print("   - Electronegativity and atomic number are known to correlate with bandgap")
print("   - This is actually a GOOD sign - your dataset has learnable signal")
print("\n2. SOAP adds structural information")
print("   - Different structures of same composition can have different bandgaps")
print("   - SOAP helps distinguish these cases")
print("\n3. XGBoost is a very powerful model")
print("   - Tree ensembles are excellent at capturing complex patterns")
print("\n4. Check if similar compositions have similar bandgaps")
print("   - If compositions cluster by bandgap, interpolation is easier")

print("=" * 60)

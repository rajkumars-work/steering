"""
Get statistics for bandgap vs eps_0 from the dielectric dataset
"""
import numpy as np
from ase.io import read
from glob import glob
from scipy.stats import pearsonr, spearmanr

# Load structures
EXTXYZ_PATH = "/data/assets/atlas/dielectrics/DATABASE/db-mp_updated/*.extxyz"

print("Loading structures...")
files = glob(EXTXYZ_PATH)
print(f"Found {len(files)} extxyz files")

structures = []
for file in files:
    atoms_list = read(file, index=":")
    if isinstance(atoms_list, list):
        structures.extend(atoms_list)
    else:
        structures.append(atoms_list)

print(f"Loaded {len(structures)} structures")

# Extract bandgap and eps_0 values
bandgaps = []
eps_0s = []

for atoms in structures:
    if "bandgap" in atoms.info and "eps_0" in atoms.info:
        bandgaps.append(atoms.info["bandgap"])
        eps_0s.append(atoms.info["eps_0"])

bandgaps = np.array(bandgaps)
eps_0s = np.array(eps_0s)

print(f"\nDataset statistics:")
print(f"- {len(bandgaps):,} materials with both properties")
print(f"- Bandgap range: {bandgaps.min():.3f} to {bandgaps.max():.3f} eV")
print(f"- eps_0 range: {eps_0s.min():.3f} to {eps_0s.max():.3f}")

# Compute correlation
pearson_r, pearson_p = pearsonr(eps_0s, bandgaps)
spearman_r, spearman_p = spearmanr(eps_0s, bandgaps)

print(f"\nCorrelation analysis:")
print(f"- Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
print(f"- Spearman ρ: {spearman_r:.3f} (p={spearman_p:.3e})")

# Additional statistics
print(f"\nAdditional statistics:")
print(f"- Mean bandgap: {bandgaps.mean():.3f} eV (std: {bandgaps.std():.3f})")
print(f"- Mean eps_0: {eps_0s.mean():.3f} (std: {eps_0s.std():.3f})")
print(f"- Median bandgap: {np.median(bandgaps):.3f} eV")
print(f"- Median eps_0: {np.median(eps_0s):.3f}")

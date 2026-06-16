"""
Plot bandgap vs eps_inf from the dielectric dataset
"""
import numpy as np
from ase.io import read
from glob import glob
import matplotlib.pyplot as plt

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

# Extract bandgap and eps_inf values
bandgaps = []
eps_infs = []

for atoms in structures:
    if "bandgap" in atoms.info and "eps_inf" in atoms.info:
        bandgaps.append(atoms.info["bandgap"])
        eps_infs.append(atoms.info["eps_inf"])

bandgaps = np.array(bandgaps)
eps_infs = np.array(eps_infs)

print(f"\nFound {len(bandgaps)} structures with both bandgap and eps_inf")
print(f"Bandgap range: [{bandgaps.min():.3f}, {bandgaps.max():.3f}]")
print(f"eps_inf range: [{eps_infs.min():.3f}, {eps_infs.max():.3f}]")

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(eps_infs, bandgaps, alpha=0.5, s=20, edgecolors='none')
plt.xlabel('eps_inf (high-frequency dielectric constant)', fontsize=12)
plt.ylabel('Bandgap (eV)', fontsize=12)
plt.title(f'Bandgap vs eps_inf (n={len(bandgaps)} materials)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_file = 'bandgap_vs_eps_inf.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to {output_file}")

# Also compute correlation
from scipy.stats import pearsonr, spearmanr
pearson_r, pearson_p = pearsonr(eps_infs, bandgaps)
spearman_r, spearman_p = spearmanr(eps_infs, bandgaps)

print(f"\nCorrelation statistics:")
print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
print(f"  Spearman ρ: {spearman_r:.3f} (p={spearman_p:.3e})")

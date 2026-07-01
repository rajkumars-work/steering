"""
Comprehensive analysis of eps_inf (optical dielectric constant) from the dataset
"""
import numpy as np
from ase.io import read
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")

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

# Extract eps_inf and related properties
eps_infs = []
refractive_indices = []
bandgaps = []
materials = []

for atoms in structures:
    if "eps_inf" in atoms.info:
        eps_infs.append(atoms.info["eps_inf"])
        refractive_indices.append(atoms.info.get("refractive_index", np.nan))
        bandgaps.append(atoms.info.get("bandgap", np.nan))
        materials.append(atoms.info.get("material_id", "unknown"))

eps_infs = np.array(eps_infs)
refractive_indices = np.array(refractive_indices)
bandgaps = np.array(bandgaps)

print(f"\n{'='*80}")
print(f"ANALYSIS OF eps_inf (Optical Dielectric Constant)")
print(f"{'='*80}")

print(f"\nDataset: {len(eps_infs):,} materials with eps_inf")

# Basic statistics
print(f"\n{'='*80}")
print("BASIC STATISTICS")
print(f"{'='*80}")
print(f"Mean:     {eps_infs.mean():.3f}")
print(f"Median:   {np.median(eps_infs):.3f}")
print(f"Std dev:  {eps_infs.std():.3f}")
print(f"Min:      {eps_infs.min():.3f}")
print(f"Max:      {eps_infs.max():.3f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(eps_infs, p):.3f}")

# Check relationship with refractive index (eps_inf ≈ n²)
valid_n = ~np.isnan(refractive_indices)
if valid_n.sum() > 0:
    n_squared = refractive_indices[valid_n] ** 2
    eps_inf_valid = eps_infs[valid_n]

    print(f"\n{'='*80}")
    print("RELATIONSHIP WITH REFRACTIVE INDEX (eps_inf ≈ n²)")
    print(f"{'='*80}")
    print(f"Materials with both eps_inf and n: {valid_n.sum():,}")

    # Compute correlation
    pearson_r, pearson_p = pearsonr(eps_inf_valid, n_squared)
    spearman_r, spearman_p = spearmanr(eps_inf_valid, n_squared)

    print(f"\nCorrelation between eps_inf and n²:")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

    # Compute difference
    diff = eps_inf_valid - n_squared
    print(f"\nDifference (eps_inf - n²):")
    print(f"  Mean:   {diff.mean():.3f}")
    print(f"  Median: {np.median(diff):.3f}")
    print(f"  Std:    {diff.std():.3f}")
    print(f"  |Mean|: {np.abs(diff).mean():.3f}")

    # Count how many are close
    close_1pct = np.abs(diff / eps_inf_valid) < 0.01
    close_5pct = np.abs(diff / eps_inf_valid) < 0.05
    close_10pct = np.abs(diff / eps_inf_valid) < 0.10

    print(f"\nAgreement:")
    print(f"  Within 1% : {close_1pct.sum():,} ({close_1pct.sum()/len(diff)*100:.1f}%)")
    print(f"  Within 5% : {close_5pct.sum():,} ({close_5pct.sum()/len(diff)*100:.1f}%)")
    print(f"  Within 10%: {close_10pct.sum():,} ({close_10pct.sum()/len(diff)*100:.1f}%)")

# Correlation with bandgap
valid_bg = ~np.isnan(bandgaps)
if valid_bg.sum() > 0:
    print(f"\n{'='*80}")
    print("CORRELATION WITH BANDGAP")
    print(f"{'='*80}")
    print(f"Materials with both properties: {valid_bg.sum():,}")

    pearson_r, pearson_p = pearsonr(eps_infs[valid_bg], bandgaps[valid_bg])
    spearman_r, spearman_p = spearmanr(eps_infs[valid_bg], bandgaps[valid_bg])

    print(f"\nCorrelation between eps_inf and bandgap:")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

# Physical validity check
print(f"\n{'='*80}")
print("PHYSICAL VALIDITY CHECKS")
print(f"{'='*80}")

# Typical range: 2-10, up to 20 for high-index materials
typical_min = 1.0  # Allow slightly below 2 for some materials
typical_max = 20.0
very_high = 50.0

in_typical = (eps_infs >= typical_min) & (eps_infs <= typical_max)
high = (eps_infs > typical_max) & (eps_infs <= very_high)
very_high_count = (eps_infs > very_high).sum()
negative = (eps_infs < 0).sum()

print(f"\neps_inf ranges:")
print(f"  Typical (1-20):     {in_typical.sum():,} ({in_typical.sum()/len(eps_infs)*100:.1f}%)")
print(f"  High (20-50):       {high.sum():,} ({high.sum()/len(eps_infs)*100:.1f}%)")
print(f"  Very high (>50):    {very_high_count:,} ({very_high_count/len(eps_infs)*100:.1f}%)")
print(f"  Negative (<0):      {negative:,} ({negative/len(eps_infs)*100:.1f}%)")

if very_high_count > 0:
    print(f"\nMaterials with eps_inf > 50:")
    very_high_idx = np.where(eps_infs > very_high)[0]
    for idx in very_high_idx[:10]:  # Show first 10
        print(f"  {materials[idx]}: eps_inf = {eps_infs[idx]:.2f}")
    if very_high_count > 10:
        print(f"  ... and {very_high_count - 10} more")

# Create plots
print(f"\n{'='*80}")
print("CREATING PLOTS")
print(f"{'='*80}")

# 1. Distribution histogram
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram (full range)
ax = axes[0, 0]
ax.hist(eps_infs, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('eps_inf', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Distribution of eps_inf (n={len(eps_infs):,})', fontsize=12)
ax.axvline(eps_infs.mean(), color='r', linestyle='--', label=f'Mean: {eps_infs.mean():.2f}')
ax.axvline(np.median(eps_infs), color='g', linestyle='--', label=f'Median: {np.median(eps_infs):.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram (zoomed to typical range)
ax = axes[0, 1]
eps_infs_typical = eps_infs[(eps_infs >= 0) & (eps_infs <= 25)]
ax.hist(eps_infs_typical, bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('eps_inf', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Distribution (0-25 range, n={len(eps_infs_typical):,})', fontsize=12)
ax.grid(True, alpha=0.3)

# eps_inf vs n² scatter plot
if valid_n.sum() > 0:
    ax = axes[1, 0]
    ax.scatter(n_squared, eps_inf_valid, alpha=0.5, s=10, edgecolors='none')

    # Add y=x line
    max_val = max(n_squared.max(), eps_inf_valid.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (perfect agreement)', linewidth=2)

    ax.set_xlabel('n² (refractive index squared)', fontsize=11)
    ax.set_ylabel('eps_inf', fontsize=11)
    ax.set_title(f'eps_inf vs n² (r={pearson_r:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

# eps_inf vs bandgap scatter plot
if valid_bg.sum() > 0:
    ax = axes[1, 1]
    ax.scatter(eps_infs[valid_bg], bandgaps[valid_bg], alpha=0.5, s=10, edgecolors='none')
    ax.set_xlabel('eps_inf', fontsize=11)
    ax.set_ylabel('Bandgap (eV)', fontsize=11)
    ax.set_title(f'eps_inf vs Bandgap (ρ={spearman_r:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'eps_inf_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plots saved to {output_file}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

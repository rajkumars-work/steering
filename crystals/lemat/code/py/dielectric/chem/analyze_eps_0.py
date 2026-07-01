"""
Comprehensive analysis of eps_0 (static dielectric constant) from the dataset
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

# Extract eps_0 and related properties
eps_0s = []
eps_infs = []
bandgaps = []
materials = []

for atoms in structures:
    if "eps_0" in atoms.info:
        eps_0s.append(atoms.info["eps_0"])
        eps_infs.append(atoms.info.get("eps_inf", np.nan))
        bandgaps.append(atoms.info.get("bandgap", np.nan))
        materials.append(atoms.info.get("material_id", "unknown"))

eps_0s = np.array(eps_0s)
eps_infs = np.array(eps_infs)
bandgaps = np.array(bandgaps)

print(f"\n{'='*80}")
print(f"ANALYSIS OF eps_0 (Static Dielectric Constant)")
print(f"{'='*80}")

print(f"\nDataset: {len(eps_0s):,} materials with eps_0")

# Basic statistics
print(f"\n{'='*80}")
print("BASIC STATISTICS")
print(f"{'='*80}")
print(f"Mean:     {eps_0s.mean():.3f}")
print(f"Median:   {np.median(eps_0s):.3f}")
print(f"Std dev:  {eps_0s.std():.3f}")
print(f"Min:      {eps_0s.min():.3f}")
print(f"Max:      {eps_0s.max():.3f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(eps_0s, p):.3f}")

# Check relationship with eps_inf (eps_0 > eps_inf)
valid_both = ~np.isnan(eps_infs)
if valid_both.sum() > 0:
    eps_0_valid = eps_0s[valid_both]
    eps_inf_valid = eps_infs[valid_both]

    print(f"\n{'='*80}")
    print("RELATIONSHIP WITH eps_inf (eps_0 should be > eps_inf)")
    print(f"{'='*80}")
    print(f"Materials with both eps_0 and eps_inf: {valid_both.sum():,}")

    # Check eps_0 > eps_inf
    correct = eps_0_valid > eps_inf_valid
    equal = np.isclose(eps_0_valid, eps_inf_valid, rtol=1e-3)
    incorrect = eps_0_valid < eps_inf_valid

    print(f"\nPhysical relationship check:")
    print(f"  eps_0 > eps_inf:  {correct.sum():,} ({correct.sum()/len(eps_0_valid)*100:.1f}%)")
    print(f"  eps_0 ≈ eps_inf:  {equal.sum():,} ({equal.sum()/len(eps_0_valid)*100:.1f}%)")
    print(f"  eps_0 < eps_inf:  {incorrect.sum():,} ({incorrect.sum()/len(eps_0_valid)*100:.1f}%) ⚠️")

    # Compute ratio eps_0 / eps_inf
    ratio = eps_0_valid / eps_inf_valid
    ratio_finite = ratio[np.isfinite(ratio)]

    print(f"\nRatio eps_0 / eps_inf:")
    print(f"  Mean:   {ratio_finite.mean():.3f}")
    print(f"  Median: {np.median(ratio_finite):.3f}")
    print(f"  Std:    {ratio_finite.std():.3f}")
    print(f"  Min:    {ratio_finite.min():.3f}")
    print(f"  Max:    {ratio_finite.max():.3f}")

    # Typical ratio should be 1.5-5
    typical_ratio = (ratio_finite >= 1.0) & (ratio_finite <= 5.0)
    high_ratio = (ratio_finite > 5.0) & (ratio_finite <= 10.0)
    very_high_ratio = (ratio_finite > 10.0)

    print(f"\nRatio ranges:")
    print(f"  Typical (1-5):      {typical_ratio.sum():,} ({typical_ratio.sum()/len(ratio_finite)*100:.1f}%)")
    print(f"  High (5-10):        {high_ratio.sum():,} ({high_ratio.sum()/len(ratio_finite)*100:.1f}%)")
    print(f"  Very high (>10):    {very_high_ratio.sum():,} ({very_high_ratio.sum()/len(ratio_finite)*100:.1f}%)")

    # Compute correlation
    pearson_r, pearson_p = pearsonr(eps_0_valid, eps_inf_valid)
    spearman_r, spearman_p = spearmanr(eps_0_valid, eps_inf_valid)

    print(f"\nCorrelation between eps_0 and eps_inf:")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

    # Compute ionic contribution (eps_0 - eps_inf)
    ionic = eps_0_valid - eps_inf_valid

    print(f"\nIonic contribution (eps_0 - eps_inf):")
    print(f"  Mean:   {ionic.mean():.3f}")
    print(f"  Median: {np.median(ionic):.3f}")
    print(f"  Std:    {ionic.std():.3f}")
    print(f"  Min:    {ionic.min():.3f}")
    print(f"  Max:    {ionic.max():.3f}")

# Correlation with bandgap
valid_bg = ~np.isnan(bandgaps)
if valid_bg.sum() > 0:
    print(f"\n{'='*80}")
    print("CORRELATION WITH BANDGAP")
    print(f"{'='*80}")
    print(f"Materials with both properties: {valid_bg.sum():,}")

    pearson_r_bg, pearson_p_bg = pearsonr(eps_0s[valid_bg], bandgaps[valid_bg])
    spearman_r_bg, spearman_p_bg = spearmanr(eps_0s[valid_bg], bandgaps[valid_bg])

    print(f"\nCorrelation between eps_0 and bandgap:")
    print(f"  Pearson r:  {pearson_r_bg:.4f} (p={pearson_p_bg:.2e})")
    print(f"  Spearman ρ: {spearman_r_bg:.4f} (p={spearman_p_bg:.2e})")

# Physical validity check
print(f"\n{'='*80}")
print("PHYSICAL VALIDITY CHECKS")
print(f"{'='*80}")

# Typical range: 5-100, can be higher for ferroelectrics
typical_min = 2.0
typical_max = 100.0
high = 200.0
very_high = 500.0

in_typical = (eps_0s >= typical_min) & (eps_0s <= typical_max)
in_high = (eps_0s > typical_max) & (eps_0s <= high)
in_very_high = (eps_0s > high) & (eps_0s <= very_high)
extreme = (eps_0s > very_high)
negative = (eps_0s < 0).sum()

print(f"\neps_0 ranges:")
print(f"  Typical (2-100):    {in_typical.sum():,} ({in_typical.sum()/len(eps_0s)*100:.1f}%)")
print(f"  High (100-200):     {in_high.sum():,} ({in_high.sum()/len(eps_0s)*100:.1f}%)")
print(f"  Very high (200-500): {in_very_high.sum():,} ({in_very_high.sum()/len(eps_0s)*100:.1f}%)")
print(f"  Extreme (>500):     {extreme.sum():,} ({extreme.sum()/len(eps_0s)*100:.1f}%)")
print(f"  Negative (<0):      {negative:,} ({negative/len(eps_0s)*100:.1f}%)")

if extreme.sum() > 0:
    print(f"\nMaterials with eps_0 > 500:")
    extreme_idx = np.where(eps_0s > very_high)[0]
    for idx in extreme_idx[:10]:  # Show first 10
        print(f"  {materials[idx]}: eps_0 = {eps_0s[idx]:.2f}")
    if extreme.sum() > 10:
        print(f"  ... and {extreme.sum() - 10} more")

# Create plots
print(f"\n{'='*80}")
print("CREATING PLOTS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution histogram (full range)
ax = axes[0, 0]
ax.hist(eps_0s, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('eps_0', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Distribution of eps_0 (n={len(eps_0s):,})', fontsize=12)
ax.axvline(eps_0s.mean(), color='r', linestyle='--', label=f'Mean: {eps_0s.mean():.2f}')
ax.axvline(np.median(eps_0s), color='g', linestyle='--', label=f'Median: {np.median(eps_0s):.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Distribution histogram (zoomed to typical range)
ax = axes[0, 1]
eps_0s_typical = eps_0s[(eps_0s >= 0) & (eps_0s <= 150)]
ax.hist(eps_0s_typical, bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('eps_0', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Distribution (0-150 range, n={len(eps_0s_typical):,})', fontsize=12)
ax.grid(True, alpha=0.3)

# 3. eps_0 vs eps_inf scatter plot
if valid_both.sum() > 0:
    ax = axes[1, 0]
    ax.scatter(eps_inf_valid, eps_0_valid, alpha=0.5, s=10, edgecolors='none')

    # Add y=x line (where eps_0 = eps_inf, which should not happen)
    max_val = max(eps_inf_valid.max(), eps_0_valid.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (eps_0 = eps_inf)', linewidth=2)

    ax.set_xlabel('eps_inf', fontsize=11)
    ax.set_ylabel('eps_0', fontsize=11)
    ax.set_title(f'eps_0 vs eps_inf (r={pearson_r:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(100, eps_inf_valid.max()))
    ax.set_ylim(0, min(150, eps_0_valid.max()))

# 4. eps_0 vs bandgap scatter plot
if valid_bg.sum() > 0:
    ax = axes[1, 1]
    ax.scatter(eps_0s[valid_bg], bandgaps[valid_bg], alpha=0.5, s=10, edgecolors='none')
    ax.set_xlabel('eps_0', fontsize=11)
    ax.set_ylabel('Bandgap (eV)', fontsize=11)
    ax.set_title(f'eps_0 vs Bandgap (ρ={spearman_r_bg:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(150, eps_0s[valid_bg].max()))

plt.tight_layout()
output_file = 'eps_0_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plots saved to {output_file}")

# Create additional plot: Ratio distribution
fig, ax = plt.subplots(figsize=(10, 6))
if valid_both.sum() > 0:
    ratio_plot = ratio_finite[ratio_finite < 20]  # Limit to reasonable range for visualization
    ax.hist(ratio_plot, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('eps_0 / eps_inf', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of eps_0/eps_inf Ratio (n={len(ratio_plot):,})', fontsize=12)
    ax.axvline(ratio_plot.mean(), color='r', linestyle='--', label=f'Mean: {ratio_plot.mean():.2f}')
    ax.axvline(np.median(ratio_plot), color='g', linestyle='--', label=f'Median: {np.median(ratio_plot):.2f}')

    # Mark typical range
    ax.axvspan(1.5, 5.0, alpha=0.2, color='green', label='Typical range (1.5-5)')

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file2 = 'eps_ratio_distribution.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Ratio plot saved to {output_file2}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

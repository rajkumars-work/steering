"""
Test if di_props.py computed values correlate with stored values in data files
"""
import numpy as np
from ase.io import read
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")

# Import the Dielectrics class
from di_props import Dielectrics

print("="*80)
print("TESTING CORRELATION: Computed vs Stored Properties")
print("="*80)

# Load a sample of structures from data directory
data_files = glob("data/*.extxyz")[:20]  # Test on first 20 files
print(f"\nLoading {len(data_files)} structures from data directory...")

structures = []
stored_bandgaps = []
stored_eps_infs = []
stored_eps_0s = []
material_ids = []

for file in data_files:
    atoms = read(file)

    # Store original properties
    if "bandgap" in atoms.info and "eps_inf" in atoms.info and "eps_0" in atoms.info:
        structures.append(atoms)
        stored_bandgaps.append(atoms.info["bandgap"])
        stored_eps_infs.append(atoms.info["eps_inf"])
        stored_eps_0s.append(atoms.info["eps_0"])
        material_ids.append(atoms.info.get("material_id", "unknown"))

print(f"Found {len(structures)} structures with bandgap, eps_inf, and eps_0")

if len(structures) == 0:
    print("ERROR: No structures found with required properties!")
    exit(1)

# Compute properties using di_props.py
print(f"\n{'='*80}")
print("COMPUTING PROPERTIES USING di_props.py")
print(f"{'='*80}")
print("\nThis may take a few minutes (requires Ray cluster and model inference)...")

try:
    with Dielectrics() as di:
        computed_results = di.compute(structures)

    # Extract computed values
    computed_bandgaps = [r["bandgap"] for r in computed_results]
    computed_epsilons = [r["epsilon"] for r in computed_results]
    computed_eps_infs = [r["eps_inf"] for r in computed_results]
    computed_eps_0s = [r["eps_0"] for r in computed_results]

    print(f"\n✓ Successfully computed properties for {len(computed_results)} structures")

except Exception as e:
    print(f"\n✗ ERROR during computation: {e}")
    print("\nThis might require Ray cluster setup and model access.")
    print("Skipping computation test.")
    exit(1)

# Compare stored vs computed
print(f"\n{'='*80}")
print("COMPARISON: Stored vs Computed Values")
print(f"{'='*80}")

# Convert to numpy arrays
stored_bandgaps = np.array(stored_bandgaps)
stored_eps_infs = np.array(stored_eps_infs)
stored_eps_0s = np.array(stored_eps_0s)
computed_bandgaps = np.array(computed_bandgaps)
computed_epsilons = np.array(computed_epsilons)
computed_eps_infs = np.array(computed_eps_infs)
computed_eps_0s = np.array(computed_eps_0s)

print(f"\nSample size: {len(structures)} materials")

# Bandgap comparison
print(f"\n{'='*80}")
print("BANDGAP COMPARISON")
print(f"{'='*80}")

print(f"\nStored bandgap:   min={stored_bandgaps.min():.3f}, max={stored_bandgaps.max():.3f}, mean={stored_bandgaps.mean():.3f}")
print(f"Computed bandgap: min={computed_bandgaps.min():.3f}, max={computed_bandgaps.max():.3f}, mean={computed_bandgaps.mean():.3f}")

# Correlation
pearson_bg, p_bg = pearsonr(stored_bandgaps, computed_bandgaps)
spearman_bg, sp_bg = spearmanr(stored_bandgaps, computed_bandgaps)

print(f"\nCorrelation (Stored vs Computed):")
print(f"  Pearson r:  {pearson_bg:.4f} (p={p_bg:.2e})")
print(f"  Spearman ρ: {spearman_bg:.4f} (p={sp_bg:.2e})")

# Compute differences
diff_bg = computed_bandgaps - stored_bandgaps
mae_bg = np.abs(diff_bg).mean()
rmse_bg = np.sqrt((diff_bg**2).mean())

print(f"\nDifference (Computed - Stored):")
print(f"  Mean:  {diff_bg.mean():.3f}")
print(f"  MAE:   {mae_bg:.3f}")
print(f"  RMSE:  {rmse_bg:.3f}")

# eps_inf / epsilon comparison
print(f"\n{'='*80}")
print("EPSILON (eps_inf) COMPARISON")
print(f"{'='*80}")

print(f"\nStored eps_inf:     min={stored_eps_infs.min():.3f}, max={stored_eps_infs.max():.3f}, mean={stored_eps_infs.mean():.3f}")
print(f"Computed epsilon:   min={computed_epsilons.min():.3f}, max={computed_epsilons.max():.3f}, mean={computed_epsilons.mean():.3f}")
print(f"Computed eps_inf:   min={computed_eps_infs.min():.3f}, max={computed_eps_infs.max():.3f}, mean={computed_eps_infs.mean():.3f}")

# Note: computed_epsilons and computed_eps_infs should be the same
assert np.allclose(computed_epsilons, computed_eps_infs), "ERROR: epsilon and eps_inf should be equal!"

# Correlation
pearson_eps, p_eps = pearsonr(stored_eps_infs, computed_epsilons)
spearman_eps, sp_eps = spearmanr(stored_eps_infs, computed_epsilons)

print(f"\nCorrelation (Stored eps_inf vs Computed epsilon):")
print(f"  Pearson r:  {pearson_eps:.4f} (p={p_eps:.2e})")
print(f"  Spearman ρ: {spearman_eps:.4f} (p={sp_eps:.2e})")

# Compute differences
diff_eps = computed_epsilons - stored_eps_infs
mae_eps = np.abs(diff_eps).mean()
rmse_eps = np.sqrt((diff_eps**2).mean())

print(f"\nDifference (Computed - Stored):")
print(f"  Mean:  {diff_eps.mean():.3f}")
print(f"  MAE:   {mae_eps:.3f}")
print(f"  RMSE:  {rmse_eps:.3f}")

# eps_0 comparison
print(f"\n{'='*80}")
print("eps_0 (STATIC) COMPARISON")
print(f"{'='*80}")

print(f"\nStored eps_0:     min={stored_eps_0s.min():.3f}, max={stored_eps_0s.max():.3f}, mean={stored_eps_0s.mean():.3f}")
print(f"Computed eps_0:   min={computed_eps_0s.min():.3f}, max={computed_eps_0s.max():.3f}, mean={computed_eps_0s.mean():.3f}")

# Correlation
pearson_eps0, p_eps0 = pearsonr(stored_eps_0s, computed_eps_0s)
spearman_eps0, sp_eps0 = spearmanr(stored_eps_0s, computed_eps_0s)

print(f"\nCorrelation (Stored vs Computed):")
print(f"  Pearson r:  {pearson_eps0:.4f} (p={p_eps0:.2e})")
print(f"  Spearman ρ: {spearman_eps0:.4f} (p={sp_eps0:.2e})")

# Compute differences
diff_eps0 = computed_eps_0s - stored_eps_0s
mae_eps0 = np.abs(diff_eps0).mean()
rmse_eps0 = np.sqrt((diff_eps0**2).mean())

print(f"\nDifference (Computed - Stored):")
print(f"  Mean:  {diff_eps0.mean():.3f}")
print(f"  MAE:   {mae_eps0:.3f}")
print(f"  RMSE:  {rmse_eps0:.3f}")

# Detailed comparison table
print(f"\n{'='*80}")
print("DETAILED COMPARISON TABLE")
print(f"{'='*80}")

print(f"\n{'Material ID':<25} {'Stored BG':>10} {'Comp BG':>10} {'Stored ε∞':>10} {'Comp ε∞':>10} {'Stored ε₀':>10} {'Comp ε₀':>10}")
print("-"*100)

for i in range(min(10, len(structures))):
    mat_id = material_ids[i][:24]  # Truncate long IDs
    print(f"{mat_id:<25} {stored_bandgaps[i]:>10.3f} {computed_bandgaps[i]:>10.3f} "
          f"{stored_eps_infs[i]:>10.3f} {computed_epsilons[i]:>10.3f} "
          f"{stored_eps_0s[i]:>10.3f} {computed_eps_0s[i]:>10.3f}")

if len(structures) > 10:
    print(f"... and {len(structures) - 10} more materials")

# Create comparison plots
print(f"\n{'='*80}")
print("CREATING COMPARISON PLOTS")
print(f"{'='*80}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bandgap comparison
ax = axes[0]
ax.scatter(stored_bandgaps, computed_bandgaps, alpha=0.6, s=50)
ax.plot([stored_bandgaps.min(), stored_bandgaps.max()],
        [stored_bandgaps.min(), stored_bandgaps.max()],
        'r--', label='y=x (perfect match)', linewidth=2)
ax.set_xlabel('Stored Bandgap (eV)', fontsize=11)
ax.set_ylabel('Computed Bandgap (eV)', fontsize=11)
ax.set_title(f'Bandgap\nr={pearson_bg:.3f}, MAE={mae_bg:.3f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# eps_inf comparison
ax = axes[1]
ax.scatter(stored_eps_infs, computed_epsilons, alpha=0.6, s=50)
ax.plot([stored_eps_infs.min(), stored_eps_infs.max()],
        [stored_eps_infs.min(), stored_eps_infs.max()],
        'r--', label='y=x (perfect match)', linewidth=2)
ax.set_xlabel('Stored eps_inf', fontsize=11)
ax.set_ylabel('Computed epsilon', fontsize=11)
ax.set_title(f'Epsilon (eps_inf)\nr={pearson_eps:.3f}, MAE={mae_eps:.3f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# eps_0 comparison
ax = axes[2]
ax.scatter(stored_eps_0s, computed_eps_0s, alpha=0.6, s=50)
ax.plot([stored_eps_0s.min(), stored_eps_0s.max()],
        [stored_eps_0s.min(), stored_eps_0s.max()],
        'r--', label='y=x (perfect match)', linewidth=2)
ax.set_xlabel('Stored eps_0', fontsize=11)
ax.set_ylabel('Computed eps_0', fontsize=11)
ax.set_title(f'eps_0 (Static)\nr={pearson_eps0:.3f}, MAE={mae_eps0:.3f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'stored_vs_computed_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plots saved to {output_file}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nCorrelation Summary:")
print(f"  Bandgap:  r={pearson_bg:.3f}, ρ={spearman_bg:.3f}, MAE={mae_bg:.3f}")
print(f"  eps_inf:  r={pearson_eps:.3f}, ρ={spearman_eps:.3f}, MAE={mae_eps:.3f}")
print(f"  eps_0:    r={pearson_eps0:.3f}, ρ={spearman_eps0:.3f}, MAE={mae_eps0:.3f}")

print(f"\nInterpretation:")
if pearson_bg > 0.9:
    print(f"  ✓ Bandgap: EXCELLENT correlation (r={pearson_bg:.3f})")
elif pearson_bg > 0.7:
    print(f"  ✓ Bandgap: GOOD correlation (r={pearson_bg:.3f})")
elif pearson_bg > 0.5:
    print(f"  ⚠ Bandgap: MODERATE correlation (r={pearson_bg:.3f})")
else:
    print(f"  ✗ Bandgap: WEAK correlation (r={pearson_bg:.3f})")

if pearson_eps > 0.9:
    print(f"  ✓ eps_inf: EXCELLENT correlation (r={pearson_eps:.3f})")
elif pearson_eps > 0.7:
    print(f"  ✓ eps_inf: GOOD correlation (r={pearson_eps:.3f})")
elif pearson_eps > 0.5:
    print(f"  ⚠ eps_inf: MODERATE correlation (r={pearson_eps:.3f})")
else:
    print(f"  ✗ eps_inf: WEAK correlation (r={pearson_eps:.3f})")

if pearson_eps0 > 0.9:
    print(f"  ✓ eps_0: EXCELLENT correlation (r={pearson_eps0:.3f})")
elif pearson_eps0 > 0.7:
    print(f"  ✓ eps_0: GOOD correlation (r={pearson_eps0:.3f})")
elif pearson_eps0 > 0.5:
    print(f"  ⚠ eps_0: MODERATE correlation (r={pearson_eps0:.3f})")
else:
    print(f"  ✗ eps_0: WEAK correlation (r={pearson_eps0:.3f})")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")

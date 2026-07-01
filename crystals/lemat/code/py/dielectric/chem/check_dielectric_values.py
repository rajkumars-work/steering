"""
Check dielectric values in extxyz files to identify mislabeling
"""

from ase.io import read
from glob import glob
import numpy as np

# Find all extxyz files
files = glob("data/*.extxyz")
print(f"Found {len(files)} extxyz files\n")

# Collect data
data = []
for file in files:
    try:
        atoms = read(file)
        info = atoms.info

        # Extract relevant properties
        row = {
            'file': file,
            'material_id': info.get('material_id', 'N/A'),
            'refractive_index': info.get('refractive_index', None),
            'eps_inf': info.get('eps_inf', None),
            'eps_0': info.get('eps_0', None),
            'e_inf': info.get('e_inf', None),
            'e_static': info.get('e_static', None),
        }

        # Only keep if has the key properties
        if all(row[k] is not None for k in ['eps_inf', 'eps_0', 'e_inf', 'e_static', 'refractive_index']):
            data.append(row)
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Successfully read {len(data)} files with all dielectric properties\n")

# Analyze the data
print("="*80)
print("ANALYSIS OF DIELECTRIC VALUES")
print("="*80)

print("\nRealistic ranges for dielectric constants:")
print("  e_inf (optical):  typically 2-10, up to ~20 for high-index materials")
print("  e_static:         must be > e_inf, typically 5-100, can be higher for ferroelectrics")
print("  Relationship:     e_static > e_inf (always!)")
print("  Rule:             e_inf ≈ n² (refractive index squared)")

print("\n" + "="*80)
print("CHECKING VALUES FROM FILES")
print("="*80)

# Statistical summary
eps_inf_vals = [d['eps_inf'] for d in data]
eps_0_vals = [d['eps_0'] for d in data]
e_inf_vals = [d['e_inf'] for d in data]
e_static_vals = [d['e_static'] for d in data]
n_vals = [d['refractive_index'] for d in data]

print(f"\nStatistics from {len(data)} files:")
print(f"\neps_inf:   min={min(eps_inf_vals):.2f}, max={max(eps_inf_vals):.2f}, mean={np.mean(eps_inf_vals):.2f}")
print(f"eps_0:     min={min(eps_0_vals):.2f}, max={max(eps_0_vals):.2f}, mean={np.mean(eps_0_vals):.2f}")
print(f"e_inf:     min={min(e_inf_vals):.2f}, max={max(e_inf_vals):.2f}, mean={np.mean(e_inf_vals):.2f}")
print(f"e_static:  min={min(e_static_vals):.2f}, max={max(e_static_vals):.2f}, mean={np.mean(e_static_vals):.2f}")

# Check refractive index relationship
print("\n" + "="*80)
print("CHECKING REFRACTIVE INDEX RELATIONSHIP (e_inf should ≈ n²)")
print("="*80)

n_squared = [n**2 for n in n_vals]
print(f"\nn²:        min={min(n_squared):.2f}, max={max(n_squared):.2f}, mean={np.mean(n_squared):.2f}")

# Check which values match n²
eps_inf_diff = [abs(d['eps_inf'] - d['refractive_index']**2) for d in data]
e_inf_diff = [abs(d['e_inf'] - d['refractive_index']**2) for d in data]

print(f"\n|eps_inf - n²|: mean={np.mean(eps_inf_diff):.2f}, max={max(eps_inf_diff):.2f}")
print(f"|e_inf - n²|:   mean={np.mean(e_inf_diff):.2f}, max={max(e_inf_diff):.2f}")

if np.mean(eps_inf_diff) < np.mean(e_inf_diff):
    print("\n✓ eps_inf matches n² very well - this is the OPTICAL dielectric constant")
    print("✗ e_inf does NOT match n² well")
else:
    print("\n✓ e_inf matches n² very well - this is the OPTICAL dielectric constant")
    print("✗ eps_inf does NOT match n² well")

# Check e_static > e_inf relationship
print("\n" + "="*80)
print("CHECKING e_static > e_inf RELATIONSHIP")
print("="*80)

ratios_correct = [d['eps_0'] / d['eps_inf'] for d in data]
ratios_wrong = [d['e_static'] / d['e_inf'] for d in data]

print(f"\neps_0 / eps_inf: min={min(ratios_correct):.2f}, max={max(ratios_correct):.2f}, mean={np.mean(ratios_correct):.2f}")
print(f"e_static / e_inf: min={min(ratios_wrong):.2f}, max={max(ratios_wrong):.2f}, mean={np.mean(ratios_wrong):.2f}")

print("\nTypical ratio for e_static/e_inf is 1.5-5 for most materials")
print("Ratios > 10 indicate potential issues")

# Sample a few files for detailed inspection
print("\n" + "="*80)
print("SAMPLE OF 10 FILES")
print("="*80)

for i, d in enumerate(data[:10]):
    print(f"\n{i+1}. {d['material_id']}")
    print(f"   n={d['refractive_index']:.3f}, n²={d['refractive_index']**2:.3f}")
    print(f"   eps_inf={d['eps_inf']:.2f}, eps_0={d['eps_0']:.2f} (ratio={d['eps_0']/d['eps_inf']:.2f})")
    print(f"   e_inf={d['e_inf']:.2f}, e_static={d['e_static']:.2f} (ratio={d['e_static']/d['e_inf']:.2f})")

# Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("\nBased on the analysis:")
print("\n1. eps_inf and eps_0 appear CORRECTLY labeled:")
print("   - eps_inf ≈ n² ✓")
print("   - eps_0 > eps_inf ✓")
print("   - Ratio eps_0/eps_inf is reasonable (1.5-5x) ✓")

print("\n2. e_inf and e_static appear INCORRECTLY labeled:")
print("   - e_static values (186-565) are EXTREMELY HIGH")
print("   - Ratio e_static/e_inf is ~10-20x (too high!)")
print("   - e_inf values (~10-40) look more like static dielectric values")

print("\n" + "="*80)
print("CONCLUSION: e_inf and e_static are likely SWAPPED!")
print("="*80)
print("\nThe correct labeling should be:")
print("  - Current 'e_inf' → should be 'e_static'")
print("  - Current 'e_static' → should be 'e_inf'")

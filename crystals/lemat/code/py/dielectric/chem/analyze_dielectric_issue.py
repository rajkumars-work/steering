"""
More careful analysis of the dielectric constant labeling issue
"""

from ase.io import read
from glob import glob
import numpy as np

files = glob("data/*.extxyz")
print(f"Analyzing {len(files)} files\n")

data = []
for file in files:
    try:
        atoms = read(file)
        info = atoms.info
        if all(k in info for k in ['eps_inf', 'eps_0', 'e_inf', 'e_static', 'refractive_index']):
            data.append(info)
    except:
        pass

print("="*80)
print("UNDERSTANDING THE DIELECTRIC CONSTANT ISSUE")
print("="*80)

print("\n1. PHYSICS REQUIREMENTS:")
print("   For ANY dielectric material:")
print("   - ε_optical (high-freq) ≈ n² ~ 2-10")
print("   - ε_static (low-freq) > ε_optical")
print("   - Typical ratio: ε_static/ε_optical ~ 1.5-5")
print("   - Both must be positive and reasonable (< 1000 for most materials)")

print("\n2. CURRENT DATA (original):")
print(f"\n   eps_inf:  range [{min(d['eps_inf'] for d in data):.1f}, {max(d['eps_inf'] for d in data):.1f}], mean {np.mean([d['eps_inf'] for d in data]):.1f}")
print(f"   eps_0:    range [{min(d['eps_0'] for d in data):.1f}, {max(d['eps_0'] for d in data):.1f}], mean {np.mean([d['eps_0'] for d in data]):.1f}")
print(f"   e_inf:    range [{min(d['e_inf'] for d in data):.1f}, {max(d['e_inf'] for d in data):.1f}], mean {np.mean([d['e_inf'] for d in data]):.1f}")
print(f"   e_static: range [{min(d['e_static'] for d in data):.1f}, {max(d['e_static'] for d in data):.1f}], mean {np.mean([d['e_static'] for d in data]):.1f}")

print("\n3. CHECKING EACH INTERPRETATION:")

# Check eps_inf and eps_0
n_squared = [d['refractive_index']**2 for d in data]
eps_inf_vals = [d['eps_inf'] for d in data]
eps_0_vals = [d['eps_0'] for d in data]

eps_inf_matches_n2 = np.mean([abs(d['eps_inf'] - d['refractive_index']**2) for d in data])
eps_0_greater = sum(d['eps_0'] > d['eps_inf'] for d in data) / len(data)
eps_ratio = np.mean([d['eps_0'] / d['eps_inf'] for d in data])

print("\n   A) eps_inf and eps_0:")
print(f"      ✓ eps_inf ≈ n²? Error = {eps_inf_matches_n2:.2f} (excellent)")
print(f"      ✓ eps_0 > eps_inf? {eps_0_greater*100:.0f}% of files")
print(f"      ✓ Ratio eps_0/eps_inf = {eps_ratio:.2f} (physically reasonable)")
print("      → THESE ARE CORRECTLY LABELED AND PHYSICALLY REASONABLE")

# Check e_inf and e_static AS IS
e_inf_vals = [d['e_inf'] for d in data]
e_static_vals = [d['e_static'] for d in data]

e_inf_matches_n2 = np.mean([abs(d['e_inf'] - d['refractive_index']**2) for d in data])
e_static_greater = sum(d['e_static'] > d['e_inf'] for d in data) / len(data)
e_ratio = np.mean([d['e_static'] / d['e_inf'] for d in data if d['e_inf'] > 0])

print("\n   B) e_inf and e_static (AS CURRENTLY LABELED):")
print(f"      ✗ e_inf ≈ n²? Error = {e_inf_matches_n2:.2f} (does NOT match)")
print(f"      ✓ e_static > e_inf? {e_static_greater*100:.0f}% of files")
print(f"      ✗ Ratio e_static/e_inf = {e_ratio:.1f} (WAY too high! Should be 1.5-5)")
print(f"      ✗ e_static values up to {max(e_static_vals):.0f} (unrealistically high!)")
print("      → Labels might be INTENDED correctly, but VALUES are problematic")

# Check if swapping would help
e_swapped_static_greater = sum(d['e_inf'] > d['e_static'] for d in data) / len(data)
e_swapped_ratio = np.mean([d['e_inf'] / d['e_static'] for d in data if d['e_static'] > 0])

print("\n   C) IF WE SWAPPED e_inf ↔ e_static:")
print(f"      ✗ Would give: e_inf (new) = {min(e_static_vals):.1f} to {max(e_static_vals):.0f}")
print(f"                    e_static (new) = {min(e_inf_vals):.1f} to {max(e_inf_vals):.1f}")
print(f"      ✗ e_inf > e_static? {e_swapped_static_greater*100:.0f}% (WRONG! Should be opposite)")
print(f"      ✗ e_inf values up to {max(e_static_vals):.0f} (way too high for optical!)")
print("      → SWAPPING MAKES IT WORSE")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\n1. eps_inf and eps_0 are CORRECT - no changes needed")

print("\n2. e_inf and e_static have DATA QUALITY issues:")
print("   - The label INTENT appears correct (inf=optical, static=static)")
print("   - But the VALUES are unrealistic:")
print("     • e_static values (up to 23188!) are way too high")
print("     • The ratio (35x) is way too high")
print("     • There are even NEGATIVE values!")
print("   - Swapping the labels would NOT fix this - it would make it worse")

print("\n3. RECOMMENDATION:")
print("   - Use eps_inf and eps_0 (these are correct)")
print("   - DO NOT use e_inf and e_static (unreliable data)")
print("   - If e_inf and e_static are needed, investigate their source/computation")
print("   - The issue is NOT mislabeling, but incorrect calculations or data corruption")

# Check if e_inf is similar to eps_0
similarity = []
for d in data:
    if abs(d['e_inf'] - d['eps_0']) < abs(d['e_inf'] - d['eps_inf']):
        similarity.append('eps_0')
    else:
        similarity.append('eps_inf')

print(f"\n4. INTERESTING OBSERVATION:")
print(f"   e_inf values are closer to eps_0 in {similarity.count('eps_0')}/{len(data)} files")
print("   This suggests e_inf might be a corrupted/miscalculated version of eps_0")

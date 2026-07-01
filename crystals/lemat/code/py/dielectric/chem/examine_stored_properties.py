"""
Examine what properties are stored in the data files
Check if there are multiple bandgap/epsilon values
"""
import numpy as np
from ase.io import read
from glob import glob

print("="*80)
print("EXAMINING STORED PROPERTIES IN DATA FILES")
print("="*80)

# Load structures from data directory
data_files = glob("data/*.extxyz")[:20]
print(f"\nExamining {len(data_files)} files from data directory...")

print(f"\n{'='*80}")
print("PROPERTIES IN EACH FILE")
print(f"{'='*80}")

all_properties = {}

for i, file in enumerate(data_files):
    atoms = read(file)
    mat_id = atoms.info.get("material_id", "unknown")

    print(f"\n{i+1}. {mat_id}")
    print(f"   File: {file}")

    # Check for different bandgap-related properties
    bg_props = []
    if "band_gap" in atoms.info:
        bg_props.append(f"band_gap={atoms.info['band_gap']:.4f}")
    if "bandgap" in atoms.info:
        bg_props.append(f"bandgap={atoms.info['bandgap']:.4f}")
    if "gap" in atoms.info:
        bg_props.append(f"gap={atoms.info['gap']:.4f}")
    if "bandgap_proxy" in atoms.info:
        bg_props.append(f"bandgap_proxy={atoms.info['bandgap_proxy']:.4f}")

    if bg_props:
        print(f"   Bandgap properties: {', '.join(bg_props)}")
    else:
        print(f"   Bandgap properties: NONE")

    # Check for epsilon-related properties
    eps_props = []
    if "eps_inf" in atoms.info:
        eps_props.append(f"eps_inf={atoms.info['eps_inf']:.4f}")
    if "eps_0" in atoms.info:
        eps_props.append(f"eps_0={atoms.info['eps_0']:.4f}")
    if "epsilon" in atoms.info:
        eps_props.append(f"epsilon={atoms.info['epsilon']:.4f}")
    if "epsilon_ionic" in atoms.info:
        eps_props.append(f"epsilon_ionic={atoms.info['epsilon_ionic']:.4f}")

    if eps_props:
        print(f"   Epsilon properties: {', '.join(eps_props)}")
    else:
        print(f"   Epsilon properties: NONE")

    # Collect all property keys
    for key in atoms.info.keys():
        if key not in all_properties:
            all_properties[key] = 0
        all_properties[key] += 1

print(f"\n{'='*80}")
print("SUMMARY: ALL PROPERTIES ACROSS FILES")
print(f"{'='*80}")

# Sort by frequency
sorted_props = sorted(all_properties.items(), key=lambda x: x[1], reverse=True)

print(f"\nFound {len(sorted_props)} unique properties:")
print(f"\n{'Property':<35} {'Count':<10} {'Frequency':<10}")
print("-"*60)
for prop, count in sorted_props:
    freq = count / len(data_files) * 100
    print(f"{prop:<35} {count:<10} {freq:.1f}%")

# Statistical comparison of different bandgap/epsilon values
print(f"\n{'='*80}")
print("COMPARING DIFFERENT VALUES (WHERE AVAILABLE)")
print(f"{'='*80}")

band_gaps = []
bandgaps = []
eps_infs = []
eps_0s = []

for file in data_files:
    atoms = read(file)
    if "band_gap" in atoms.info:
        band_gaps.append(atoms.info["band_gap"])
    if "bandgap" in atoms.info:
        bandgaps.append(atoms.info["bandgap"])
    if "eps_inf" in atoms.info:
        eps_infs.append(atoms.info["eps_inf"])
    if "eps_0" in atoms.info:
        eps_0s.append(atoms.info["eps_0"])

if band_gaps and bandgaps and len(band_gaps) == len(bandgaps):
    band_gaps = np.array(band_gaps)
    bandgaps = np.array(bandgaps)

    print(f"\n'band_gap' vs 'bandgap':")
    print(f"  band_gap:  min={band_gaps.min():.3f}, max={band_gaps.max():.3f}, mean={band_gaps.mean():.3f}")
    print(f"  bandgap:   min={bandgaps.min():.3f}, max={bandgaps.max():.3f}, mean={bandgaps.mean():.3f}")

    from scipy.stats import pearsonr
    r, p = pearsonr(band_gaps, bandgaps)
    print(f"  Correlation: r={r:.4f} (p={p:.2e})")

    diff = bandgaps - band_gaps
    print(f"  Difference (bandgap - band_gap):")
    print(f"    Mean: {diff.mean():.3f}")
    print(f"    MAE:  {np.abs(diff).mean():.3f}")

if eps_infs and eps_0s:
    eps_infs = np.array(eps_infs)
    eps_0s = np.array(eps_0s)

    print(f"\n'eps_inf' vs 'eps_0':")
    print(f"  eps_inf: min={eps_infs.min():.3f}, max={eps_infs.max():.3f}, mean={eps_infs.mean():.3f}")
    print(f"  eps_0:   min={eps_0s.min():.3f}, max={eps_0s.max():.3f}, mean={eps_0s.mean():.3f}")

    ratio = eps_0s / eps_infs
    print(f"  Ratio (eps_0 / eps_inf):")
    print(f"    Mean:   {ratio.mean():.3f}")
    print(f"    Median: {np.median(ratio):.3f}")
    print(f"    Min:    {ratio.min():.3f}")
    print(f"    Max:    {ratio.max():.3f}")

    # Check physical constraint
    valid = eps_0s > eps_infs
    print(f"  Physical constraint (eps_0 > eps_inf): {valid.sum()}/{len(valid)} ({valid.sum()/len(valid)*100:.1f}%)")

print(f"\n{'='*80}")
print("EXAMINATION COMPLETE")
print(f"{'='*80}")

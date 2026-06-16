#!/usr/bin/env python
"""Example: Computing static dielectric constant (ε₀) from ASE Atoms

This script demonstrates how to compute eps_0 (static dielectric constant)
for crystal structures using the di_props module.
"""

from ase.io import read
from di_props import compute_eps_0, compute_epsilon, compute_all_properties

def example_basic():
    """Basic example: Compute eps_0 for a single structure."""
    print("="*80)
    print("Example 1: Computing ε₀ (static dielectric constant)")
    print("="*80)

    # Load a structure
    atoms = read("data/mp-226-mp.extxyz")
    print(f"Structure: {atoms.get_chemical_formula()}")
    print(f"Material ID: {atoms.info.get('material_id', 'N/A')}")

    # Compute static dielectric constant
    eps_0 = compute_eps_0(atoms)
    print(f"\nComputed ε₀ (static): {eps_0:.4f}")

    # Compare with reference if available
    if 'eps_0' in atoms.info:
        ref_eps_0 = atoms.info['eps_0']
        print(f"Reference ε₀ (DFT):   {ref_eps_0:.4f}")
        print(f"Difference:           {abs(eps_0 - ref_eps_0):.4f}")


def example_detailed():
    """Detailed example: Show breakdown of epsilon vs eps_0."""
    print("\n" + "="*80)
    print("Example 2: Understanding ε∞ vs ε₀")
    print("="*80)

    atoms = read("data/K1I1-61374d62bcd91806d39bb2be-rg.extxyz")
    print(f"Structure: {atoms.get_chemical_formula()}")

    # Compute all properties
    props = compute_all_properties(atoms)

    print(f"\n{'Property':<30} {'Computed':>12} {'Reference':>12} {'Unit':>10}")
    print("-"*80)

    # Electronic dielectric constant
    ref_eps_inf = atoms.info.get('eps_inf', None)
    print(f"{'ε∞ (electronic)':<30} {props['epsilon']:>12.4f} ", end="")
    if ref_eps_inf:
        print(f"{ref_eps_inf:>12.4f}", end="")
    print(f"{'':>10}")

    # Static dielectric constant
    ref_eps_0 = atoms.info.get('eps_0', None)
    print(f"{'ε₀ (static)':<30} {props['eps_0']:>12.4f} ", end="")
    if ref_eps_0:
        print(f"{ref_eps_0:>12.4f}", end="")
    print(f"{'':>10}")

    # Ionic contribution
    ionic_contrib = props['eps_0'] - props['epsilon']
    if ref_eps_0 and ref_eps_inf:
        ref_ionic = ref_eps_0 - ref_eps_inf
        print(f"{'ε_ionic (contribution)':<30} {ionic_contrib:>12.4f} {ref_ionic:>12.4f}")
    else:
        print(f"{'ε_ionic (contribution)':<30} {ionic_contrib:>12.4f}")

    print()
    print(f"{'Bandgap':<30} {props['bandgap']:>12.4f} {'':>12} {'eV':>10}")
    print(f"{'Min TO phonon':<30} {props['min_TO_phonon']:>12.4f} {'':>12} {'THz':>10}")
    print(f"{'Max Born charge':<30} {props['max_born_charge']:>12.4f} {'':>12} {'e':>10}")


def example_comparison():
    """Compare electronic vs static dielectric for multiple structures."""
    print("\n" + "="*80)
    print("Example 3: Comparing ε∞ and ε₀ for multiple structures")
    print("="*80)

    structures = [
        "data/K1I1-61374d62bcd91806d39bb2be-rg.extxyz",
        "data/Ga1P1-61374862bcd91806d398534d-rg.extxyz",
        "data/mp-226-mp.extxyz",
    ]

    print(f"\n{'Formula':<15} {'ε∞':>10} {'ε₀':>10} {'ε_ionic':>10} {'Bandgap':>10}")
    print("-"*65)

    for fname in structures:
        try:
            atoms = read(fname)
            props = compute_all_properties(atoms)

            formula = atoms.get_chemical_formula()
            eps_inf = props['epsilon']
            eps_0 = props['eps_0']
            eps_ionic = eps_0 - eps_inf
            bandgap = props['bandgap']

            print(f"{formula:<15} {eps_inf:>10.2f} {eps_0:>10.2f} {eps_ionic:>10.2f} {bandgap:>10.2f}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nObservations:")
    print("- ε₀ > ε∞ always (ionic contribution is additive)")
    print("- Larger bandgaps typically correlate with smaller ε∞")
    print("- Ionic contribution varies widely depending on material")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DIELECTRIC CONSTANT CALCULATION EXAMPLES")
    print("="*80)
    print("\nKey Concepts:")
    print("- ε∞ (eps_inf/epsilon): Electronic dielectric constant (high-frequency)")
    print("- ε₀ (eps_0):           Static dielectric constant (low-frequency)")
    print("- Relationship:         ε₀ = ε∞ + ε_ionic")
    print()

    try:
        example_basic()
        example_detailed()
        example_comparison()

        print("\n" + "="*80)
        print("Examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: These examples require:")
        print("- GPU access for ML models")
        print("- Ray cluster initialization")
        print("- Structure files in data/ directory")


if __name__ == "__main__":
    main()

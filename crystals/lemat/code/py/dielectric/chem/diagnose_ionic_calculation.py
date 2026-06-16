#!/usr/bin/env python
"""Diagnose why ionic dielectric calculations are so large."""

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import ray
from physics import calculate_phonons, calculate_dielectric_properties, PhysicsActor
import numpy as np

def diagnose_structure(filename):
    """Diagnose ionic dielectric calculation for a structure."""
    print(f"\n{'='*80}")
    print(f"Diagnosing: {filename}")
    print(f"{'='*80}")

    atoms = read(filename)
    structure = AseAtomsAdaptor.get_structure(atoms)

    print(f"Formula: {structure.composition.formula}")
    print(f"Volume: {structure.volume:.2f} Å³")
    print(f"Number of atoms: {len(structure)}")

    # Reference values
    ref_eps_inf = atoms.info.get('eps_inf', None)
    ref_eps_0 = atoms.info.get('eps_0', None)
    if ref_eps_inf and ref_eps_0:
        ref_ionic = ref_eps_0 - ref_eps_inf
        print(f"\nReference (DFT):")
        print(f"  eps_inf: {ref_eps_inf:.4f}")
        print(f"  eps_0:   {ref_eps_0:.4f}")
        print(f"  ionic:   {ref_ionic:.4f}")

    # Setup Ray and actor
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    actor = PhysicsActor.options(num_gpus=1).remote(model_name="egip-inf")

    # Calculate phonons
    print(f"\nCalculating phonons...")
    phonon_future = calculate_phonons.remote(
        actor, structure, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    phonon_data = ray.get(phonon_future)

    # Calculate dielectric
    print(f"Calculating dielectric properties...")
    diel_future = calculate_dielectric_properties.remote(structure, phonon_data)
    diel_data = ray.get(diel_future)

    # Analyze results
    print(f"\n{'='*80}")
    print("PHONON ANALYSIS:")
    print(f"{'='*80}")

    freqs = phonon_data['frequencies_gamma']
    print(f"Gamma-point frequencies (THz):")
    print(f"  Acoustic (3 lowest): {sorted(np.abs(freqs))[:3]}")
    print(f"  Optical (rest): {sorted(np.abs(freqs))[3:]}")
    print(f"  Lowest optical: {phonon_data['lowest_optical_frequency_THz']:.4f} THz")

    print(f"\n{'='*80}")
    print("BORN EFFECTIVE CHARGES:")
    print(f"{'='*80}")

    born_charges = diel_data['born_effective_charges']
    born_method = diel_data['born_method']
    print(f"Method: {born_method}")
    print(f"Charges: {born_charges}")
    print(f"Max |Z*|: {max(np.abs(born_charges)):.2f}")
    print(f"Mean |Z*|: {np.mean(np.abs(born_charges)):.2f}")

    # Compare with typical Born charges
    print(f"\nTypical Born effective charges:")
    print(f"  Simple ionic: ~1-3 for alkali halides")
    print(f"  Covalent: ~1-2 for semiconductors")
    print(f"  Highly polar: up to ~6-8 for ferroelectrics")

    print(f"\n{'='*80}")
    print("IONIC DIELECTRIC CALCULATION:")
    print(f"{'='*80}")

    ionic_tensor = diel_data['ionic_dielectric_constant_nominal']
    if ionic_tensor is not None:
        ionic_tensor = np.array(ionic_tensor)
        ionic_diag = np.diag(ionic_tensor)
        ionic_mean = np.mean(ionic_diag)

        print(f"Ionic dielectric tensor diagonal: {ionic_diag}")
        print(f"Mean ionic contribution: {ionic_mean:.4f}")

        mode_contribs = diel_data.get('mode_dielectric_contributions', [])
        if mode_contribs:
            mode_contribs_sorted = sorted([c for c in mode_contribs if c > 0], reverse=True)
            print(f"\nTop 5 mode contributions:")
            for i, contrib in enumerate(mode_contribs_sorted[:5]):
                print(f"  Mode {i+1}: {contrib:.4f}")

            total_contrib = sum(mode_contribs)
            print(f"\nTotal from all modes: {total_contrib:.4f}")

            if ref_ionic:
                print(f"Reference ionic (DFT): {ref_ionic:.4f}")
                print(f"Ratio (calc/ref): {ionic_mean/ref_ionic:.2f}x")

    # Check formula
    print(f"\n{'='*80}")
    print("FORMULA CHECK:")
    print(f"{'='*80}")

    # The formula is: ε_ionic = Σ_m (e^2 / (ε₀ V ω_m^2)) * |Z* · u_m|^2
    # Large contributions come from:
    # 1. Large Born charges (Z*)
    # 2. Low frequencies (ω_m)
    # 3. Large mode polarity (Z* · u_m)

    print("Contributions scale as:")
    print("  - Z*² (Born charge squared)")
    print("  - 1/ω² (inverse frequency squared)")
    print("  - 1/V (inverse volume)")

    if ionic_tensor is not None and ref_ionic:
        print(f"\nPossible issues:")
        if ionic_mean / ref_ionic > 10:
            print("  ✗ Ionic contribution is >10x too large!")
            print("    Likely causes:")
            print("    - Born charges estimated from oxidation states are wrong")
            print("    - Should use DFT Born charges from VASP DFPT")
            print("    - Phonon frequencies may be too soft")

    ray.kill(actor)

    return {
        'ref_ionic': ref_ionic if ref_ionic else 0,
        'calc_ionic': ionic_mean if ionic_tensor is not None else 0,
        'born_charges': born_charges,
        'born_method': born_method,
        'lowest_optical_freq': phonon_data['lowest_optical_frequency_THz']
    }


def main():
    print("="*80)
    print("DIAGNOSTIC: Ionic Dielectric Calculation")
    print("="*80)
    print("\nThis script diagnoses why ionic dielectric contributions are overestimated.")

    # Test on KI (worst case)
    result = diagnose_structure("data/K1I1-61374d62bcd91806d39bb2be-rg.extxyz")

    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print(f"{'='*80}")
    print(f"The ionic dielectric calculation uses ESTIMATED Born charges from")
    print(f"oxidation states, not DFT-calculated Born effective charges.")
    print(f"")
    print(f"The stored eps_0 values in the data are from VASP DFPT calculations")
    print(f"which use accurate DFT Born charges.")
    print(f"")
    print(f"RECOMMENDATION:")
    print(f"  1. Extract DFT Born charges from Materials Project if available")
    print(f"  2. Use those instead of estimated oxidation state charges")
    print(f"  3. OR: Just use the egip_eps model for eps_inf, don't compute eps_0")
    print(f"  4. OR: Train a direct surrogate model for eps_0 (like egip_eps for eps_inf)")


if __name__ == "__main__":
    main()

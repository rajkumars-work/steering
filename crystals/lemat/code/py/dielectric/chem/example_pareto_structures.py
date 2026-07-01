#!/usr/bin/env python3
"""
Example: Generate High-Dielectric Crystal Structures using Pareto Front Selection

This script demonstrates the complete workflow:
1. Load the dielectric dataset
2. Use trained models to predict eps_0 and energy_above_hull
3. Find Pareto front compositions (high dielectric, low energy)
4. Generate crystal structures using prototype-based approach
"""

from structure_generator import (
    load_dataset,
    deduplicate_by_composition,
    load_trained_model,
    predict_properties,
    find_pareto_front,
    generate_structure
)
import numpy as np
import os
from ase.io import write

# ============================================================================
# Configuration
# ============================================================================

EXTXYZ_PATH = "/data/assets/datasets/dielectric/mp/*.extxyz"
N_SAMPLES = 1000  # Sample size for faster testing (None = all)
MIN_EPS_0 = 15.0  # Minimum dielectric constant
MAX_ENERGY = 0.05  # Maximum energy above hull (eV/atom)
OUTPUT_DIR = "pareto_structures"

# ============================================================================
# Main Workflow
# ============================================================================

def main():
    print("="*80)
    print("HIGH-DIELECTRIC CRYSTAL STRUCTURE GENERATION")
    print("Using Pareto Front Selection Strategy")
    print("="*80)

    # Step 1: Load Dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_dataset(EXTXYZ_PATH)
    dataset = deduplicate_by_composition(dataset)

    # Sample for faster testing
    if N_SAMPLES and N_SAMPLES < len(dataset):
        print(f"Sampling {N_SAMPLES} structures...")
        import random
        random.seed(42)
        dataset = random.sample(dataset, N_SAMPLES)

    atoms_list = [atoms for atoms, _ in dataset]
    pmg_structures = [pmg_struct for _, pmg_struct in dataset]

    # Step 2: Load Trained Models
    print("\n[2/6] Loading trained XGBoost models...")
    models_dict = {}

    for model_name in ['eps_0', 'energy_above_hull']:
        try:
            model, metadata = load_trained_model(model_name)
            models_dict[model_name] = (model, metadata)
            print(f"  ✓ Loaded {model_name} model")
            if metadata:
                print(f"    - Spearman: {metadata.get('spearman', 0):.3f}")
                print(f"    - MAE: {metadata.get('mae', 0):.3f}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            print("\nPlease run composition.py first to train the models:")
            print("  python composition.py")
            return

    # Step 3: Predict Properties
    print("\n[3/6] Predicting properties for all structures...")
    predictions = predict_properties(atoms_list, models_dict)

    eps_0_pred = predictions['eps_0']
    energy_pred = predictions['energy_above_hull']

    print(f"\nPrediction statistics:")
    print(f"  eps_0:")
    print(f"    Range: {eps_0_pred.min():.2f} - {eps_0_pred.max():.2f}")
    print(f"    Mean: {eps_0_pred.mean():.2f}")
    print(f"    Median: {np.median(eps_0_pred):.2f}")
    print(f"  energy_above_hull:")
    print(f"    Range: {energy_pred.min():.4f} - {energy_pred.max():.4f}")
    print(f"    Mean: {energy_pred.mean():.4f}")
    print(f"    Median: {np.median(energy_pred):.4f}")

    # Step 4: Find Pareto Front
    print("\n[4/6] Finding Pareto front candidates...")
    print(f"  Criteria: eps_0 >= {MIN_EPS_0}, energy_above_hull <= {MAX_ENERGY}")

    pareto_candidates = find_pareto_front(
        atoms_list, eps_0_pred, energy_pred,
        min_eps_0=MIN_EPS_0, max_energy=MAX_ENERGY
    )

    if not pareto_candidates:
        print("\n⚠ No Pareto front candidates found!")
        print("Try relaxing the criteria (lower min_eps_0 or higher max_energy)")

        # Show distribution to help adjust criteria
        print("\nDistribution of predictions:")
        print(f"  Structures with eps_0 >= {MIN_EPS_0}: {np.sum(eps_0_pred >= MIN_EPS_0)}")
        print(f"  Structures with E_hull <= {MAX_ENERGY}: {np.sum(energy_pred <= MAX_ENERGY)}")
        print(f"  Structures meeting BOTH: {np.sum((eps_0_pred >= MIN_EPS_0) & (energy_pred <= MAX_ENERGY))}")
        return

    # Step 5: Display Top Candidates
    print(f"\n[5/6] Top {min(20, len(pareto_candidates))} Pareto front candidates:")
    print("-"*80)
    print(f"{'Rank':<6} {'Formula':<25} {'eps_0':<12} {'E_hull':<12} {'Score':<10}")
    print("-"*80)

    for i, (atoms, eps_0, energy) in enumerate(pareto_candidates[:20]):
        formula = atoms.get_chemical_formula()
        # Score: weighted combination (higher eps_0, lower energy is better)
        score = eps_0 - 10 * energy  # Adjust weights as needed
        print(f"{i+1:<6} {formula:<25} {eps_0:>10.2f}  {energy:>10.4f}  {score:>8.2f}")

    # Step 6: Save Structures
    print(f"\n[6/6] Saving top structures to {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved_count = 0
    for i, (atoms, eps_0, energy) in enumerate(pareto_candidates[:10]):
        formula = atoms.get_chemical_formula()
        output_file = os.path.join(OUTPUT_DIR, f"pareto_{i+1:02d}_{formula}_eps{eps_0:.1f}.cif")

        try:
            write(output_file, atoms)
            saved_count += 1
            if i < 5:  # Print details for top 5
                print(f"\n  {i+1}. {formula}")
                print(f"     eps_0 (predicted): {eps_0:.2f}")
                print(f"     E_hull (predicted): {energy:.4f} eV/atom")
                print(f"     Saved to: {output_file}")
        except Exception as e:
            print(f"  ✗ Error saving {formula}: {e}")

    # Step 7: Generate Structure for Novel Composition (Example)
    print("\n" + "="*80)
    print("BONUS: Example of Structure Generation for Novel Composition")
    print("="*80)

    if pareto_candidates:
        # Use top candidate as inspiration
        template_atoms, template_eps, template_energy = pareto_candidates[0]
        template_formula = template_atoms.get_chemical_formula()

        print(f"\nBest Pareto candidate: {template_formula}")
        print(f"  eps_0: {template_eps:.2f}")
        print(f"  E_hull: {template_energy:.4f}")

        # Example: Generate structure for same composition using nearest prototype
        # (In practice, you'd use a novel composition from composition_generator.py)
        target_formula = template_formula

        print(f"\nGenerating structure for: {target_formula}")
        print("Using prototype-based structure generation...")

        new_structure = generate_structure(target_formula, pmg_structures)

        if new_structure:
            output_file = os.path.join(OUTPUT_DIR, f"generated_{target_formula}.cif")
            new_structure.to(filename=output_file)
            print(f"✓ Generated structure saved!")
            print(f"  File: {output_file}")
            print(f"  Formula: {new_structure.composition.reduced_formula}")
            print(f"  Space group: {new_structure.get_space_group_info()}")
        else:
            print(f"✗ Failed to generate structure for {target_formula}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Dataset size: {len(atoms_list)} structures")
    print(f"  Pareto front candidates: {len(pareto_candidates)}")
    print(f"  Saved structures: {saved_count}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Examine saved structures in pareto_structures/")
    print("  2. Use composition_generator.py to create novel compositions")
    print("  3. Use these high-performing prototypes for structure generation")
    print("  4. Run DFT calculations to validate predictions")
    print("="*80)


if __name__ == "__main__":
    main()

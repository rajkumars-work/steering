from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

import random
import copy
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

# Example: load one structure
structure_path = "pipeline_stage5_filtered.csv"  # or specific path
atoms = read(structure_path)



def generate_perturbed_variants(atoms, n_variants=2, max_displacement=0.05, max_scale=0.02):
    """
    atoms: ASE Atoms object
    n_variants: number of new perturbed structures
    max_displacement: max atomic displacement in Å
    max_scale: max lattice scaling fraction (e.g., 0.02 = 2%)
    """
    variants = []

    for i in range(n_variants):
        new_atoms = atoms.copy()

        # Perturb atomic positions
        displacements = np.random.uniform(-max_displacement, max_displacement, size=new_atoms.positions.shape)
        new_atoms.positions += displacements

        # Small lattice scaling
        scale_factor = 1 + random.uniform(-max_scale, max_scale)
        new_atoms.set_cell(new_atoms.get_cell() * scale_factor, scale_atoms=True)

        variants.append(new_atoms)

    return variants


variants_pmg = [AseAtomsAdaptor.get_structure(v) for v in variants]

# Example ASE workflow
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator

# Replace this with your MLIP calculator
calc = MLIP_Calculator()  # pseudocode

for i, variant in enumerate(variants):
    variant.set_calculator(calc)
    dyn = BFGS(variant)
    dyn.run(fmax=0.05)  # force convergence
    variant.write(f"relaxed_variant_{i}.extxyz")



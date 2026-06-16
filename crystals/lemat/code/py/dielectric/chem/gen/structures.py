"""
Prototype-based structure generator.

Strategy:
---------
1. Find nearest known composition.
2. Copy its structure.
3. Substitute elements.
4. Optionally scale lattice.

This produces HIGH-PROBABILITY crystals.

Integration with Dataset:
------------------------
This module now integrates with the dielectric dataset from composition.py
and can generate structures for high-dielectric compositions identified
through Pareto front analysis (high eps_0, low energy_above_hull).
"""

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from ase.io import read as ase_read
from ase import Atoms
from glob import glob
import numpy as np
import os
import pickle
import json
import xgboost as xgb
from typing import List, Dict, Tuple, Optional
import warnings


# -------------------------------------------------
# Helper: composition distance
# -------------------------------------------------

def composition_distance(comp1, comp2):
    """
    L1 distance between fractional composition vectors.
    """
    c1 = Composition(comp1).fractional_composition
    c2 = Composition(comp2).fractional_composition

    elements = set(c1.elements + c2.elements)

    dist = 0
    for el in elements:
        dist += abs(
            c1.get_atomic_fraction(el) -
            c2.get_atomic_fraction(el)
        )

    return dist


# -------------------------------------------------
# Find nearest prototype
# -------------------------------------------------

def find_nearest_structure(target_formula, structures):
    """
    structures: list of pymatgen Structure objects
    """

    best = None
    best_dist = 1e9

    for s in structures:
        d = composition_distance(target_formula, s.composition)

        if d < best_dist:
            best_dist = d
            best = s

    return best


# -------------------------------------------------
# Substitute species
# -------------------------------------------------

def substitute_structure(prototype, target_formula):
    """
    Replace elements while keeping coordinates.
    """

    proto_comp = prototype.composition
    target_comp = Composition(target_formula)

    proto_elements = sorted(proto_comp.elements,
                            key=lambda e: proto_comp[e],
                            reverse=True)

    target_elements = sorted(target_comp.elements,
                             key=lambda e: target_comp[e],
                             reverse=True)

    if len(proto_elements) != len(target_elements):
        return None

    mapping = {
        str(proto_elements[i]): str(target_elements[i])
        for i in range(len(proto_elements))
    }

    new_structure = prototype.copy()
    new_structure.replace_species(mapping)

    return new_structure


# -------------------------------------------------
# Optional lattice scaling
# -------------------------------------------------

def scale_lattice(structure):
    """
    crude volume scaling based on atomic radii
    """

    radii = [
        site.specie.atomic_radius or 1.5
        for site in structure
    ]

    avg_radius = np.mean(radii)

    scale = avg_radius / 1.5

    structure.scale_lattice(
        structure.volume * scale**3
    )

    return structure


# -------------------------------------------------
# Master generator
# -------------------------------------------------

def generate_structure(target_formula, known_structures):
    proto = find_nearest_structure(
        target_formula,
        known_structures
    )

    if proto is None:
        return None

    new_struct = substitute_structure(
        proto,
        target_formula
    )

    if new_struct is None:
        return None

    new_struct = scale_lattice(new_struct)

    return new_struct


# -------------------------------------------------
# High-throughput generation (CPU/GPU)
# -------------------------------------------------

def _composition_vector(comp, element_index):
    """
    Build fractional composition vector for a pymatgen Composition.
    """
    vec = np.zeros(len(element_index), dtype=np.float32)
    for el in comp.elements:
        key = str(el)
        idx = element_index.get(key)
        if idx is not None:
            vec[idx] = comp.get_atomic_fraction(el)
    return vec


class PrototypeIndex:
    """
    Precomputed index for fast nearest-prototype lookup.
    Groups prototypes by number of elements and caches composition matrices.
    """
    def __init__(self, structures: List[Structure]):
        self.groups = {}

        by_n_elements = {}
        for s in structures:
            n = len(s.composition.elements)
            by_n_elements.setdefault(n, []).append(s)

        for n, structs in by_n_elements.items():
            elements = sorted(
                {str(el) for s in structs for el in s.composition.elements}
            )
            element_index = {el: i for i, el in enumerate(elements)}
            matrix = np.vstack([
                _composition_vector(s.composition.fractional_composition, element_index)
                for s in structs
            ])
            self.groups[n] = {
                "structures": structs,
                "elements": elements,
                "element_index": element_index,
                "matrix": matrix,
            }


def _nearest_indices_numpy(target_matrix, proto_matrix, chunk_size=1024):
    """
    Find nearest prototype indices using L1 distance (CPU, numpy).
    """
    n_targets = target_matrix.shape[0]
    nearest = np.empty(n_targets, dtype=np.int32)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        chunk = target_matrix[start:end]
        # distances shape: (chunk, n_proto)
        dists = np.abs(chunk[:, None, :] - proto_matrix[None, :, :]).sum(axis=2)
        nearest[start:end] = np.argmin(dists, axis=1)

    return nearest


def _nearest_indices_torch(target_matrix, proto_matrix, chunk_size=4096, device="cuda"):
    """
    Find nearest prototype indices using L1 distance (GPU via torch).
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Torch is required for GPU distance computation") from e

    t_proto = torch.tensor(proto_matrix, device=device)
    n_targets = target_matrix.shape[0]
    nearest = np.empty(n_targets, dtype=np.int32)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        chunk = torch.tensor(target_matrix[start:end], device=device)
        # torch.cdist supports p=1 for L1 distance
        dists = torch.cdist(chunk, t_proto, p=1)
        idx = torch.argmin(dists, dim=1).cpu().numpy()
        nearest[start:end] = idx

    return nearest


def _nearest_topk_numpy(target_matrix, proto_matrix, k, chunk_size=1024):
    """
    Find top-k nearest prototype indices using L1 distance (CPU, numpy).
    Returns (indices, distances) arrays of shape (n_targets, k).
    """
    n_targets = target_matrix.shape[0]
    n_proto = proto_matrix.shape[0]
    k = min(k, n_proto)
    indices = np.empty((n_targets, k), dtype=np.int32)
    distances = np.empty((n_targets, k), dtype=np.float32)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        chunk = target_matrix[start:end]
        dists = np.abs(chunk[:, None, :] - proto_matrix[None, :, :]).sum(axis=2)
        idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row = np.arange(idx.shape[0])[:, None]
        sort_order = np.argsort(dists[row, idx], axis=1)
        idx = idx[row, sort_order]
        indices[start:end] = idx
        distances[start:end] = dists[row, idx]

    return indices, distances


def _nearest_topk_torch(target_matrix, proto_matrix, k, chunk_size=4096, device="cuda"):
    """
    Find top-k nearest prototype indices using L1 distance (GPU via torch).
    Returns (indices, distances) arrays of shape (n_targets, k).
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Torch is required for GPU distance computation") from e

    t_proto = torch.tensor(proto_matrix, device=device)
    n_targets = target_matrix.shape[0]
    n_proto = proto_matrix.shape[0]
    k = min(k, n_proto)
    indices = np.empty((n_targets, k), dtype=np.int32)
    distances = np.empty((n_targets, k), dtype=np.float32)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        chunk = torch.tensor(target_matrix[start:end], device=device)
        dists = torch.cdist(chunk, t_proto, p=1)
        vals, idx = torch.topk(dists, k=k, largest=False)
        indices[start:end] = idx.cpu().numpy()
        distances[start:end] = vals.cpu().numpy()

    return indices, distances


def generate_structures_for_compositions(
    target_formulas: List[str],
    known_structures: List[Structure],
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
    chunk_size: int = 1024,
):
    """
    High-throughput structure generation for many compositions.

    Args:
        target_formulas: List of target compositions (formula strings).
        known_structures: List of pymatgen Structure prototypes.
        n_jobs: Number of parallel workers for substitution/scaling (CPU).
        use_gpu: If True, use torch on GPU for nearest-prototype search.
        gpu_device: Torch device string (e.g., "cuda", "cuda:0").
        chunk_size: Batch size for distance computation.

    Returns:
        List of pymatgen Structure objects (or None for failures).
    """
    if not target_formulas:
        return []

    index = PrototypeIndex(known_structures)
    results = [None] * len(target_formulas)

    # Group targets by number of elements
    targets_by_n = {}
    parsed = []
    for i, formula in enumerate(target_formulas):
        comp = Composition(formula).fractional_composition
        n = len(comp.elements)
        parsed.append(comp)
        targets_by_n.setdefault(n, []).append(i)

    for n, idxs in targets_by_n.items():
        group = index.groups.get(n)
        if not group:
            for i in idxs:
                results[i] = None
            continue

        element_index = group["element_index"]
        proto_matrix = group["matrix"]
        structures = group["structures"]

        target_matrix = np.vstack([
            _composition_vector(parsed[i], element_index) for i in idxs
        ])

        if use_gpu:
            nearest = _nearest_indices_torch(
                target_matrix, proto_matrix, chunk_size=chunk_size, device=gpu_device
            )
        else:
            nearest = _nearest_indices_numpy(
                target_matrix, proto_matrix, chunk_size=chunk_size
            )

        tasks = [(structures[nearest[j]], target_formulas[idxs[j]]) for j in range(len(idxs))]

        def _generate_one(proto, formula):
            new_struct = substitute_structure(proto, formula)
            if new_struct is None:
                return None
            return scale_lattice(new_struct)

        if n_jobs is None:
            n_jobs = os.cpu_count()

        if n_jobs and n_jobs > 1:
            from joblib import Parallel, delayed
            generated = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_generate_one)(proto, formula) for proto, formula in tasks
            )
        else:
            generated = [_generate_one(proto, formula) for proto, formula in tasks]

        for j, i in enumerate(idxs):
            results[i] = generated[j]

    return results


def generate_structures_for_compositions_topk(
    target_formulas: List[str],
    known_structures: List[Structure],
    k: int = 10,
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
    chunk_size: int = 1024,
):
    """
    Generate up to k structures per composition using nearest prototypes.

    Returns:
        List of lists. Each inner list contains dicts:
        {
          "structure": pymatgen Structure | None,
          "prototype": pymatgen Structure,
          "prototype_formula": str,
          "distance": float
        }
    """
    if not target_formulas:
        return []

    index = PrototypeIndex(known_structures)
    results = [[] for _ in target_formulas]

    targets_by_n = {}
    parsed = []
    for i, formula in enumerate(target_formulas):
        comp = Composition(formula).fractional_composition
        n = len(comp.elements)
        parsed.append(comp)
        targets_by_n.setdefault(n, []).append(i)

    def _generate_one(proto, formula):
        new_struct = substitute_structure(proto, formula)
        if new_struct is None:
            return None
        return scale_lattice(new_struct)

    if n_jobs is None:
        n_jobs = os.cpu_count()

    for n, idxs in targets_by_n.items():
        group = index.groups.get(n)
        if not group:
            continue

        element_index = group["element_index"]
        proto_matrix = group["matrix"]
        structures = group["structures"]

        target_matrix = np.vstack([
            _composition_vector(parsed[i], element_index) for i in idxs
        ])

        if use_gpu:
            nearest_idx, nearest_dist = _nearest_topk_torch(
                target_matrix, proto_matrix, k=k, chunk_size=chunk_size, device=gpu_device
            )
        else:
            nearest_idx, nearest_dist = _nearest_topk_numpy(
                target_matrix, proto_matrix, k=k, chunk_size=chunk_size
            )

        tasks = []
        task_meta = []
        for local_i, target_i in enumerate(idxs):
            for j in range(nearest_idx.shape[1]):
                proto = structures[int(nearest_idx[local_i, j])]
                tasks.append((proto, target_formulas[target_i]))
                task_meta.append((target_i, proto, float(nearest_dist[local_i, j])))

        if n_jobs and n_jobs > 1 and tasks:
            from joblib import Parallel, delayed
            generated = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_generate_one)(proto, formula) for proto, formula in tasks
            )
        else:
            generated = [_generate_one(proto, formula) for proto, formula in tasks]

        for (target_i, proto, dist), new_struct in zip(task_meta, generated):
            results[target_i].append(
                {
                    "structure": new_struct,
                    "prototype": proto,
                    "prototype_formula": proto.composition.reduced_formula,
                    "distance": dist,
                }
            )

    return results


# -------------------------------------------------
# Dataset Loading and Integration
# -------------------------------------------------

def load_dataset(extxyz_path="/data/assets/datasets/dielectric/mp/*.extxyz"):
    """
    Load the dielectric dataset from extxyz files.

    Returns:
        List of (ASE Atoms, pymatgen Structure) tuples with properties
    """
    print(f"Loading dataset from {extxyz_path}...")
    files = glob(extxyz_path)
    print(f"Found {len(files)} extxyz files")

    dataset = []
    for file in files:
        atoms_list = ase_read(file, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        for atoms in atoms_list:
            # Convert to pymatgen Structure
            try:
                pmg_struct = Structure(
                    lattice=atoms.cell,
                    species=atoms.get_chemical_symbols(),
                    coords=atoms.get_scaled_positions(),
                    coords_are_cartesian=False
                )
                dataset.append((atoms, pmg_struct))
            except Exception as e:
                warnings.warn(f"Could not convert structure: {e}")
                continue

    print(f"Loaded {len(dataset)} structures")
    return dataset


def deduplicate_by_composition(dataset, keep_lowest_energy=True):
    """
    Deduplicate dataset by composition, keeping only the lowest energy_above_hull.

    Args:
        dataset: List of (atoms, pmg_struct) tuples
        keep_lowest_energy: If True, keep structure with lowest energy_above_hull

    Returns:
        Deduplicated list of (atoms, pmg_struct) tuples
    """
    print("Deduplicating by composition...")

    composition_dict = {}
    for atoms, pmg_struct in dataset:
        comp_str = atoms.get_chemical_formula()
        energy = atoms.info.get('energy_above_hull', float('inf'))

        if comp_str not in composition_dict:
            composition_dict[comp_str] = (atoms, pmg_struct)
        elif keep_lowest_energy:
            existing_energy = composition_dict[comp_str][0].info.get('energy_above_hull', float('inf'))
            if energy < existing_energy:
                composition_dict[comp_str] = (atoms, pmg_struct)

    result = list(composition_dict.values())
    print(f"After deduplication: {len(result)} unique compositions")
    return result


def load_trained_model(model_name='eps_0'):
    """
    Load a trained XGBoost model from composition.py.

    Args:
        model_name: One of 'bandgap', 'energy_above_hull', 'log_eps_0', 'eps_0'

    Returns:
        (model, metadata) tuple
    """
    model_file = f"data/xgb_composition_{model_name}.json"
    metadata_file = f"xgb_composition_metadata_{model_name}.json"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            f"Please run composition.py first to train models."
        )

    print(f"Loading model: {model_file}")
    model = xgb.XGBRegressor()
    model.load_model(model_file)

    metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    return model, metadata


def compute_composition_features(atoms):
    """
    Compute composition features for a structure using matminer.

    This should match the feature computation in composition.py.
    """
    from pymatgen.core import Composition as PMGComposition
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.composition import Stoichiometry
    from matminer.featurizers.composition import ValenceOrbital
    from matminer.featurizers.composition import IonProperty

    # Initialize featurizers (same as composition.py)
    magpie_featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    stoich_featurizer = Stoichiometry()
    valence_featurizer = ValenceOrbital(impute_nan=True)
    ionic_featurizer = IonProperty(impute_nan=True)

    # Get composition
    formula = atoms.get_chemical_formula()
    comp = PMGComposition(formula)

    features = []

    # 1. Magpie features
    try:
        magpie_feats = magpie_featurizer.featurize(comp)
        features.extend(magpie_feats)
    except Exception as e:
        features.extend([0] * len(magpie_featurizer.feature_labels()))

    # 2. Stoichiometry features
    try:
        stoich_feats = stoich_featurizer.featurize(comp)
        features.extend(stoich_feats)
    except Exception as e:
        features.extend([0] * len(stoich_featurizer.feature_labels()))

    # 3. Valence orbital fractions
    try:
        valence_feats = valence_featurizer.featurize(comp)
        features.extend(valence_feats)
    except Exception as e:
        features.extend([0] * len(valence_featurizer.feature_labels()))

    # 4. Ionic character
    try:
        ionic_feats = ionic_featurizer.featurize(comp)
        features.extend(ionic_feats)
    except Exception as e:
        features.extend([0] * len(ionic_featurizer.feature_labels()))

    return np.array(features)


def predict_properties(atoms_list, models_dict):
    """
    Predict properties for a list of structures.

    Args:
        atoms_list: List of ASE Atoms objects
        models_dict: Dictionary mapping property names to (model, metadata) tuples

    Returns:
        Dictionary mapping property names to arrays of predictions
    """
    from joblib import Parallel, delayed

    print(f"Computing features for {len(atoms_list)} structures...")
    X = Parallel(n_jobs=os.cpu_count())(
        delayed(compute_composition_features)(atoms) for atoms in atoms_list
    )
    X = np.vstack(X)

    print(f"Predicting properties...")
    predictions = {}
    for prop_name, (model, metadata) in models_dict.items():
        pred = model.predict(X)

        # Apply inverse transform if needed
        if metadata and metadata.get('log_transform', False):
            pred = np.expm1(pred)  # Inverse of log1p

        predictions[prop_name] = pred

    return predictions


def find_pareto_front(atoms_list, eps_0_values, energy_above_hull_values,
                      min_eps_0=10.0, max_energy=0.1):
    """
    Find structures on the Pareto front (high eps_0, low energy_above_hull).

    Args:
        atoms_list: List of ASE Atoms objects
        eps_0_values: Array of dielectric constant predictions
        energy_above_hull_values: Array of energy above hull predictions
        min_eps_0: Minimum dielectric constant threshold
        max_energy: Maximum energy above hull threshold

    Returns:
        List of (atoms, eps_0, energy) tuples on the Pareto front
    """
    print("\nFinding Pareto front candidates...")
    print(f"Criteria: eps_0 >= {min_eps_0}, energy_above_hull <= {max_energy}")

    pareto_candidates = []

    for i, atoms in enumerate(atoms_list):
        eps_0 = eps_0_values[i]
        energy = energy_above_hull_values[i]

        # Apply filters
        if eps_0 < min_eps_0 or energy > max_energy:
            continue

        # Check if dominated by existing candidates
        is_dominated = False
        for _, existing_eps, existing_energy in pareto_candidates:
            # Dominated if another point has both higher eps_0 AND lower energy
            if existing_eps >= eps_0 and existing_energy <= energy:
                if existing_eps > eps_0 or existing_energy < energy:
                    is_dominated = True
                    break

        if not is_dominated:
            # Remove candidates dominated by this one
            pareto_candidates = [
                (a, e, en) for a, e, en in pareto_candidates
                if not (eps_0 >= e and energy <= en and (eps_0 > e or energy < en))
            ]
            pareto_candidates.append((atoms, eps_0, energy))

    print(f"Found {len(pareto_candidates)} Pareto front candidates")

    # Sort by eps_0 (descending)
    pareto_candidates.sort(key=lambda x: -x[1])

    return pareto_candidates


# -------------------------------------------------
# Complete Example: Pareto Front Structure Generation
# -------------------------------------------------

def example_pareto_front_generation(
    extxyz_path="/data/assets/datasets/dielectric/mp/*.extxyz",
    n_samples=1000,
    min_eps_0=15.0,
    max_energy=0.05,
    output_dir="pareto_structures"
):
    """
    Complete example: Load dataset, find Pareto front compositions,
    generate structures for novel compositions.

    Args:
        extxyz_path: Path to dataset
        n_samples: Number of dataset structures to sample (None = all)
        min_eps_0: Minimum dielectric constant
        max_energy: Maximum energy above hull
        output_dir: Directory to save generated structures
    """
    print("="*80)
    print("PARETO FRONT STRUCTURE GENERATION EXAMPLE")
    print("="*80)

    # 1. Load dataset
    dataset = load_dataset(extxyz_path)
    dataset = deduplicate_by_composition(dataset)

    # Sample if requested
    if n_samples and n_samples < len(dataset):
        print(f"\nSampling {n_samples} structures from dataset...")
        import random
        random.seed(42)
        dataset = random.sample(dataset, n_samples)

    atoms_list = [atoms for atoms, _ in dataset]
    pmg_structures = [pmg_struct for _, pmg_struct in dataset]

    # 2. Load trained models
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    models_dict = {}
    for model_name in ['eps_0', 'energy_above_hull']:
        try:
            model, metadata = load_trained_model(model_name)
            models_dict[model_name] = (model, metadata)
            print(f"✓ Loaded {model_name} model")
        except FileNotFoundError as e:
            print(f"✗ {e}")
            return

    # 3. Predict properties
    print("\n" + "="*80)
    print("PREDICTING PROPERTIES")
    print("="*80)

    predictions = predict_properties(atoms_list, models_dict)

    eps_0_pred = predictions['eps_0']
    energy_pred = predictions['energy_above_hull']

    print(f"\nPrediction statistics:")
    print(f"  eps_0: {eps_0_pred.min():.2f} - {eps_0_pred.max():.2f} (mean: {eps_0_pred.mean():.2f})")
    print(f"  energy_above_hull: {energy_pred.min():.4f} - {energy_pred.max():.4f} (mean: {energy_pred.mean():.4f})")

    # 4. Find Pareto front
    print("\n" + "="*80)
    print("PARETO FRONT ANALYSIS")
    print("="*80)

    pareto_candidates = find_pareto_front(
        atoms_list, eps_0_pred, energy_pred,
        min_eps_0=min_eps_0, max_energy=max_energy
    )

    if not pareto_candidates:
        print("No Pareto front candidates found. Try relaxing the criteria.")
        return

    # 5. Display top candidates
    print("\n" + "="*80)
    print(f"TOP {min(20, len(pareto_candidates))} PARETO FRONT CANDIDATES")
    print("="*80)
    print(f"{'Rank':<6} {'Formula':<20} {'eps_0 (pred)':<15} {'E_hull (pred)':<15}")
    print("-"*80)

    for i, (atoms, eps_0, energy) in enumerate(pareto_candidates[:20]):
        formula = atoms.get_chemical_formula()
        print(f"{i+1:<6} {formula:<20} {eps_0:>12.2f}    {energy:>12.4f}")

    # 6. Generate structures for novel compositions
    print("\n" + "="*80)
    print("GENERATING STRUCTURES FOR NOVEL COMPOSITIONS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # For this example, we'll use the top Pareto candidates as prototypes
    # In practice, you would generate novel compositions and use these as templates

    print(f"\nTop 5 candidates can be used as prototypes for structure generation:")
    for i, (atoms, eps_0, energy) in enumerate(pareto_candidates[:5]):
        formula = atoms.get_chemical_formula()
        output_file = os.path.join(output_dir, f"pareto_candidate_{i+1}_{formula}.extxyz")

        # Save structure
        try:
            from ase.io import write
            write(output_file, atoms)
            print(f"\n{i+1}. {formula}")
            print(f"   Predicted eps_0: {eps_0:.2f}")
            print(f"   Predicted E_hull: {energy:.4f}")
            print(f"   Saved to: {output_file}")
        except Exception as e:
            print(f"   Error saving: {e}")

    # 7. Example: Generate structure for a target composition
    print("\n" + "="*80)
    print("EXAMPLE: STRUCTURE GENERATION FOR TARGET COMPOSITION")
    print("="*80)

    # Pick a target composition (e.g., modify a high-performing one)
    if pareto_candidates:
        template_atoms, _, _ = pareto_candidates[0]
        template_formula = template_atoms.get_chemical_formula()

        print(f"\nUsing template: {template_formula}")
        print(f"Available prototypes: {len(pmg_structures)} structures")

        # Example: Try to generate structure for same composition
        # (In practice, you'd use a novel composition here)
        target_formula = template_formula

        print(f"\nGenerating structure for: {target_formula}")
        new_structure = generate_structure(target_formula, pmg_structures)

        if new_structure:
            output_file = os.path.join(output_dir, f"generated_{target_formula}.extxyz")
            # Convert pymatgen Structure to ASE Atoms for extxyz format
            from ase import Atoms as ASEAtoms
            from ase.io import write
            generated_atoms = ASEAtoms(
                symbols=[str(site.specie.symbol) for site in new_structure],
                positions=new_structure.cart_coords,
                cell=new_structure.lattice.matrix,
                pbc=True
            )
            write(output_file, generated_atoms)
            print(f"✓ Generated structure saved to: {output_file}")
            print(f"  Formula: {new_structure.composition.reduced_formula}")
            print(f"  Lattice: {new_structure.lattice}")
        else:
            print(f"✗ Failed to generate structure for {target_formula}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Pareto front candidates: {len(pareto_candidates)}")

    return pareto_candidates


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate crystal structures for high-dielectric compositions"
    )
    parser.add_argument(
        "--extxyz_path",
        default="/data/assets/datasets/dielectric/mp/*.extxyz",
        help="Path to dataset (glob pattern)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of structures to sample (None = all)"
    )
    parser.add_argument(
        "--min_eps_0",
        type=float,
        default=15.0,
        help="Minimum dielectric constant threshold"
    )
    parser.add_argument(
        "--max_energy",
        type=float,
        default=0.05,
        help="Maximum energy above hull threshold"
    )
    parser.add_argument(
        "--output_dir",
        default="pareto_structures",
        help="Output directory for generated structures"
    )

    args = parser.parse_args()

    # Run the example
    example_pareto_front_generation(
        extxyz_path=args.extxyz_path,
        n_samples=args.n_samples,
        min_eps_0=args.min_eps_0,
        max_energy=args.max_energy,
        output_dir=args.output_dir
    )

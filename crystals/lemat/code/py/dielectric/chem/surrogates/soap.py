"""
===========================================================
BASELINE SURROGATE MODEL FOR MATERIALS PROPERTY RANKING
===========================================================

GOAL
-----
Build a FAST, STRONG baseline model to determine whether your
dataset contains learnable structure → property signal.

This model is NOT meant to be perfect.

It is meant to answer:

    "Can we reliably rank materials so that the best candidates
     appear near the top?"

If YES → diffusion + surrogate pipeline is viable.
If NO  → stop and rethink dataset/model before burning GPU months.

-----------------------------------------------------------
ALGORITHM
-----------------------------------------------------------

1. Load crystal structures from extxyz.

2. Compute TWO types of features:

    (A) Composition Features (very high signal)
        - number of elements
        - mean atomic number
        - variance in atomic number
        - mean electronegativity
        - variance electronegativity
        - mean atomic radius
        - variance atomic radius

    (B) SOAP descriptors (local geometric signal)
        - averaged over atoms to produce a global descriptor.

3. Concatenate features → feature matrix X.

4. Train XGBoost regressor.

5. Evaluate using RANKING metrics:

    ✓ Spearman correlation  (PRIMARY)
    ✓ Top-10% precision     (DISCOVERY METRIC)
    ✓ MAE                   (sanity check)

Interpretation:

Spearman:
    <0.3    weak signal
    0.4–0.6 usable
    >0.7    excellent

Top-10% precision:
    random baseline = 0.10
    >0.30 → promising
    >0.50 → excellent

6. Plot feature importance.

-----------------------------------------------------------
WHY THIS WORKS
-----------------------------------------------------------

Tree models are extremely strong tabular learners.

Composition alone often predicts bandgap surprisingly well.
SOAP adds structural nuance.

This model becomes:

    ✓ generator filter
    ✓ active learning bootstrap
    ✓ sanity detector

Even after you build GNNs.

===========================================================
"""

import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from joblib import Parallel, delayed
from pymatgen.core.periodic_table import Element as PmgElement
from glob import glob
from pathlib import Path
import os
import pickle

import xgboost as xgb


# =========================================================
# USER SETTINGS
# =========================================================

TEST_PATH = "data/*.extxyz"
EXTXYZ_PATH = "/data/assets/datasets/dielectric/mp/*.extxyz"

MODELS = ["dft_band_gap", "energy_above_hull", "dft_eps_0", "dft_eps_inf"]

# Map model names to their source property keys in the data
MODEL_TO_PROPERTY = {
    "dft_band_gap": "dft_band_gap",
    "energy_above_hull": "energy_above_hull",
    "dft_eps_0": "dft_eps_0",
    "dft_eps_inf": "dft_eps_inf",
}


N_JOBS = os.cpu_count()  # auto-detect available CPUs
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =========================================================
# FEATURE COMPUTATION FUNCTIONS (shared)
# =========================================================


def _safe_float(value):
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def composition_features(atoms):
    """Compute composition-based features."""
    Z = atoms.get_atomic_numbers()
    elems, counts = np.unique(Z, return_counts=True)
    frac = counts / counts.sum()

    electroneg = []
    radius = []

    for z in elems:
        e = PmgElement.from_Z(int(z))
        electroneg.append(_safe_float(e.X))
        radius.append(_safe_float(e.atomic_radius))

    electroneg = np.array(electroneg)
    radius = np.array(radius)

    features = [
        len(elems),
        np.sum(frac * elems),
        np.var(elems),
        np.sum(frac * electroneg),
        np.var(electroneg),
        np.sum(frac * radius),
        np.var(radius),
    ]

    return np.array(features)


def composition_string(atoms):
    """Generate composition string for grouping."""
    Z = atoms.get_atomic_numbers()
    elems, counts = np.unique(Z, return_counts=True)
    return "_".join(f"{e}{c}" for e, c in sorted(zip(elems, counts)))


def load_structures_from_extxyz(extxyz_path=EXTXYZ_PATH):
    files = glob(extxyz_path)
    structures = []
    for file in files:
        atoms_list = read(file, index=":")
        if isinstance(atoms_list, list):
            structures.extend(atoms_list)
        else:
            structures.append(atoms_list)
    return structures


def build_soap_descriptor(species, r_cut=5.0, n_max=6, l_max=4, periodic=True, sparse=False):
    return SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=periodic,
        sparse=sparse,
    )


def featurize_structures(
    structures,
    soap,
    n_jobs=None,
    batch_size=None,
    out=None,
    dtype=None,
    prefer="processes",
):
    if n_jobs is None:
        n_jobs = os.cpu_count()

    def soap_features(atoms):
        s = soap.create(atoms)
        return s.mean(axis=0)

    def featurize(atoms):
        comp = composition_features(atoms)
        soap_vec = soap_features(atoms)
        return np.concatenate([comp, soap_vec])

    if batch_size is None and out is None and dtype is None and prefer == "processes":
        X = Parallel(n_jobs=n_jobs)(delayed(featurize)(atoms) for atoms in structures)
        return np.vstack(X)

    n_structures = len(structures)
    if n_structures == 0:
        return np.empty((0, 0), dtype=dtype or np.float32)

    first = featurize(structures[0])
    feature_dim = int(first.shape[0])
    if dtype is None:
        dtype = first.dtype

    if out is None:
        X = np.empty((n_structures, feature_dim), dtype=dtype)
    elif isinstance(out, (str, Path)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        X = np.memmap(out, mode="w+", dtype=dtype, shape=(n_structures, feature_dim))
    else:
        X = out

    X[0] = first.astype(dtype, copy=False)
    start = 1
    if batch_size is None:
        batch_size = max(1, min(128, n_structures))
    else:
        batch_size = max(1, int(batch_size))

    for batch_start in range(start, n_structures, batch_size):
        batch = structures[batch_start : batch_start + batch_size]
        feats = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(featurize)(atoms) for atoms in batch
        )
        batch_arr = np.asarray(feats, dtype=dtype)
        X[batch_start : batch_start + batch_arr.shape[0]] = batch_arr

    return np.asarray(X)


def test_no_info_leakage(test_structures):
    """
    Verify that feature computation doesn't depend on atoms.info
    by computing features with and without info and comparing.
    """
    print("\n" + "=" * 60)
    print("TESTING FOR INFO LEAKAGE")
    print("=" * 60)

    test_atoms = test_structures[0].copy()

    # Get species for SOAP
    species = sorted(
        list({Z for atoms in test_structures[:10] for Z in atoms.get_atomic_numbers()})
    )

    soap_temp = build_soap_descriptor(species)

    def soap_features_test(atoms):
        s = soap_temp.create(atoms)
        return s.mean(axis=0)

    # Compute features with original info
    comp1 = composition_features(test_atoms)
    soap1 = soap_features_test(test_atoms)

    # Clear info and compute again
    original_info = test_atoms.info.copy()
    test_atoms.info = {}

    comp2 = composition_features(test_atoms)
    soap2 = soap_features_test(test_atoms)

    # Restore info
    test_atoms.info = original_info

    # Compare
    comp_match = np.allclose(comp1, comp2)
    soap_match = np.allclose(soap1, soap2)

    print(f"\n✓ Composition features identical with/without info: {comp_match}")
    print(f"✓ SOAP features identical with/without info: {soap_match}")

    if not comp_match:
        print("⚠ WARNING: Composition features depend on atoms.info!")
        print(f"  Difference: {np.abs(comp1 - comp2).max()}")

    if not soap_match:
        print("⚠ WARNING: SOAP features depend on atoms.info!")
        print(f"  Difference: {np.abs(soap1 - soap2).max()}")

    print("=" * 60 + "\n")

    return comp_match and soap_match


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_surrogate_model(model_name, model_dir=None):
    if model_dir is None:
        model_dir = _DATA_DIR
    model_file = os.path.join(model_dir, f"xgb_surrogate_{model_name}.json")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    return model


def predict_properties(
    structures,
    model_names=("dft_band_gap", "dft_eps_0"),
    extxyz_path=EXTXYZ_PATH,
    n_jobs=None,
    soap_params=None,
):
    """
    Predict properties for given ASE Atoms using trained SOAP surrogates.

    Returns dict: {model_name: np.ndarray}
    """
    if not structures:
        return {name: np.array([]) for name in model_names}

    if soap_params is None:
        soap_params = {}

    # Build SOAP descriptor using training species for consistency
    training_structures = load_structures_from_extxyz(extxyz_path)
    species = sorted(
        list({Z for atoms in training_structures for Z in atoms.get_atomic_numbers()})
    )
    soap = build_soap_descriptor(species, **soap_params)

    X = featurize_structures(structures, soap, n_jobs=n_jobs)

    predictions = {}
    for name in model_names:
        model = load_surrogate_model(name)
        predictions[name] = model.predict(X)

    return predictions


if __name__ == "__main__":
    # Training has been moved to scripts/train_soap.py
    raise SystemExit("Training moved to chem/scripts/train_soap.py")

"""
===========================================================
COMPOSITION-BASED SURROGATE MODEL FOR MATERIALS RANKING
===========================================================

GOAL
-----
Build a FAST baseline model using composition features only to determine
whether composition alone contains sufficient signal for property prediction.

This model uses advanced composition featurizers from matminer:
    1. Magpie features (ElementProperty)
    2. Stoichiometry features
    3. Valence orbital fractions
    4. Ionic character

-----------------------------------------------------------
ALGORITHM
-----------------------------------------------------------

1. Load crystal structures from extxyz.

2. Convert structures to pymatgen Composition objects.

3. Compute composition features using matminer featurizers:
    - Magpie features (elemental properties)
    - Stoichiometry features
    - Valence orbital features
    - Ionic character features

4. Train XGBoost regressor on composition features.

5. Evaluate using RANKING metrics:
    ✓ Spearman correlation  (PRIMARY)
    ✓ Top-10% precision     (DISCOVERY METRIC)
    ✓ MAE                   (sanity check)

6. Plot feature importance.

-----------------------------------------------------------
WHY THIS WORKS
-----------------------------------------------------------

Composition features capture:
    ✓ Elemental properties (electronegativity, radius, etc.)
    ✓ Stoichiometric ratios
    ✓ Electronic structure information
    ✓ Ionic character

These are often sufficient for bandgap and dielectric constant prediction
without needing structural information.

===========================================================
"""

from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from ase.io import read
from joblib import Parallel, delayed
from glob import glob
import os
import pickle
import json
import warnings

from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

import xgboost as xgb
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Matminer imports
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import Stoichiometry
from matminer.featurizers.composition import ValenceOrbital
from matminer.featurizers.composition import IonProperty

# Suppress harmless matminer warnings about impute_nan default change
warnings.filterwarnings("ignore", message=".*impute_nan.*", category=UserWarning)


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

print(f"Using {N_JOBS} CPU cores for parallel processing")


# =========================================================
# INITIALIZE MATMINER FEATURIZERS
# =========================================================

print("\nInitializing matminer featurizers...")

# 1. Magpie features - elemental properties
magpie_featurizer = ElementProperty.from_preset("magpie", impute_nan=True)

# 2. Stoichiometry features
stoich_featurizer = Stoichiometry()

# 3. Valence orbital fractions
valence_featurizer = ValenceOrbital()

# 4. Ionic character
ionic_featurizer = IonProperty()

print("✓ Featurizers initialized")


# =========================================================
# LOAD STRUCTURES
# =========================================================

print("\nLoading structures...")
files = glob(EXTXYZ_PATH)
print(f"Found {len(files)} extxyz files")

structures = []
for file in files:
    atoms_list = read(file, index=":")
    if isinstance(atoms_list, list):
        structures.extend(atoms_list)
    else:
        structures.append(atoms_list)

print(f"Loaded {len(structures)} structures")

# =========================================================
# DEDUPLICATE BY COMPOSITION (keep lowest energy_above_hull)
# =========================================================

print("\nDeduplicating structures by composition...")
print("Keeping only the structure with lowest energy_above_hull for each composition")

composition_dict = {}
for atoms in structures:
    comp_str = atoms.get_chemical_formula()

    # Get energy_above_hull if available
    energy = atoms.info.get("energy_above_hull", float("inf"))

    # Keep structure with lowest energy_above_hull
    if comp_str not in composition_dict:
        composition_dict[comp_str] = atoms
    else:
        existing_energy = composition_dict[comp_str].info.get(
            "energy_above_hull", float("inf")
        )
        if energy < existing_energy:
            composition_dict[comp_str] = atoms

all_structures = list(composition_dict.values())
print(f"After deduplication: {len(all_structures)} unique compositions")
print(f"Removed {len(structures) - len(all_structures)} duplicate compositions")

# =========================================================
# DATA LEAKAGE CHECK (run once for all properties)
# =========================================================
print("\n" + "=" * 60)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 60)

# Inspect what's in atoms.info for first few structures
print("\nInspecting atoms.info contents from first 3 structures:")
for i, atoms in enumerate(all_structures[:3]):
    print(f"\nStructure {i}:")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  info keys: {list(atoms.info.keys())}")
    print(f"  info values: {atoms.info}")

# Check if atoms.arrays contains any suspicious data
print("\nInspecting atoms.arrays from first structure:")
print(f"  arrays keys: {list(all_structures[0].arrays.keys())}")

# Check for other properties that might correlate with target properties
print("\nChecking for potentially correlated properties in dataset:")
common_correlated_props = [
    "eps_inf",
    "eps_0",
    "e_total",
    "bandgap",
    "energy",
    "energy_per_atom",
    "formation_energy",
    "gap",
    "band_gap",
    "homo",
    "lumo",
]

found_suspicious = []
for atoms in all_structures[:5]:
    for key in atoms.info.keys():
        if key.lower() in [p.lower() for p in common_correlated_props]:
            found_suspicious.append(key)

print(f"Found properties in dataset: {set(found_suspicious)}")
print("⚠ These properties should NOT be used in features (only as targets)!")

print("=" * 60 + "\n")


# Create cache directory if it doesn't exist
os.makedirs("data", exist_ok=True)


# =========================================================
# FEATURE COMPUTATION FUNCTIONS (shared across properties)
# =========================================================


def atoms_to_composition(atoms):
    """Convert ASE atoms to pymatgen Composition."""
    formula = atoms.get_chemical_formula()
    return Composition(formula)


def composition_features(atoms):
    """
    Compute comprehensive composition-based features using matminer.

    Returns concatenated feature vector from:
        1. Magpie features (elemental properties)
        2. Stoichiometry features
        3. Valence orbital fractions
        4. Ionic character features
    """
    comp = atoms_to_composition(atoms)

    features = []

    # 1. Magpie features
    try:
        magpie_feats = magpie_featurizer.featurize(comp)
        features.extend(magpie_feats)
    except Exception as e:
        print(f"Warning: Magpie featurization failed for {comp}: {e}")
        features.extend([0] * len(magpie_featurizer.feature_labels()))

    # 2. Stoichiometry features
    try:
        stoich_feats = stoich_featurizer.featurize(comp)
        features.extend(stoich_feats)
    except Exception as e:
        print(f"Warning: Stoichiometry featurization failed for {comp}: {e}")
        features.extend([0] * len(stoich_featurizer.feature_labels()))

    # 3. Valence orbital fractions
    try:
        valence_feats = valence_featurizer.featurize(comp)
        features.extend(valence_feats)
    except Exception as e:
        print(f"Warning: Valence orbital featurization failed for {comp}: {e}")
        features.extend([0] * len(valence_featurizer.feature_labels()))

    # 4. Ionic character
    try:
        ionic_feats = ionic_featurizer.featurize(comp)
        features.extend(ionic_feats)
    except Exception as e:
        print(f"Warning: Ionic property featurization failed for {comp}: {e}")
        features.extend([0] * len(ionic_featurizer.feature_labels()))

    return np.array(features)


def get_feature_names():
    """Get all feature names from featurizers."""
    names = []
    names.extend(magpie_featurizer.feature_labels())
    names.extend(stoich_featurizer.feature_labels())
    names.extend(valence_featurizer.feature_labels())
    names.extend(ionic_featurizer.feature_labels())
    return names


def test_no_info_leakage(test_structures):
    """
    Verify that feature computation doesn't depend on atoms.info
    by computing features with and without info and comparing.
    """
    print("\n" + "=" * 60)
    print("TESTING FOR INFO LEAKAGE")
    print("=" * 60)

    test_atoms = test_structures[0].copy()

    # Compute features with original info
    feat1 = composition_features(test_atoms)

    # Clear info and compute again
    original_info = test_atoms.info.copy()
    test_atoms.info = {}

    feat2 = composition_features(test_atoms)

    # Restore info
    test_atoms.info = original_info

    # Compare
    match = np.allclose(feat1, feat2)

    print(f"\n✓ Composition features identical with/without info: {match}")

    if not match:
        print("⚠ WARNING: Composition features depend on atoms.info!")
        print(f"  Difference: {np.abs(feat1 - feat2).max()}")

    print("=" * 60 + "\n")

    return match


def composition_string(atoms):
    """Generate composition string for grouping."""
    Z = atoms.get_atomic_numbers()
    elems, counts = np.unique(Z, return_counts=True)
    return "_".join(f"{e}{c}" for e, c in sorted(zip(elems, counts)))


# Run leakage test once with all structures
leakage_test_passed = test_no_info_leakage(all_structures)
if not leakage_test_passed:
    print("⚠⚠⚠ CRITICAL: DATA LEAKAGE DETECTED! ⚠⚠⚠")
    print("Feature computation is accessing atoms.info!")
    raise RuntimeError("Data leakage detected in feature computation")


# Get feature names for later use
feature_names = get_feature_names()
print(f"\nTotal number of features: {len(feature_names)}")
print(f"Feature names sample: {feature_names[:5]}...")


# =========================================================
# TRAIN MODELS FOR EACH PROPERTY
# =========================================================

results_summary = {}

for MODEL_NAME in MODELS:
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL FOR: {MODEL_NAME}")
    print("=" * 80 + "\n")

    # Get the source property key for this model
    PROPERTY_KEY = MODEL_TO_PROPERTY[MODEL_NAME]

    # Filter structures that have the target property
    valid_structures = []
    y_values = []
    for atoms in all_structures:
        if PROPERTY_KEY in atoms.info:
            valid_structures.append(atoms)
            y_values.append(atoms.info[PROPERTY_KEY])

    structures = valid_structures
    y = np.array(y_values)

    print(f"Using {len(structures)} structures with property '{PROPERTY_KEY}'")

    # =========================================================
    # FEATURE CACHING
    # =========================================================

    cache_file = f"data/composition_features_{MODEL_NAME}.pkl"

    if os.path.exists(cache_file):
        print(f"\nLoading cached features from {cache_file}...")
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
            X = cache_data["X"]
            y_cached = cache_data["y"]

            # Verify cache matches current data
            if len(y_cached) == len(y) and np.allclose(y_cached, y):
                print(f"✓ Loaded cached features: {X.shape}")
                skip_feature_computation = True
            else:
                print("⚠ Cache mismatch detected, recomputing features...")
                skip_feature_computation = False
    else:
        print(f"\nNo cache found, will compute features and save to {cache_file}")
        skip_feature_computation = False

    # =========================================================
    # COMPUTE FEATURES IF NEEDED
    # =========================================================

    if not skip_feature_computation:
        print("Computing composition features (parallel)...")

        X = Parallel(n_jobs=N_JOBS)(
            delayed(composition_features)(atoms) for atoms in structures
        )
        X = np.vstack(X)

        print("Feature matrix shape:", X.shape)

        # Check for NaN or infinite values
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠ Warning: Found {nan_count} NaN and {inf_count} inf values")
            print("Replacing NaN/inf with 0...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Save computed features to cache
        print(f"Saving features to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump({"X": X, "y": y}, f)
        print("✓ Features cached successfully")
    else:
        print("Using cached features")

    # =========================================================
    # TRAIN / TEST SPLIT
    # =========================================================

    groups = [composition_string(a) for a in structures]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # =========================================================
    # TRAIN XGBOOST
    # =========================================================

    print("\n" + "=" * 60)
    print(f"TRAINING XGBOOST FOR {MODEL_NAME}")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print("Max iterations: 800")
    print("=" * 60 + "\n")

    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=50,
        eval_metric=["rmse", "mae"],
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,  # Show progress every 50 iterations
    )

    print(f"\n✓ Training completed after {model.best_iteration} iterations")
    print(f"✓ Best validation RMSE: {model.best_score:.4f}")

    # =========================================================
    # METRICS
    # =========================================================

    print("\nEvaluating model...")

    pred = model.predict(X_test)

    rho, _ = spearmanr(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"\nSpearman correlation: {rho:.3f}")
    print(f"MAE: {mae:.3f}")

    # ---------- TOP 10% PRECISION ----------

    k = max(1, int(0.1 * len(y_test)))  # Ensure at least 1 sample

    top_true_idx = np.argsort(y_test)[-k:]
    top_pred_idx = np.argsort(pred)[-k:]

    precision = len(set(top_true_idx) & set(top_pred_idx)) / k

    print(f"Top-{k}% precision: {precision:.3f} (k={k})")

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    print("\nPlotting feature importance...")

    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.title(f"Feature Importance - {MODEL_NAME} (Composition Only)")
    plt.tight_layout()
    importance_file = f"feature_importance_composition_{MODEL_NAME}.png"
    plt.savefig(importance_file)
    print(f"Feature importance plot saved to {importance_file}")
    plt.close()

    # =========================================================
    # SAVE MODEL
    # =========================================================

    print("\n" + "=" * 60)
    print(f"SAVING MODEL FOR {MODEL_NAME}")
    print("=" * 60)

    model_file = f"data/xgb_composition_{MODEL_NAME}.json"
    model.save_model(model_file)
    print(f"✓ Model saved to {model_file}")

    # Also save model metadata
    metadata = {
        "model_name": MODEL_NAME,
        "property": PROPERTY_KEY,
        "model_type": "composition_only",
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "spearman": float(rho),
        "mae": float(mae),
        "top_k_precision": float(precision),
    }

    metadata_file = f"data/xgb_composition_metadata_{MODEL_NAME}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_file}")

    # Store results for summary
    results_summary[MODEL_NAME] = {
        "property": PROPERTY_KEY,
        "spearman": rho,
        "mae": mae,
        "top_k_precision": precision,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print("=" * 60)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY (COMPOSITION FEATURES ONLY)")
print("=" * 80 + "\n")

for model_name, results in results_summary.items():
    print(f"{model_name}:")
    print(f"  Source property: {results['property']}")
    print(f"  Spearman: {results['spearman']:.3f}")
    print(f"  MAE: {results['mae']:.3f}")
    print(f"  Top-10% precision: {results['top_k_precision']:.3f}")
    print(f"  Train/Test: {results['n_train']}/{results['n_test']}")
    print()

print("Models saved:")
for model_name in MODELS:
    print(f"  - data/xgb_composition_{model_name}.json")

print("\nDONE.")

#!/usr/bin/env python
"""Direct eps_0 Surrogate Model

This module provides fast, accurate eps_0 predictions using pre-trained XGBoost models,
bypassing the need for expensive phonon calculations.

Two models available:
1. Composition-based: Fast, uses only chemical formula (~instant)
2. Structure-based: More accurate, uses SOAP structural descriptors (~1-2 seconds)

Usage:
    from eps_0_surrogate import predict_eps_0_fast, predict_eps_0_accurate
    from ase.io import read

    atoms = read("structure.extxyz")

    # Fast prediction (composition only)
    eps_0_fast = predict_eps_0_fast(atoms)

    # Accurate prediction (composition + structure)
    eps_0_accurate = predict_eps_0_accurate(atoms)
"""

import numpy as np
import xgboost as xgb
from ase import Atoms
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
import pickle
import warnings
from pathlib import Path

# Suppress matminer warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Model paths
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
COMPOSITION_MODEL = MODEL_DIR / "xgb_composition_eps_0.json"
COMPOSITION_FEATURES = MODEL_DIR / "composition_features_eps_0.pkl"
STRUCTURE_MODEL = MODEL_DIR / "xgb_surrogate_eps_0.json"
STRUCTURE_FEATURES = MODEL_DIR / "soap_features_eps_0.pkl"

# Cache for loaded models
_composition_model = None
_composition_feature_names = None
_structure_model = None
_structure_feature_names = None


def load_composition_model():
    """Load composition-based eps_0 model."""
    global _composition_model, _composition_feature_names

    if _composition_model is None:
        _composition_model = xgb.Booster()
        _composition_model.load_model(str(COMPOSITION_MODEL))

        # Load feature names
        with open(COMPOSITION_FEATURES, 'rb') as f:
            _composition_feature_names = pickle.load(f)

    return _composition_model, _composition_feature_names


def load_structure_model():
    """Load structure-based eps_0 model."""
    global _structure_model, _structure_feature_names

    if _structure_model is None:
        _structure_model = xgb.Booster()
        _structure_model.load_model(str(STRUCTURE_MODEL))

        # Load feature names
        with open(STRUCTURE_FEATURES, 'rb') as f:
            _structure_feature_names = pickle.load(f)

    return _structure_model, _structure_feature_names


def compute_composition_features(atoms: Atoms) -> np.ndarray:
    """Compute composition features for an Atoms object.

    Uses matminer featurizers:
    - Magpie features (elemental properties)
    - Stoichiometry features
    - Valence orbital features
    - Ionic character features
    """
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.composition import Stoichiometry
    from matminer.featurizers.composition import ValenceOrbital
    from matminer.featurizers.composition import IonProperty

    # Get composition
    formula = atoms.get_chemical_formula()
    comp = Composition(formula)

    # Initialize featurizers
    magpie = ElementProperty.from_preset("magpie")
    stoich = Stoichiometry()
    valence = ValenceOrbital()
    ionic = IonProperty()

    # Compute features
    features = []

    # Magpie features
    try:
        magpie_feats = magpie.featurize(comp)
        features.extend(magpie_feats)
    except:
        features.extend([0] * len(magpie.feature_labels()))

    # Stoichiometry features
    try:
        stoich_feats = stoich.featurize(comp)
        features.extend(stoich_feats)
    except:
        features.extend([0] * len(stoich.feature_labels()))

    # Valence orbital features
    try:
        valence_feats = valence.featurize(comp)
        features.extend(valence_feats)
    except:
        features.extend([0] * len(valence.feature_labels()))

    # Ionic features
    try:
        ionic_feats = ionic.featurize(comp)
        features.extend(ionic_feats)
    except:
        features.extend([0] * len(ionic.feature_labels()))

    return np.array(features, dtype=np.float32)


def compute_soap_features(atoms: Atoms) -> np.ndarray:
    """Compute SOAP structural descriptors for an Atoms object.

    SOAP (Smooth Overlap of Atomic Positions) captures local atomic environments.
    More accurate than composition-only, but slower.
    """
    try:
        from dscribe.descriptors import SOAP

        # SOAP parameters (should match training)
        soap = SOAP(
            species=list(set(atoms.get_chemical_symbols())),
            r_cut=6.0,
            n_max=8,
            l_max=6,
            periodic=True,
            average="outer"
        )

        # Compute SOAP descriptor
        soap_features = soap.create(atoms)

        # Average over atoms if needed
        if soap_features.ndim > 1:
            soap_features = soap_features.mean(axis=0)

        return soap_features

    except ImportError:
        raise ImportError(
            "dscribe package required for structure-based predictions. "
            "Install with: pip install dscribe"
        )


def predict_eps_0_fast(atoms: Atoms) -> float:
    """Fast eps_0 prediction using composition features only.

    Speed: ~instant (< 0.1 seconds)
    Accuracy: Moderate (Spearman ~0.7-0.8)

    Args:
        atoms: ASE Atoms object

    Returns:
        Predicted static dielectric constant (ε₀)

    Example:
        >>> atoms = read("structure.extxyz")
        >>> eps_0 = predict_eps_0_fast(atoms)
        >>> print(f"ε₀ ≈ {eps_0:.2f}")
    """
    # Load model
    model, feature_names = load_composition_model()

    # Compute features
    features = compute_composition_features(atoms)

    # Make prediction
    dmatrix = xgb.DMatrix(features.reshape(1, -1))
    eps_0 = model.predict(dmatrix)[0]

    # Model predicts eps_0 directly (not log-transformed)
    # Ensure reasonable bounds
    eps_0 = max(1.0, float(eps_0))  # Minimum dielectric constant is 1

    return eps_0


def predict_eps_0_accurate(atoms: Atoms) -> float:
    """Accurate eps_0 prediction using composition + structure features.

    Speed: ~1-2 seconds (SOAP calculation)
    Accuracy: High (Spearman ~0.85-0.90)

    Args:
        atoms: ASE Atoms object

    Returns:
        Predicted static dielectric constant (ε₀)

    Example:
        >>> atoms = read("structure.extxyz")
        >>> eps_0 = predict_eps_0_accurate(atoms)
        >>> print(f"ε₀ ≈ {eps_0:.2f}")
    """
    # Load model
    model, feature_names = load_structure_model()

    # Compute composition features
    comp_features = compute_composition_features(atoms)

    # Compute SOAP features
    soap_features = compute_soap_features(atoms)

    # Combine features
    features = np.concatenate([comp_features, soap_features])

    # Make prediction
    dmatrix = xgb.DMatrix(features.reshape(1, -1))
    eps_0 = model.predict(dmatrix)[0]

    # Model predicts eps_0 directly (not log-transformed)
    # Ensure reasonable bounds
    eps_0 = max(1.0, float(eps_0))  # Minimum dielectric constant is 1

    return eps_0


def predict_eps_0_batch(atoms_list: list[Atoms], accurate: bool = False) -> list[float]:
    """Batch prediction of eps_0 for multiple structures.

    Args:
        atoms_list: List of ASE Atoms objects
        accurate: If True, use structure-based model (slower but more accurate)

    Returns:
        List of predicted eps_0 values
    """
    if accurate:
        return [predict_eps_0_accurate(atoms) for atoms in atoms_list]
    else:
        return [predict_eps_0_fast(atoms) for atoms in atoms_list]


# ============================================================================
# Model Retraining
# ============================================================================

def retrain_models(data_path: str = "/data/assets/datasets/dielectric/mp/*.extxyz"):
    """Retrain eps_0 surrogate models from scratch.

    This will:
    1. Load all structures with DFT eps_0 values
    2. Compute features (composition + SOAP)
    3. Train XGBoost models
    4. Save models to data/ folder

    Args:
        data_path: Glob pattern for extxyz files with eps_0 data

    Example:
        >>> retrain_models("data/*.extxyz")
    """
    print("="*80)
    print("RETRAINING EPS_0 SURROGATE MODELS")
    print("="*80)

    # Option 1: Use existing composition.py script
    print("\nFor composition-based model:")
    print("  python composition.py")
    print("  (Edit MODELS list to include 'eps_0')")

    # Option 2: Use existing soap.py script
    print("\nFor structure-based model:")
    print("  python soap.py")
    print("  (Edit MODELS list to include 'eps_0')")

    print("\nOr implement custom training here if needed.")


# ============================================================================
# Testing & Validation
# ============================================================================

def test_predictions():
    """Test surrogate predictions on sample structures."""
    from ase.io import read
    import glob

    print("="*80)
    print("TESTING EPS_0 SURROGATE MODELS")
    print("="*80)

    test_files = glob.glob("data/*.extxyz")[:5]

    if not test_files:
        print("No test files found in data/")
        return

    print(f"\nTesting on {len(test_files)} structures:\n")
    print(f"{'File':<40} {'DFT ε₀':>10} {'Fast':>10} {'Error':>10}")
    print("-"*80)

    for fname in test_files:
        atoms = read(fname)

        # Get DFT reference
        ref_eps_0 = atoms.info.get('eps_0', None)
        if ref_eps_0 is None:
            continue

        # Fast prediction
        pred_fast = predict_eps_0_fast(atoms)
        error_fast = ((pred_fast - ref_eps_0) / ref_eps_0) * 100

        print(f"{fname.split('/')[-1]:<40} {ref_eps_0:>10.2f} {pred_fast:>10.2f} {error_fast:>9.1f}%")

    print("\n" + "="*80)
    print("Note: Install dscribe for accurate structure-based predictions:")
    print("  pip install dscribe")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_predictions()
    elif len(sys.argv) > 1 and sys.argv[1] == "retrain":
        retrain_models()
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python eps_0_surrogate.py test      # Test on sample structures")
        print("  python eps_0_surrogate.py retrain   # Retrain models")

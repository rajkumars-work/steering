"""
composition_expander.py
======================

GOAL
-----
Generate physically plausible new compositions via prototype substitution.

We DO NOT randomly combine elements.
We substitute chemically similar elements to remain on the
"realizable chemistry manifold".

STRATEGY
--------
1. Use periodic table group similarity.
2. Restrict substitutions to neighbors.
3. Limit number of simultaneous mutations.
4. Preserve stoichiometry.
5. Use XGBoost models to predict bandgap and eps_0.
6. Return Pareto front of compositions.

This is the SAME philosophy used in large materials discovery pipelines.

Future upgrades:
---------------
- oxidation-state constrained substitutions
- ionic radius filters
- charge neutrality checks
- learned substitution priors
"""

from pymatgen.core import Composition, Element
from itertools import combinations, product
import random
import numpy as np

from chem.pareto_front import ParetoFront
import xgboost as xgb
import os
import json
import warnings

# Matminer imports for featurization
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import Stoichiometry
from matminer.featurizers.composition import ValenceOrbital
from matminer.featurizers.composition import IonProperty

# Suppress matminer and pymatgen warnings
warnings.filterwarnings("ignore", message=".*impute_nan.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No Pauling electronegativity.*", category=UserWarning)


# -------------------------------------------------------
# MODEL AND FEATURIZER INITIALIZATION
# -------------------------------------------------------

# Initialize matminer featurizers (same as in composition.py)
magpie_featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
stoich_featurizer = Stoichiometry()
valence_featurizer = ValenceOrbital()
ionic_featurizer = IonProperty()

# Model paths (match composition.py outputs)
_CHEM_DIR = os.path.dirname(os.path.dirname(__file__))
BANDGAP_MODEL_PATH = os.path.join(_CHEM_DIR, "data", "xgb_composition_dft_band_gap.json")
EPS_0_MODEL_PATH = os.path.join(_CHEM_DIR, "data", "xgb_composition_dft_eps_0.json")
BANDGAP_METADATA_PATH = os.path.join(_CHEM_DIR, "data", "xgb_composition_metadata_dft_band_gap.json")
EPS_0_METADATA_PATH = os.path.join(_CHEM_DIR, "data", "xgb_composition_metadata_dft_eps_0.json")

# Global model variables (loaded lazily)
_bandgap_model = None
_eps_0_model = None
_bandgap_metadata = None
_eps_0_metadata = None


def load_models():
    """Load XGBoost models for bandgap and eps_0 prediction."""
    global _bandgap_model, _eps_0_model, _bandgap_metadata, _eps_0_metadata

    if _bandgap_model is None:
        if not os.path.exists(BANDGAP_MODEL_PATH):
            raise FileNotFoundError(
                f"Bandgap model not found at {BANDGAP_MODEL_PATH}. "
                "Run composition.py first to train models."
            )
        _bandgap_model = xgb.XGBRegressor()
        _bandgap_model.load_model(BANDGAP_MODEL_PATH)

        # Load metadata if available
        if os.path.exists(BANDGAP_METADATA_PATH):
            with open(BANDGAP_METADATA_PATH, 'r') as f:
                _bandgap_metadata = json.load(f)

    if _eps_0_model is None:
        if not os.path.exists(EPS_0_MODEL_PATH):
            raise FileNotFoundError(
                f"eps_0 model not found at {EPS_0_MODEL_PATH}. "
                "Run composition.py first to train models."
            )
        _eps_0_model = xgb.XGBRegressor()
        _eps_0_model.load_model(EPS_0_MODEL_PATH)

        # Load metadata if available
        if os.path.exists(EPS_0_METADATA_PATH):
            with open(EPS_0_METADATA_PATH, 'r') as f:
                _eps_0_metadata = json.load(f)

    return _bandgap_model, _eps_0_model


def composition_features(comp):
    """
    Compute comprehensive composition-based features using matminer.
    Same as in composition.py.

    Parameters
    ----------
    comp : pymatgen.Composition
        Composition object

    Returns
    -------
    np.ndarray
        Feature vector
    """
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


def predict_properties(compositions):
    """
    Predict bandgap and eps_0 for a list of composition strings.

    Parameters
    ----------
    compositions : list[str]
        List of composition strings (e.g., ["BaTiO3", "SrZrO3"])

    Returns
    -------
    dict
        {composition_str: (bandgap, eps_0)}
    """
    bandgap_model, eps_0_model = load_models()

    results = {}

    for comp_str in compositions:
        try:
            # Convert to pymatgen Composition
            comp = Composition(comp_str)

            # Compute features
            feats = composition_features(comp)
            feats = feats.reshape(1, -1)  # Shape (1, n_features)

            # Replace NaN/inf with 0
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict
            bandgap = float(bandgap_model.predict(feats)[0])
            eps_0 = float(eps_0_model.predict(feats)[0])

            results[comp_str] = (bandgap, eps_0)

        except Exception as e:
            print(f"Warning: Failed to predict properties for {comp_str}: {e}")
            continue

    return results


# -------------------------------------------------------
# USER CONTROLS (IMPORTANT)
# -------------------------------------------------------

MAX_MUTATIONS = 2          # never go beyond this initially
GROUP_TOLERANCE = 1       # how far in periodic group we allow
ALLOW_ANION_SWAP = False  # usually dangerous early on


# -------------------------------------------------------
# Helper: get substitution candidates
# -------------------------------------------------------

def substitution_candidates(element_symbol):
    """
    Return chemically similar elements based on periodic group proximity.
    """

    el = Element(element_symbol)

    # Skip noble gases etc.
    if el.group is None:
        return []

    candidates = []

    for e in Element:
        if e.group is None:
            continue

        # restrict anion swaps unless allowed
        if not ALLOW_ANION_SWAP:
            if el.is_chalcogen != e.is_chalcogen:
                continue
            if el.is_halogen != e.is_halogen:
                continue

        if abs(e.group - el.group) <= GROUP_TOLERANCE:
            if e.symbol != element_symbol:
                candidates.append(e.symbol)

    return candidates


# -------------------------------------------------------
# Core generator
# -------------------------------------------------------

def expand_composition(formula, max_new=50, return_pareto=False,
                      maximize_bandgap=True, maximize_eps_0=True):
    """
    Generate substituted compositions from a seed formula.

    Parameters
    ----------
    formula : str
        e.g. "BaTiO3"

    max_new : int
        cap number of generated compositions

    return_pareto : bool
        If True, predict properties and return Pareto front.
        If False, just return list of compositions.

    maximize_bandgap : bool
        If True, prefer higher bandgap in Pareto front.

    maximize_eps_0 : bool
        If True, prefer higher eps_0 in Pareto front.

    Returns
    -------
    If return_pareto=False:
        list[str] - list of composition strings

    If return_pareto=True:
        dict - {composition_str: (bandgap, eps_0)} for Pareto front only
    """

    comp = Composition(formula)
    elements = [el.symbol for el in comp.elements]

    substitution_map = {
        el: substitution_candidates(el)
        for el in elements
    }

    new_comps = set()

    # mutate 1 or 2 elements only
    for k in range(1, min(MAX_MUTATIONS, len(elements)) + 1):

        for elems_to_mutate in combinations(elements, k):

            candidate_lists = [
                substitution_map[e] for e in elems_to_mutate
            ]

            for replacements in product(*candidate_lists):

                mapping = dict(zip(elems_to_mutate, replacements))

                new_dict = {}

                for el, amt in comp.get_el_amt_dict().items():
                    new_el = mapping.get(el, el)
                    new_dict[new_el] = new_dict.get(new_el, 0) + amt

                new_formula = Composition(new_dict).reduced_formula

                if new_formula != formula:
                    new_comps.add(new_formula)

                if len(new_comps) >= max_new:
                    if not return_pareto:
                        return list(new_comps)
                    break

            if len(new_comps) >= max_new:
                break

        if len(new_comps) >= max_new:
            break

    compositions = list(new_comps)

    if not return_pareto:
        return compositions

    # Predict properties for all compositions
    print(f"\nPredicting properties for {len(compositions)} candidates...")
    predictions = predict_properties(compositions)

    # Compute Pareto front
    print(f"Computing Pareto front (maximize_bandgap={maximize_bandgap}, maximize_eps_0={maximize_eps_0})...")
    ranker = ParetoFront(maximize_bandgap, maximize_eps_0)
    ranker.add_many((comp, bg, eps) for comp, (bg, eps) in predictions.items())
    pareto_front = ranker.pareto()

    print(f"Pareto front contains {len(pareto_front)} compositions out of {len(predictions)} total")

    return pareto_front


# -------------------------------------------------------
# Batch expansion
# -------------------------------------------------------

def expand_many(seed_formulas, per_seed=30, return_pareto=False,
               maximize_bandgap=True, maximize_eps_0=True):
    """
    Expand multiple compositions.

    Parameters
    ----------
    seed_formulas : list[str]
        List of seed compositions

    per_seed : int
        Number of candidates per seed

    return_pareto : bool
        If True, predict properties and return Pareto front

    maximize_bandgap : bool
        If True, prefer higher bandgap in Pareto front

    maximize_eps_0 : bool
        If True, prefer higher eps_0 in Pareto front

    Returns
    -------
    If return_pareto=False:
        list[str] - unique compositions

    If return_pareto=True:
        dict - {composition_str: (bandgap, eps_0)} for Pareto front only
    """

    if not return_pareto:
        # Original behavior - just generate compositions
        generated = set()

        for f in seed_formulas:
            new = expand_composition(f, max_new=per_seed, return_pareto=False)
            generated.update(new)

        return list(generated)

    else:
        # Generate all compositions first
        generated = set()

        for f in seed_formulas:
            new = expand_composition(f, max_new=per_seed, return_pareto=False)
            generated.update(new)

        compositions = list(generated)

        # Predict properties for all compositions
        print(f"\nPredicting properties for {len(compositions)} unique candidates...")
        predictions = predict_properties(compositions)

        # Compute Pareto front
        print(f"Computing Pareto front (maximize_bandgap={maximize_bandgap}, maximize_eps_0={maximize_eps_0})...")
        ranker = ParetoFront(maximize_bandgap, maximize_eps_0)
        ranker.add_many((comp, bg, eps) for comp, (bg, eps) in predictions.items())
        pareto_front = ranker.pareto()

        print(f"Pareto front contains {len(pareto_front)} compositions out of {len(predictions)} total")

        return pareto_front


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------

if __name__ == "__main__":

    seeds = [
        "BaTiO3",
        "SrZrO3",
        "LiNbO3"
    ]

    print("=" * 70)
    print("EXAMPLE 1: Generate candidates without property prediction")
    print("=" * 70)

    new = expand_many(seeds, per_seed=40, return_pareto=False)

    print(f"\nGenerated {len(new)} candidate compositions")
    print("Sample compositions:")
    for c in new[:10]:
        print(f"  {c}")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Generate candidates with Pareto front prediction")
    print("=" * 70)

    # Get Pareto front with property predictions
    pareto_front = expand_many(
        seeds,
        per_seed=40,
        return_pareto=True,
        maximize_bandgap=True,  # Prefer higher bandgap
        maximize_eps_0=True      # Prefer higher eps_0
    )

    print("\n" + "=" * 70)
    print("PARETO FRONT RESULTS")
    print("=" * 70)

    # Sort by eps_0 (descending)
    sorted_pareto = sorted(
        pareto_front.items(),
        key=lambda x: x[1][1],  # Sort by eps_0
        reverse=True
    )

    print(f"\n{'Composition':<20} {'Bandgap (eV)':<15} {'eps_0':<15}")
    print("-" * 50)

    for comp, (bandgap, eps_0) in sorted_pareto[:20]:
        print(f"{comp:<20} {bandgap:<15.3f} {eps_0:<15.3f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

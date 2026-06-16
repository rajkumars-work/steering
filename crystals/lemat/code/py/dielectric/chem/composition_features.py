"""Shared composition feature utilities.

Includes:
- Magpie ElementProperty features (matminer)
- Electronegativity range (max - min)
- Stoichiometric entropy (-sum f log f)
- Anion flags for O, F, Cl, S, N
"""

from __future__ import annotations

from typing import List, Tuple
from collections import Counter

import numpy as np
from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementProperty

ANION_ELEMENTS = ["O", "F", "Cl", "S", "N"]
HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"S", "Se", "Te"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
COMMON_ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P"}


def get_magpie_featurizer() -> ElementProperty:
    return ElementProperty.from_preset("magpie", impute_nan=True)


def has_element(comp: Composition, symbol: str) -> bool:
    return symbol in comp.get_el_amt_dict()


def electronegativity_range(comp: Composition) -> float:
    vals = []
    for el in comp.elements:
        x = Element(el.symbol).X
        if x is not None:
            vals.append(float(x))
    if not vals:
        return 0.0
    return max(vals) - min(vals)


def stoichiometric_entropy(comp: Composition) -> float:
    fracs = [float(comp.get_atomic_fraction(el)) for el in comp.elements]
    ent = 0.0
    for f in fracs:
        if f > 0:
            ent -= f * np.log(f)
    return float(ent)


def compute_magpie_and_custom(
    comp: Composition, magpie: ElementProperty
) -> Tuple[np.ndarray, List[str]]:
    features: List[float] = []
    labels: List[str] = []

    try:
        magpie_feats = magpie.featurize(comp)
    except Exception:
        magpie_feats = [0.0] * len(magpie.feature_labels())
    features.extend(magpie_feats)
    labels.extend(magpie.feature_labels())

    features.append(electronegativity_range(comp))
    labels.append("electronegativity_range")

    features.append(stoichiometric_entropy(comp))
    labels.append("stoichiometric_entropy")

    for sym in ANION_ELEMENTS:
        features.append(1.0 if has_element(comp, sym) else 0.0)
        labels.append(f"has_{sym}")

    return np.asarray(features, dtype=np.float32), labels


def _fraction_by_elements(comp: Composition, elements: set[str]) -> float:
    return float(
        sum(
            comp.get_atomic_fraction(Element(sym))
            for sym in elements
            if sym in comp.get_el_amt_dict()
        )
    )


def _cation_fractions(comp: Composition) -> List[Tuple[Element, float]]:
    fracs = []
    for el in comp.elements:
        if el.symbol in COMMON_ANIONS:
            continue
        fracs.append((el, float(comp.get_atomic_fraction(el))))
    return fracs


def _is_heavy_p_block(el: Element) -> bool:
    block = getattr(el, "block", None)
    return block == "p" and el.Z >= 31


def _classify_comp(comp: Composition) -> str:
    o_frac = _fraction_by_elements(comp, {"O"})
    hal_frac = _fraction_by_elements(comp, HALOGENS)
    nit_frac = _fraction_by_elements(comp, {"N"})
    chal_frac = _fraction_by_elements(comp, CHALCOGENS)

    if hal_frac >= 0.4 and o_frac <= 0.1 and nit_frac <= 0.1 and chal_frac <= 0.1:
        return "halides"

    if o_frac >= 0.4 and hal_frac <= 0.1:
        cation_fracs = _cation_fractions(comp)
        total_cation = sum(frac for _, frac in cation_fracs)
        if total_cation <= 0:
            return "oxides"
        alk_earth = sum(
            frac for el, frac in cation_fracs if el.symbol in ALKALINE_EARTH
        ) / total_cation
        heavy_p = sum(
            frac for el, frac in cation_fracs if _is_heavy_p_block(el)
        ) / total_cation
        if alk_earth >= 0.5:
            return "alkaline-earth oxides"
        if heavy_p >= 0.5:
            return "heavy p-block oxides"
        return "oxides"

    if nit_frac >= 0.4 and o_frac <= 0.1 and hal_frac <= 0.1:
        return "nitrides"

    if chal_frac >= 0.4 and o_frac <= 0.1 and hal_frac <= 0.1:
        return "chalcogenides"

    return "other"


def dominant_chemistry(compositions: List[str]) -> str:
    """
    Return a human-readable dominant chemistry label for a list of compositions.

    If no clear pattern emerges, returns "random mix of everything".
    """
    if not compositions:
        return "random mix of everything"

    labels: List[str] = []
    for formula in compositions:
        try:
            comp = Composition(formula)
        except Exception:
            labels.append("other")
            continue
        labels.append(_classify_comp(comp))

    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    if len(compositions) == 1:
        return label
    if count / len(compositions) >= 0.6:
        return label
    return "random mix of everything"

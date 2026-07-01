"""
Composition-level filters for pipeline stages.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Composition, Element

from composition_features import stoichiometric_entropy
from pareto_front import ParetoFront
from prototypes import is_abx3
from soap import predict_properties


def is_charge_balanced(formula: str) -> bool:
    comp = Composition(formula)
    try:
        oxi = comp.oxi_state_guesses(max_sites=-1)
        return len(oxi) > 0
    except Exception:
        return False


def oxidation_state_exists(formula: str) -> bool:
    comp = Composition(formula)
    try:
        oxi = comp.oxi_state_guesses(max_sites=-1)
        return len(oxi) > 0
    except Exception:
        return False


def electronegativity_spread(formula: str) -> Optional[float]:
    comp = Composition(formula)
    chis = []
    for el in comp.elements:
        chi = Element(el.symbol).X
        if chi is None:
            return None
        chis.append(chi)
    return max(chis) - min(chis)


def passes_electronegativity_filter(formula: str, max_spread: float = 3.3) -> bool:
    spread = electronegativity_spread(formula)
    if spread is None:
        return False
    return spread <= max_spread


def ionic_radius_ratio(formula: str) -> Optional[float]:
    comp = Composition(formula)
    radii = []
    for el in comp.elements:
        e = Element(el.symbol)
        r = e.average_ionic_radius
        if r is None:
            return None
        # Guard against missing/invalid radii encoded as 0.0
        if r <= 0:
            return None
        radii.append(r)
    return max(radii) / min(radii)


def passes_radius_filter(formula: str, max_ratio: float = 2.5) -> bool:
    ratio = ionic_radius_ratio(formula)
    if ratio is None:
        return False
    return ratio <= max_ratio


def entropy_value(formula: str) -> Optional[float]:
    try:
        comp = Composition(formula)
        return float(stoichiometric_entropy(comp))
    except Exception:
        return None


def passes_entropy_filter(formula: str, max_entropy: float = 1.3) -> bool:
    ent = entropy_value(formula)
    if ent is None:
        return False
    return ent <= max_entropy


O_RADIUS = 1.40  # Shannon O2- approx


def goldschmidt_tolerance(formula: str) -> Optional[float]:
    comp = Composition(formula)
    el_amt = comp.get_el_amt_dict()
    cations = [el for el in el_amt if el != "O"]
    if len(cations) != 2:
        return None
    radii = {el: Element(el).average_ionic_radius for el in cations}
    if None in radii.values():
        return None
    A = max(radii, key=radii.get)
    B = min(radii, key=radii.get)
    rA = radii[A]
    rB = radii[B]
    return (rA + O_RADIUS) / (np.sqrt(2) * (rB + O_RADIUS))


def passes_tolerance_filter(formula: str, lower: float = 0.825, upper: float = 1.05) -> bool:
    t = goldschmidt_tolerance(formula)
    if t is None:
        return True
    return lower <= t <= upper


def chemistry_prefilter(formula: str) -> bool:
    if not passes_electronegativity_filter(formula):
        return False
    if not passes_radius_filter(formula):
        return False
    return True


def passes_chemistry_filters(formula: str) -> bool:
    if not is_charge_balanced(formula):
        return False
    if not oxidation_state_exists(formula):
        return False
    if not passes_electronegativity_filter(formula):
        return False
    if not passes_radius_filter(formula):
        return False
    if not passes_entropy_filter(formula):
        return False
    return True


def filter_rows_by_chemistry(
    rows: Iterable[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    comp_rows: Dict[str, Dict[str, object]] = {}
    for row in rows:
        formula = str(row.get("composition_formula", "")).strip()
        if not formula:
            continue
        if not passes_chemistry_filters(formula):
            continue
        comp_rows.setdefault(formula, row)
    return comp_rows


def filter_pareto_front(
    expanded_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not expanded_rows:
        return []

    ranker = ParetoFront(maximize_bandgap=True, maximize_eps_0=True)
    for row in expanded_rows:
        formula = str(row.get("composition_formula", "")).strip()
        if not formula:
            continue
        bandgap = row.get("bandgap")
        eps_0 = row.get("eps_0")
        if bandgap is None or eps_0 is None:
            continue
        ranker.add(formula, float(bandgap), float(eps_0))

    front = ranker.pareto()
    rank1 = ranker.pareto(1)
    selected = set(front.keys()) | set(rank1.keys())

    filtered: List[Dict[str, object]] = []
    for row in expanded_rows:
        formula = str(row.get("composition_formula", "")).strip()
        if formula not in selected:
            continue
        rank = 0 if formula in front else 1
        row_out = dict(row)
        row_out["pareto_rank"] = rank
        filtered.append(row_out)

    return filtered


def stage3_filter(expanded_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return filter_pareto_front(expanded_rows)


def filter_candidates_by_training_species(
    candidate_atoms: Sequence,
    candidate_meta: Sequence[Dict[str, object]],
    comp_to_indices: Dict[str, List[int]],
    training_species: Iterable[int],
) -> Tuple[List, List[Dict[str, object]], Dict[str, List[int]], set[int]]:
    training_species_set = {int(Z) for Z in training_species}

    kept_atoms: List = []
    kept_meta: List[Dict[str, object]] = []
    old_to_new: Dict[int, int] = {}
    unknown_species: set[int] = set()

    for i, atoms in enumerate(candidate_atoms):
        present = {int(Z) for Z in atoms.get_atomic_numbers()}
        missing = present - training_species_set
        if missing:
            unknown_species.update(missing)
            continue
        new_idx = len(kept_atoms)
        old_to_new[i] = new_idx
        meta = dict(candidate_meta[i])
        meta["candidate_index"] = new_idx
        kept_atoms.append(atoms)
        kept_meta.append(meta)

    new_comp_to_indices = {
        formula: [old_to_new[i] for i in idxs if i in old_to_new]
        for formula, idxs in comp_to_indices.items()
        if any(i in old_to_new for i in idxs)
    }

    return kept_atoms, kept_meta, new_comp_to_indices, unknown_species


def filter_energy_above_hull(
    stage4_rows: List[Dict[str, object]],
    stage4_atoms: List,
    extxyz_path: str,
    max_energy_per_atom: float = 0.2,
    n_jobs: Optional[int] = None,
) -> List[Dict[str, object]]:
    if not stage4_rows:
        return []

    predictions = predict_properties(
        stage4_atoms,
        model_names=("energy_above_hull",),
        extxyz_path=extxyz_path,
        n_jobs=n_jobs,
    )
    energy_pred = predictions["energy_above_hull"]

    filtered: List[Dict[str, object]] = []
    for row, atoms, energy in zip(stage4_rows, stage4_atoms, energy_pred):
        energy_val = float(energy)
        n_atoms = len(atoms)
        if n_atoms <= 0:
            continue
        energy_per_atom = energy_val / n_atoms
        if energy_per_atom > max_energy_per_atom:
            continue
        formula = str(row.get("composition_formula", "")).strip()
        if formula and is_abx3(formula) and not passes_tolerance_filter(formula):
            continue
        row_out = dict(row)
        row_out["pred_energy_above_hull"] = energy_val
        row_out["pred_energy_above_hull_per_atom"] = energy_per_atom
        row_out["n_atoms"] = n_atoms
        filtered.append(row_out)

    deduped: Dict[str, Dict[str, object]] = {}
    for row in filtered:
        formula = str(row.get("composition_formula", "")).strip()
        if not formula:
            continue
        existing = deduped.get(formula)
        if existing is None:
            deduped[formula] = row
            continue
        if row["pred_energy_above_hull_per_atom"] < existing["pred_energy_above_hull_per_atom"]:
            deduped[formula] = row

    return list(deduped.values())


def stage5_filter_energy(
    stage4_rows: List[Dict[str, object]],
    stage4_atoms: List,
    extxyz_path: str,
    max_energy_per_atom: float = 0.2,
    n_jobs: Optional[int] = None,
) -> List[Dict[str, object]]:
    return filter_energy_above_hull(
        stage4_rows,
        stage4_atoms,
        extxyz_path=extxyz_path,
        max_energy_per_atom=max_energy_per_atom,
        n_jobs=n_jobs,
    )


__all__ = [
    "is_charge_balanced",
    "oxidation_state_exists",
    "electronegativity_spread",
    "passes_electronegativity_filter",
    "ionic_radius_ratio",
    "passes_radius_filter",
    "entropy_value",
    "passes_entropy_filter",
    "goldschmidt_tolerance",
    "passes_tolerance_filter",
    "chemistry_prefilter",
    "passes_chemistry_filters",
    "filter_rows_by_chemistry",
    "filter_pareto_front",
    "filter_candidates_by_training_species",
    "filter_energy_above_hull",
    "stage3_filter",
    "stage5_filter_energy",
]

"""
Prototype selection helpers based on simple stoichiometry patterns.
"""

from __future__ import annotations

from typing import Iterable, List, Union

from pymatgen.core import Composition


def _as_composition(comp: Union[str, Composition]) -> Composition:
    if isinstance(comp, Composition):
        return comp
    return Composition(comp)


def _matches_target_fractions(comp: Composition, target: Iterable[float], tol: float) -> bool:
    fracs = comp.fractional_composition.as_dict()
    if len(fracs) != len(list(target)):
        return False
    values = sorted(fracs.values())
    target_vals = sorted(target)
    return all(abs(v - t) < tol for v, t in zip(values, target_vals))


def is_abx3(comp: Union[str, Composition], tol: float = 0.15) -> bool:
    comp = _as_composition(comp)
    return _matches_target_fractions(comp, [1 / 5, 1 / 5, 3 / 5], tol)


def is_ao2(comp: Union[str, Composition], tol: float = 0.15) -> bool:
    comp = _as_composition(comp)
    return _matches_target_fractions(comp, [1 / 3, 2 / 3], tol)


def is_ab2o4(comp: Union[str, Composition], tol: float = 0.10) -> bool:
    comp = _as_composition(comp)
    return _matches_target_fractions(comp, [1 / 7, 2 / 7, 4 / 7], tol)


def is_ab(comp: Union[str, Composition], tol: float = 0.10) -> bool:
    comp = _as_composition(comp)
    return _matches_target_fractions(comp, [1 / 2, 1 / 2], tol)


def is_perovskite_like(comp: Union[str, Composition]) -> bool:
    return is_abx3(comp)


def is_fluorite_like(comp: Union[str, Composition]) -> bool:
    return is_ao2(comp)


def is_spinel_like(comp: Union[str, Composition]) -> bool:
    return is_ab2o4(comp)


def is_rocksalt_like(comp: Union[str, Composition]) -> bool:
    return is_ab(comp)


def is_simple(comp: Union[str, Composition]) -> bool:
    comp = _as_composition(comp)
    return len(comp.elements) <= 2


def choose_prototype(comp: Union[str, Composition]) -> List[str]:
    comp = _as_composition(comp)

    if is_perovskite_like(comp):
        return ["perovskite"]

    if is_fluorite_like(comp):
        return ["fluorite"]

    if is_spinel_like(comp):
        return ["spinel"]

    if is_rocksalt_like(comp):
        return ["rocksalt"]

    if is_simple(comp):
        return ["rocksalt"]

    return ["rocksalt"]  # fallback


__all__ = [
    "choose_prototype",
    "is_perovskite_like",
    "is_fluorite_like",
    "is_spinel_like",
    "is_rocksalt_like",
    "is_abx3",
    "is_ao2",
    "is_ab2o4",
    "is_ab",
    "is_simple",
]

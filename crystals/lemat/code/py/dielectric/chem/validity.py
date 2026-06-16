"""Structural-validity checks for ASE Atoms: SMACT + BVS + atomic distances.

Each check returns a dict with a ``*_pass`` boolean. ``check_validity()`` runs
all three under a single 30-second SIGALRM budget and merges the dicts. All
public functions are also individually timeout-wrapped so callers can use them
from any pipeline.

No external artifacts are required — pure pymatgen / smact / numpy.
"""
from __future__ import annotations

import contextlib
import itertools
import os
from collections import Counter

import numpy as np
import smact
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from smact.screening import pauling_test

from chem._timeout import CheckTimeout, run_with_timeout

_ANION_ELEMENTS = frozenset({"O", "N", "S", "F", "Cl", "Br", "I", "Se", "Te", "H"})


@contextlib.contextmanager
def _suppress_spglib_stderr():
    """Silence spglib's C-level stderr (warnings like ssm_get_exact_positions failed)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def _has_bad_coords(atoms: Atoms) -> bool:
    return (not np.all(np.isfinite(atoms.positions))
            or not np.all(np.isfinite(atoms.cell.array)))


def _safe_get_structure(atoms: Atoms):
    """ASE → pymatgen Structure, or None for degenerate cells / NaN coords."""
    if _has_bad_coords(atoms):
        return None
    try:
        s = AseAtomsAdaptor.get_structure(atoms)
        if np.any(np.isnan(s.frac_coords)):
            return None
        return s
    except (np.linalg.LinAlgError, ValueError):
        return None


def _check_min_dist_impl(atoms: Atoms, cutoff: float) -> dict:
    if _has_bad_coords(atoms):
        return {"min_dist_A": None, "volume_A3": None, "min_dist_pass": False}
    vol = float(atoms.get_volume())
    if vol < 0.1:
        return {"min_dist_A": None, "volume_A3": vol, "min_dist_pass": False}
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(dists.min())
    return {"min_dist_A": min_dist, "volume_A3": vol, "min_dist_pass": min_dist >= cutoff}


def check_atomic_distances(atoms: Atoms, cutoff: float = 0.5, timeout: float = 30) -> dict:
    """Minimum interatomic distance check (hard reject if min < cutoff Å).

    Also rejects unit cells with volume < 0.1 Å³. Mirrors CDVAE's
    ``structure_validity``.
    """
    try:
        return run_with_timeout(_check_min_dist_impl, atoms, cutoff, timeout=timeout)
    except CheckTimeout:
        return {"min_dist_A": None, "volume_A3": None, "min_dist_pass": False,
                "min_dist_timed_out": True}


# Backward-compatible alias
check_min_dist = check_atomic_distances


def _check_smact_impl(atoms: Atoms, use_pauling_test: bool, include_alloys: bool) -> dict:
    elem_counter = Counter(atoms.get_chemical_symbols())
    elems = tuple(elem_counter.keys())
    counts = np.array([elem_counter[e] for e in elems], dtype=int)
    counts = counts // np.gcd.reduce(counts)

    if len(set(elems)) == 1:
        return {"smact_pass": True}
    if include_alloys and all(e in smact.metals for e in elems):
        return {"smact_pass": True}

    space = smact.element_dictionary(elems)
    smact_elems = list(space.values())
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    if any(ox is None or len(ox) == 0 for ox in ox_combos):
        return {"smact_pass": False}

    threshold = int(counts.max())
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(int(c),) for c in counts]
        # smact 3.x returns (bool, list); smact 4.x returns just a list. Handle both.
        result = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        cn_e = bool(result[0]) if isinstance(result, tuple) else bool(result)
        if not cn_e:
            continue
        if use_pauling_test:
            try:
                if not pauling_test(ox_states, electronegs):
                    continue
            except TypeError:
                pass
        return {"smact_pass": True}

    return {"smact_pass": False}


def check_smact(atoms: Atoms, use_pauling_test: bool = True,
                include_alloys: bool = True, timeout: float = 30) -> dict:
    """SMACT charge-balance + Pauling electronegativity check (hard reject).

    Single-element and all-metal compositions pass automatically. Mirrors CDVAE's
    ``smact_validity``.
    """
    try:
        return run_with_timeout(
            _check_smact_impl, atoms, use_pauling_test, include_alloys, timeout=timeout)
    except CheckTimeout:
        return {"smact_pass": False, "smact_timed_out": True}


def _check_bvs_impl(atoms: Atoms, max_deviation: float, metallic_max_deviation: float) -> dict:
    if _has_bad_coords(atoms):
        return {"bvs_max_deviation": None, "bvs_pass": False}
    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        elements = {str(el) for el in structure.composition.elements}
        is_metallic = not bool(elements & _ANION_ELEMENTS)
        threshold = metallic_max_deviation if is_metallic else max_deviation

        with _suppress_spglib_stderr():
            valences = BVAnalyzer().get_valences(structure)
        max_dev = float(max(abs(v - round(v)) for v in valences))
        return {
            "bvs_max_deviation": round(max_dev, 3),
            "bvs_is_metallic": is_metallic,
            "bvs_pass": max_dev <= threshold,
        }
    except Exception:
        # BVAnalyzer often fails outright on metals; treat metallic as pass
        if not _has_bad_coords(atoms):
            try:
                structure = AseAtomsAdaptor.get_structure(atoms)
                elements = {str(el) for el in structure.composition.elements}
                if not (elements & _ANION_ELEMENTS):
                    return {"bvs_max_deviation": None, "bvs_is_metallic": True, "bvs_pass": True}
            except Exception:
                pass
        return {"bvs_max_deviation": None, "bvs_pass": False}


def check_bvs(atoms: Atoms, max_deviation: float = 0.5,
              metallic_max_deviation: float = 1.5, timeout: float = 30) -> dict:
    """Bond Valence Sum self-consistency check (hard reject).

    Geometry-aware complement to SMACT. Uses pymatgen's BVAnalyzer and rejects
    if the worst per-site |BVS − formal charge| exceeds *max_deviation*. Metallic
    compositions (no common anions) use the relaxed *metallic_max_deviation* since
    BVS is designed for ionic compounds.
    """
    try:
        return run_with_timeout(
            _check_bvs_impl, atoms, max_deviation, metallic_max_deviation, timeout=timeout)
    except CheckTimeout:
        return {"bvs_max_deviation": None, "bvs_pass": False, "bvs_timed_out": True}


def _check_validity_impl(atoms: Atoms, min_dist_cutoff: float,
                         bvs_max_deviation: float,
                         bvs_metallic_max_deviation: float,
                         smact_use_pauling: bool,
                         smact_include_alloys: bool) -> dict:
    result: dict = {}
    result.update(_check_min_dist_impl(atoms, min_dist_cutoff))
    result.update(_check_smact_impl(atoms, smact_use_pauling, smact_include_alloys))
    result.update(_check_bvs_impl(atoms, bvs_max_deviation, bvs_metallic_max_deviation))
    result["validity_pass"] = all(result.get(k, False) for k in
                                  ("min_dist_pass", "smact_pass", "bvs_pass"))
    return result


def check_validity(
    atoms: Atoms,
    min_dist_cutoff: float = 0.5,
    bvs_max_deviation: float = 0.5,
    bvs_metallic_max_deviation: float = 1.5,
    smact_use_pauling: bool = True,
    smact_include_alloys: bool = True,
    timeout: float = 30,
) -> dict:
    """Run SMACT + BVS + atomic-distance checks under one 30-s budget.

    Returns the merged sub-check dict plus ``validity_pass`` (all three pass).
    On timeout, returns ``validity_pass=False`` and ``validity_timed_out=True``.
    """
    try:
        return run_with_timeout(
            _check_validity_impl, atoms, min_dist_cutoff, bvs_max_deviation,
            bvs_metallic_max_deviation, smact_use_pauling, smact_include_alloys,
            timeout=timeout)
    except CheckTimeout:
        return {"validity_pass": False, "validity_timed_out": True}

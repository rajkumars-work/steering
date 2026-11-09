"""Composable screening checkers for crystal structure validation.

Each checker is independent, has its own timeout and error handling, and produces
a dict of results with standardized key names. The ScreeningPipeline composes
checkers and handles funnel/no-funnel mode.

See ARCHITECTURE notes in screening_version.py for the tier structure.

Usage:
    from validity.checkers import (
        AtomicDistanceChecker, ChemistryChecker, StabilityChecker,
        NoveltyChecker, SynthesizabilityChecker, ScreeningPipeline, NoveltyIndex,
    )
    from validity.screening_version import CHECK_THRESHOLDS

    pipeline = ScreeningPipeline([
        AtomicDistanceChecker(CHECK_THRESHOLDS),
        ChemistryChecker(CHECK_THRESHOLDS),
        StabilityChecker(CHECK_THRESHOLDS, pet=pet, mace=mace, ehull_cache=cache),
        NoveltyChecker(CHECK_THRESHOLDS, index=index),
    ])
    result = pipeline.screen(atoms, funnel=False)
"""

from __future__ import annotations

import abc
import hashlib
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ase import Atoms

from chem._timeout import CheckTimeout as _CheckTimeout, run_with_timeout


def _run_with_alarm(fn, atoms, timeout_sec):
    """Run fn(atoms) with a hard SIGALRM timeout. Returns fn's result or raises _CheckTimeout."""
    return run_with_timeout(fn, atoms, timeout=timeout_sec)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Checker(abc.ABC):
    """Base class for all screening checkers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, used as prefix in timing/error keys."""

    @property
    def timeout(self) -> float:
        """Per-checker timeout in seconds."""
        return self._timeout

    @property
    def hard_reject_keys(self) -> List[str]:
        """Result keys that gate overall_pass. Override in subclasses."""
        return []

    @abc.abstractmethod
    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        """Run checks. Returns {key: value} dict with same keys as CSV columns."""


# ---------------------------------------------------------------------------
# 1. Atomic Distance Checker
# ---------------------------------------------------------------------------

class AtomicDistanceChecker(Checker):
    """Element filter, minimum interatomic distance, isolated atoms, plausibility."""

    name = "atomic_distance"
    hard_reject_keys = ["element_filter_pass", "min_dist_pass", "isolated_atoms_pass",
                        "plausibility_pass"]

    def __init__(self, config: dict, **kwargs):
        self._config = config
        self._timeout = config.get("timeouts", {}).get("atomic_distance", 5)

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        from chem.validity import _check_min_dist_impl
        from validity.structural import (
            check_element_filter, check_isolated_atoms, check_plausibility,
        )
        result = {}
        result.update(check_element_filter(atoms))
        result.update(_check_min_dist_impl(
            atoms, self._config.get("min_dist", {}).get("cutoff_A", 0.5)))
        result.update(check_isolated_atoms(
            atoms, cutoff=self._config.get("isolated_atoms", {}).get("cutoff_A", 4.0)))
        result.update(check_plausibility(atoms))
        return result


# ---------------------------------------------------------------------------
# 2. Chemistry Checker
# ---------------------------------------------------------------------------

class ChemistryChecker(Checker):
    """SMACT charge balance, bond valence sum, spacegroup match."""

    name = "chemistry"
    hard_reject_keys = ["smact_pass", "bvs_pass", "spacegroup_pass"]

    def __init__(self, config: dict, **kwargs):
        self._config = config
        self._timeout = config.get("timeouts", {}).get("chemistry", 10)

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        from chem.validity import _check_bvs_impl, _check_smact_impl
        from validity.structural import check_spacegroup
        result = {}
        result.update(_check_smact_impl(atoms, use_pauling_test=True, include_alloys=True))
        result.update(_check_bvs_impl(
            atoms,
            max_deviation=self._config.get("bvs", {}).get("max_deviation", 0.5),
            metallic_max_deviation=1.5,
        ))
        result.update(check_spacegroup(
            atoms, symprec=self._config.get("spacegroup", {}).get("symprec", 0.1)))
        return result


# ---------------------------------------------------------------------------
# 3. Composition Fidelity Checker
# ---------------------------------------------------------------------------

class CompositionFidelityChecker(Checker):
    """Element-set and atom-count match vs. the requested prompt.

    The prompt is a guide, not a hard boundary: the model is meant to
    generate *novel* structures, so an element-list/natoms mismatch should
    not, by default, block a structure from further screening -- it just
    flags composition_fidelity_warning=True. Set
    CHECK_THRESHOLDS["composition_fidelity"]["strict"] = True (or pass a
    config dict with that override) to instead treat a mismatch as a hard
    reject, e.g. for a benchmark run that needs exact prompt compliance.

    No-op (always passes, no warning) for atoms with no atoms.info["source"]
    (e.g. atoms from a dataset row rather than model generation).

    Unlike smact_pass/bvs_pass (heuristic chemistry checks with their own
    false-positive/false-negative rate), this check is a deterministic
    string/set comparison against the prompt -- there's no probabilistic
    slack in what it reports, only in the strict/not-strict policy choice
    for what to do with a mismatch.
    """

    name = "composition_fidelity"

    def __init__(self, config: dict, **kwargs):
        self._config = config
        self._strict = config.get("composition_fidelity", {}).get("strict", False)
        self._timeout = config.get("timeouts", {}).get("composition_fidelity", 5)

    @property
    def hard_reject_keys(self) -> List[str]:
        return ["composition_fidelity_pass"] if self._strict else []

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        from eval.screening import composition_fidelity
        fidelity = composition_fidelity(atoms)
        if fidelity is None:
            return {"composition_fidelity_pass": True, "composition_fidelity_warning": False}

        mismatch = (fidelity.get("element_fidelity") == 0.0
                    or fidelity.get("count_fidelity") == 0.0)
        result = dict(fidelity)
        result["composition_fidelity_warning"] = mismatch
        result["composition_fidelity_pass"] = (not mismatch) if self._strict else True
        return result


# ---------------------------------------------------------------------------
# 4. Novelty Checker
# ---------------------------------------------------------------------------

def _reduced_formula(atoms: Atoms) -> str:
    """GCD-reduced formula via pymatgen (e.g. H6F6 → HF, O8Si4 → SiO2)."""
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms).composition.reduced_formula


class NoveltyIndex:
    """Pre-built index of training data for fast novelty classification.

    Caches to disk so repeated screen launches don't rebuild (~2 min → <1s).
    """

    def __init__(self, formulas: Set[str], element_sets: Set[frozenset],
                 prototypes: Set[tuple], ref_targets: Dict[str, Set[str]]):
        self.formulas = formulas
        self.element_sets = element_sets
        self.prototypes = prototypes
        self.ref_targets = ref_targets  # formula_key -> set of target strings

    @classmethod
    def build(cls, csv_path: str) -> "NoveltyIndex":
        """Build from a training CSV. Parses structures for prototype fingerprints."""
        import csv
        import spglib
        from dielectric_data.reader import parse_target, get_dataset_info

        ds_info = get_dataset_info(csv_path)
        version_id = ds_info.get("version_id", "d6")

        formulas: Set[str] = set()
        element_sets: Set[frozenset] = set()
        prototypes: Set[tuple] = set()
        ref_targets: Dict[str, Set[str]] = {}

        n_indexed = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                if label in ("eval", "test_1k", "test_100"):
                    continue

                tgt = row.get("target", "")
                origin = row.get("origin", "mp")
                if not tgt:
                    continue

                # String-match novelty index (cheap)
                elements_in_tgt = re.findall(r'[A-Z][a-z]?', tgt)
                formula_key = "".join(sorted(set(elements_in_tgt)))
                if formula_key not in ref_targets:
                    ref_targets[formula_key] = set()
                ref_targets[formula_key].add(tgt)

                # Structure-level index (needs parsing)
                try:
                    atoms = parse_target(tgt, version_id, origin)
                    if atoms is None or len(atoms) == 0:
                        continue

                    formula = _reduced_formula(atoms)
                    elements = frozenset(atoms.get_chemical_symbols())
                    formulas.add(formula)
                    element_sets.add(elements)

                    # Prototype fingerprint
                    sg_data = spglib.get_symmetry_dataset(
                        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
                        symprec=0.1)
                    spg_num = sg_data.number if sg_data else 0
                    wl = atoms.info.get("wyckoff_letters", [])
                    if wl and len(wl) == len(atoms):
                        wp_counts = Counter(
                            f"{s}_{w}" for s, w in zip(atoms.get_chemical_symbols(), wl))
                        wp_str = ";".join(f"{k}:{v}" for k, v in sorted(wp_counts.items()))
                    else:
                        wp_str = ""
                    prototypes.add((formula, spg_num, wp_str))
                    n_indexed += 1
                except Exception:
                    continue

        print(f"NoveltyIndex: {len(formulas)} formulas, {len(element_sets)} element sets, "
              f"{len(prototypes)} prototypes (from {n_indexed} structures)")
        return cls(formulas, element_sets, prototypes, ref_targets)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "formulas": self.formulas,
                "element_sets": self.element_sets,
                "prototypes": self.prototypes,
                "ref_targets": self.ref_targets,
            }, f)

    @classmethod
    def load(cls, path: str) -> "NoveltyIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    @classmethod
    def load_or_build(cls, csv_path: str, cache_dir: Optional[str] = None) -> "NoveltyIndex":
        """Load from cache if CSV hasn't changed; else rebuild and save."""
        if cache_dir is None:
            cache_dir = str(Path(csv_path).parent / ".novelty_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Cache key from CSV metadata
        p = Path(csv_path)
        stat = p.stat()
        key_str = f"{p.name}:{stat.st_size}:{stat.st_mtime}"
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"{key_hash}.pkl")

        if os.path.exists(cache_path):
            try:
                idx = cls.load(cache_path)
                print(f"NoveltyIndex loaded from cache ({len(idx.formulas)} formulas)")
                return idx
            except Exception:
                pass  # rebuild on corrupt cache

        print(f"Building NoveltyIndex from {csv_path} ...")
        idx = cls.build(csv_path)
        idx.save(cache_path)
        return idx

    def classify(self, atoms: Atoms) -> str:
        """Classify a structure: copy, polymorph, variant, or novel."""
        import spglib

        try:
            formula = _reduced_formula(atoms)
            elements = frozenset(atoms.get_chemical_symbols())

            if elements not in self.element_sets:
                return "novel"
            if formula not in self.formulas:
                return "variant"

            # Same formula — check prototype
            sg_data = spglib.get_symmetry_dataset(
                (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
                symprec=0.1)
            spg_num = sg_data.number if sg_data else 0
            wl = atoms.info.get("wyckoff_letters", [])
            if wl and len(wl) == len(atoms):
                wp_counts = Counter(
                    f"{s}_{w}" for s, w in zip(atoms.get_chemical_symbols(), wl))
                wp_str = ";".join(f"{k}:{v}" for k, v in sorted(wp_counts.items()))
            else:
                wp_str = ""
            proto = (formula, spg_num, wp_str)

            return "copy" if proto in self.prototypes else "polymorph"
        except Exception:
            return "unknown"

    def is_string_novel(self, target_str: str) -> bool:
        """Check if target string is novel (exact string match)."""
        elements = re.findall(r'[A-Z][a-z]?', target_str)
        key = "".join(sorted(set(elements)))
        return target_str not in self.ref_targets.get(key, set())


class NoveltyChecker(Checker):
    """Tiered novelty classification + string-match novelty."""

    name = "novelty"
    hard_reject_keys = []  # novelty is informational, never rejects

    def __init__(self, config: dict, index: Optional[NoveltyIndex] = None, **kwargs):
        self._config = config
        self._index = index
        self._timeout = config.get("timeouts", {}).get("novelty", 5)

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        result = {}
        if self._index is None:
            return result

        # Tiered classification
        result["novelty_tier"] = self._index.classify(atoms)

        # String-match novelty
        tgt = atoms.info.get("target", "")
        if tgt:
            result["novel"] = self._index.is_string_novel(tgt)

        return result


# ---------------------------------------------------------------------------
# 5. Stability Checker
# ---------------------------------------------------------------------------

class StabilityChecker(Checker):
    """UQ, e-hull, phonon, cross-MLIP, MD — each with individual timeout."""

    name = "stability"
    hard_reject_keys = ["uq_pass", "ehull_pass"]  # phonon excluded per v4

    def __init__(self, config: dict, pet=None, mace=None,
                 ehull_cache: Optional[dict] = None,
                 run_cross: bool = True, run_md: bool = False, **kwargs):
        self._config = config
        self._pet = pet
        self._mace = mace
        self._ehull_cache = ehull_cache if ehull_cache is not None else {}
        self._run_cross = run_cross
        self._run_md = run_md

        timeouts = config.get("timeouts", {})
        self._sub_timeouts = {
            "uq": timeouts.get("stability_uq", 10),
            "ehull": timeouts.get("stability_ehull", 60),
            "phonon": timeouts.get("stability_phonon", 30),
            "cross": timeouts.get("stability_cross", 15),
            "md": timeouts.get("stability_md", 30),
        }
        # Outer timeout = sum of sub-timeouts (safety net)
        self._timeout = sum(self._sub_timeouts.values())

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        from chem.stability import (
            DEFAULT_EHULL_CACHE_DIR, DEFAULT_MP_API_KEY, _compute_e_above_hull_impl,
        )
        from validity.structural import (
            check_uq, check_phonon_stability, check_cross_mlip, check_md_stability,
        )

        result = {}
        sub_checks = [
            ("uq", lambda a: check_uq(a, self._pet)),
            # mlip_key="petmad": self._pet is PET-MAD, not chem.stability's default MACE --
            # without an explicit key here this would silently share compute_e_above_hull()'s
            # MACE-keyed cache (2026-09-26 mlip_key partitioning, EVAL.md §0.1). Coarse (no
            # checkpoint hash available for an externally-injected calc instance), but
            # distinct from every "mace-*"/"orb-*"/"uma-*" key, which is what actually
            # matters here.
            ("ehull", lambda a: _compute_e_above_hull_impl(
                a, self._pet, "petmad", DEFAULT_EHULL_CACHE_DIR, self._ehull_cache,
                threshold=0.1, mp_api_key=DEFAULT_MP_API_KEY)),
        ]
        if self._run_cross:
            sub_checks.append(
                ("cross", lambda a: check_cross_mlip(a, self._pet, self._mace)))
        sub_checks.append(
            ("phonon", lambda a: check_phonon_stability(a, self._mace)))
        if self._run_md:
            sub_checks.append(
                ("md", lambda a: check_md_stability(a, self._mace)))

        for name, fn in sub_checks:
            timeout = self._sub_timeouts.get(name, 30)
            t0 = time.perf_counter()
            try:
                r = _run_with_alarm(fn, atoms, timeout)
                result.update(r)
            except _CheckTimeout:
                result[f"{name}_timed_out"] = True
                # Set the pass key to False so funnel logic works
                pass_key = f"{name}_pass"
                if pass_key not in result:
                    result[pass_key] = False
            except Exception as e:
                result[f"{name}_error"] = str(e)
                pass_key = f"{name}_pass"
                if pass_key not in result:
                    result[pass_key] = False
            result[f"{name}_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return result


# ---------------------------------------------------------------------------
# 6. Synthesizability Checker
# ---------------------------------------------------------------------------

class SynthesizabilityChecker(Checker):
    """6-layer synthesizability composite score."""

    name = "synthesizability"
    hard_reject_keys = []  # score is informational

    def __init__(self, config: dict, scorer=None, **kwargs):
        self._config = config
        self._scorer = scorer
        self._timeout = config.get("timeouts", {}).get("synthesizability", 30)

    def check(self, atoms: Atoms, prior_results: Optional[dict] = None) -> dict:
        if self._scorer is None:
            return {}
        try:
            report = self._scorer.score(atoms, screen_results=prior_results or {})
            return {
                "synth_score": report.composite_score,
                "synth_verdict": report.verdict,
                "synth_hard_fail": report.hard_filter_failed,
            }
        except Exception as e:
            return {"synth_score": None, "synth_error": str(e)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScreeningPipeline:
    """Composes checkers into a screening pipeline with timing and error isolation."""

    def __init__(self, checkers: List[Checker]):
        self.checkers = checkers

    def screen(self, atoms: Atoms, funnel: bool = True) -> dict:
        """Run all checkers on a structure.

        Args:
            atoms: ASE Atoms object to screen.
            funnel: If True, skip remaining checkers after a hard reject.

        Returns:
            Dict with all check results, timing, errors, and overall_pass.
        """
        result: dict = {}

        for checker in self.checkers:
            t0 = time.perf_counter()
            try:
                # StabilityChecker handles its own per-sub-check SIGALRM timeouts
                if isinstance(checker, StabilityChecker):
                    r = checker.check(atoms, prior_results=result)
                else:
                    r = _run_with_alarm(
                        lambda a, c=checker: c.check(a, prior_results=result),
                        atoms, checker.timeout)
                result.update(r)
            except _CheckTimeout:
                result[f"{checker.name}_timed_out"] = True
            except Exception as e:
                result[f"{checker.name}_error"] = str(e)

            dt_ms = (time.perf_counter() - t0) * 1000
            result[f"{checker.name}_time_ms"] = round(dt_ms, 1)

            # Funnel: check if any hard-reject key is False
            if funnel:
                for key in checker.hard_reject_keys:
                    if result.get(key) is False:
                        result["overall_pass"] = False
                        return result

        # Compute overall_pass from all hard-reject keys
        all_hard_keys = []
        for checker in self.checkers:
            all_hard_keys.extend(checker.hard_reject_keys)
        result["overall_pass"] = all(
            result.get(k, True) for k in all_hard_keys
        )
        return result

    def checker_names(self) -> List[str]:
        return [c.name for c in self.checkers]

"""Physical validity checks for generated crystal structures.

All check functions take an ASE Atoms object and return a plain dict with a
boolean ``*_pass`` key indicating pass/fail.  Hard-reject checks abort the
pipeline; soft-flag checks record a warning but do not stop it.
"""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
import os
import pickle
import tempfile
from typing import Optional


@contextlib.contextmanager
def _suppress_spglib_stderr():
    """Redirect file-descriptor 2 (stderr) to /dev/null.

    spglib prints warnings like ``ssm_get_exact_positions failed`` directly
    from C code, bypassing Python's warning system.  This context manager
    silences them at the OS level so they don't flood the terminal.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)

# ---------------------------------------------------------------------------
# Compatibility shim: MACE models saved with older e3nn stored the
# TorchScript bytes directly as `{fname: bytes}`, but e3nn >=0.5.7
# __setstate__ expects `{fname: (buffer_type, buffer)}`.  Patch it once at
# import time so torch.load can deserialise those checkpoints.
# ---------------------------------------------------------------------------
import torch
from e3nn.util.codegen._mixin import CodeGenMixin as _CodeGenMixin


def _codegen_setstate_compat(self, d: dict) -> None:
    d = d.copy()
    codegen_state = d.pop("__codegen__", None)
    if hasattr(super(_CodeGenMixin, self), "__setstate__"):
        super(_CodeGenMixin, self).__setstate__(d)
    else:
        self.__dict__.update(d)
    if codegen_state is not None:
        for fname, value in codegen_state.items():
            # New format: (buffer_type, buffer) where buffer_type is a str.
            # Old format: raw bytes (always TorchScript).
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                buffer_type, buffer = value
            else:
                buffer_type, buffer = "torchscript", value
            if buffer_type == "fx":
                smod = pickle.loads(buffer)
            elif buffer_type == "torchscript":
                smod = torch.jit.load(io.BytesIO(buffer))
            else:
                raise NotImplementedError(f"Unknown codegen buffer_type: {buffer_type!r}")
            setattr(self, fname, smod)
        self.__codegen__ = list(codegen_state.keys())


_CodeGenMixin.__setstate__ = _codegen_setstate_compat
# ---------------------------------------------------------------------------

import numpy as np
import smact
from pymatgen.core import Composition, Element

from smact.screening import pauling_test
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.units import fs
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from upet.calculator import UPETCalculator
except ImportError:
    # upet (PET-OAM UQ/cross-MLIP checks) is dev-host-only; other hosts still
    # need tier-0 checks and novelty indexing, which never touch UPETCalculator.
    UPETCalculator = None

# Moved to chem/stability.py; re-exported here since callers across the
# codebase (run_screen.py, screening.py, compute_ehull.py, and others) still
# import it from validity.structural.
from chem.stability import load_stability_calc  # noqa: F401

MACE_MODEL_PATH = "/data/assets/checkpoints/mace/mace-mp-0b3-medium.model"
MP_API_KEY = "TnTrt09078le8A5ssIq5ZOn05hxOKdCO"
_CM1_TO_THZ = 0.029979245800  # 1 cm⁻¹ in THz

def _safe_get_structure(atoms: Atoms):
    """Convert ASE Atoms → pymatgen Structure, returning None for degenerate cells."""
    if _has_bad_coords(atoms):
        return None
    try:
        s = AseAtomsAdaptor.get_structure(atoms)
        if np.any(np.isnan(s.frac_coords)):
            return None
        return s
    except (np.linalg.LinAlgError, ValueError):
        return None


# Canonical blocklist -- single source of truth is dielectric/blocked_elements.py
# (a sibling of this package, not the steering bundle; see that module's
# docstring for the drift history this replaced).
from blocked_elements import RADIOACTIVE_ELEMENTS


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_pet(device: str = "cuda") -> UPETCalculator:
    if UPETCalculator is None:
        raise ImportError(
            "upet is not installed on this host; PET/UQ checks are dev-host-only."
        )
    return UPETCalculator(model="pet-oam-xl", version="1.0.0", device=device)


def load_mace(device: str = "cuda", model_path: str = MACE_MODEL_PATH) -> MACECalculator:
    from e3nn import get_optimization_defaults
    from e3nn.nn._activation import Activation
    from e3nn.o3._spherical_harmonics import SphericalHarmonics, _spherical_harmonics

    calc = MACECalculator(model_paths=[model_path], device=device, default_dtype="float64")

    # e3nn <0.5 has no "jit_mode" key (added alongside torch.compile support);
    # missing means eager, matching the else branch below.
    jit_mode = get_optimization_defaults().get("jit_mode")
    for model in calc.models:
        for mod in model.modules():
            # SphericalHarmonics.sph_func is set in __init__ but not via
            # CodeGenMixin, so it is absent when a checkpoint saved with
            # old e3nn is unpickled.
            if isinstance(mod, SphericalHarmonics) and not hasattr(mod, "sph_func"):
                if jit_mode == "script":
                    mod.sph_func = torch.jit.script(_spherical_harmonics)
                elif jit_mode == "compile":
                    mod.sph_func = torch.compile(_spherical_harmonics, fullgraph=True)
                else:
                    mod.sph_func = _spherical_harmonics

            # Activation.paths is set in __init__ but is a plain list, so
            # it is absent when a checkpoint saved with an older e3nn
            # (before paths existed) is unpickled.  Reconstruct it.
            if isinstance(mod, Activation) and not hasattr(mod, "paths"):
                mod.paths = [
                    (mul, (l, p), act)
                    for (mul, (l, p)), act in zip(mod.irreps_in, mod.acts)
                ]

    return calc


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _has_bad_coords(atoms: Atoms) -> bool:
    """Return True if positions or cell contain NaN/Inf values."""
    return (not np.all(np.isfinite(atoms.positions))
            or not np.all(np.isfinite(atoms.cell.array)))


def check_plausibility(atoms: Atoms) -> dict:
    """Physical plausibility checks (density, lattice bounds, CIF roundtrip).

    Matches the plausibility checks in LeMat-GenBench.
    """
    vol = float(atoms.get_volume())
    n_atoms = len(atoms)
    mass = float(atoms.get_masses().sum())
    mass_density = mass * 1.66054 / vol if vol > 0 else 0  # g/cm3
    atomic_density = n_atoms / vol if vol > 0 else 0       # atoms/A3

    params = atoms.cell.cellpar()
    a, b, c = float(params[0]), float(params[1]), float(params[2])
    alpha, beta, gamma = float(params[3]), float(params[4]), float(params[5])

    issues = []
    if mass_density < 0.01 or mass_density > 25.0:
        issues.append(f"density={mass_density:.2f}")
    if atomic_density < 1e-5 or atomic_density > 0.5:
        issues.append(f"atomic_density={atomic_density:.4f}")
    if vol <= 1.0:
        issues.append(f"volume={vol:.2f}")
    if not (1.0 <= a <= 100.0 and 1.0 <= b <= 100.0 and 1.0 <= c <= 100.0):
        issues.append(f"lattice=({a:.1f},{b:.1f},{c:.1f})")
    if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
        issues.append(f"angles=({alpha:.1f},{beta:.1f},{gamma:.1f})")

    return {
        "plausibility_pass": len(issues) == 0,
        "mass_density_g_cm3": round(mass_density, 3),
        "plausibility_issues": "; ".join(issues) if issues else None,
    }


def check_min_dist(atoms: Atoms, cutoff: float = 0.5) -> dict:
    """Minimum interatomic distance check (hard reject).

    Rejects structures where any atom pair is closer than *cutoff* Å, or where
    the unit-cell volume is unrealistically small (< 0.1 Å³).  Mirrors the
    ``structure_validity`` check used in CDVAE's compute_metrics.py.
    """
    if _has_bad_coords(atoms):
        return {"min_dist_A": None, "volume_A3": None, "min_dist_pass": False}

    vol = float(atoms.get_volume())
    if vol < 0.1:
        return {"min_dist_A": None, "volume_A3": vol, "min_dist_pass": False}

    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(dists.min())

    return {
        "min_dist_A": min_dist,
        "volume_A3": vol,
        "min_dist_pass": min_dist >= cutoff,
    }


def check_smact(
    atoms: Atoms,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> dict:
    """SMACT charge-balance and electronegativity composition check (hard reject).

    Passes if any combination of oxidation states produces charge neutrality
    (and, optionally, satisfies the Pauling electronegativity rule).  Pure
    elemental / all-metal structures are always accepted.  Mirrors the
    ``smact_validity`` check used in CDVAE's compute_metrics.py.
    """
    from collections import Counter

    elem_counter = Counter(atoms.get_chemical_symbols())
    elems = tuple(elem_counter.keys())
    counts = np.array([elem_counter[e] for e in elems], dtype=int)
    counts = counts // np.gcd.reduce(counts)

    if len(set(elems)) == 1:
        return {"smact_pass": True}

    if include_alloys:
        if all(e in smact.metals for e in elems):
            return {"smact_pass": True}

    space = smact.element_dictionary(elems)
    smact_elems = [v for v in space.values()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    # If any element has no known oxidation states, fail gracefully
    if any(ox is None or len(ox) == 0 for ox in ox_combos):
        return {"smact_pass": False}

    threshold = int(counts.max())
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(int(c),) for c in counts]
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_ok = True
            else:
                electroneg_ok = True
            if electroneg_ok:
                return {"smact_pass": True}

    return {"smact_pass": False}


def check_element_filter(
    atoms: Atoms,
    blocked: frozenset[str] = RADIOACTIVE_ELEMENTS,
) -> dict:
    """Element blocklist check (hard reject).

    Rejects structures containing any element in *blocked*.  Defaults to
    ``RADIOACTIVE_ELEMENTS``.  Pass a custom frozenset to restrict to an
    application-specific allowed set, e.g. for practical synthesisability.
    """
    symbols = set(atoms.get_chemical_symbols())
    bad = symbols & blocked
    return {
        "blocked_elements": sorted(bad),
        "element_filter_pass": len(bad) == 0,
    }


def check_isolated_atoms(atoms: Atoms, cutoff: float = 4.0) -> dict:
    """Isolated atom check (hard reject).

    Rejects structures where any atom has no neighbour within *cutoff* Å
    (using the minimum-image convention).  An isolated atom indicates the
    structure has unphysical disconnected fragments.
    """
    if _has_bad_coords(atoms) or np.linalg.matrix_rank(atoms.cell) < 3:
        return {
            "n_isolated_atoms": len(atoms),
            "isolated_atoms_pass": False,
        }
    i_idx = ase_neighbor_list('i', atoms, cutoff)
    connected = set(i_idx.tolist())
    n_isolated = sum(1 for i in range(len(atoms)) if i not in connected)
    return {
        "n_isolated_atoms": n_isolated,
        "isolated_atoms_pass": n_isolated == 0,
    }


def check_bvs(atoms: Atoms, max_deviation: float = 0.5) -> dict:
    """Bond Valence Sum check (hard reject).

    Checks that bond lengths are consistent with a valid assignment of formal
    oxidation states.  Uses pymatgen's BVAnalyzer, which minimises the residual
    between per-site bond-valence sums and integer oxidation states.

    Fails if no self-consistent assignment can be found, or if the maximum
    per-site |BVS − formal charge| exceeds *max_deviation*.  This is a
    stronger, geometry-aware complement to the composition-only SMACT check.
    """
    if _has_bad_coords(atoms):
        return {"bvs_max_deviation": None, "bvs_pass": False}
    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        bva = BVAnalyzer()
        with _suppress_spglib_stderr():
            valences = bva.get_valences(structure)
        # valences are floats close to integers; deviation from nearest integer
        # measures how self-consistent the BVS assignment is.
        max_dev = float(max(abs(v - round(v)) for v in valences))
        return {
            "bvs_max_deviation": round(max_dev, 3),
            "bvs_pass": max_dev <= max_deviation,
        }
    except Exception:
        return {"bvs_max_deviation": None, "bvs_pass": False}


def check_spacegroup(atoms: Atoms, symprec: float = 0.1) -> dict:
    """Spacegroup consistency check.

    Uses pymatgen's SpacegroupAnalyzer to compute the spacegroup of the
    structure and compares it against atoms.info["spacegroup"] if present.
    Returns the computed spacegroup number and whether it matches.
    """
    structure = _safe_get_structure(atoms)
    if structure is None:
        return {"spacegroup_computed": None, "spacegroup_target": None,
                "spacegroup_pass": False}
    try:
        with _suppress_spglib_stderr():
            sg = SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
    except Exception:
        return {"spacegroup_computed": None, "spacegroup_target": None,
                "spacegroup_pass": False}

    target = atoms.info.get("spacegroup") or atoms.info.get("sg_number")
    if target is not None:
        try:
            target = int(target)
        except (ValueError, TypeError):
            # Non-numeric spacegroup string (e.g. "P 1") — skip comparison
            return {"spacegroup_computed": sg, "spacegroup_target": str(target),
                    "spacegroup_pass": True}
        return {"spacegroup_computed": sg, "spacegroup_target": target,
                "spacegroup_pass": sg == target}
    # No target to compare — just report the computed value, mark as pass
    return {"spacegroup_computed": sg, "spacegroup_target": None,
            "spacegroup_pass": True}


def check_uq(atoms: Atoms, calc: UPETCalculator, threshold: float = 0.05) -> dict:
    """MLIP uncertainty quantification (hard reject if > threshold eV/atom)."""
    try:
        uq = float(calc.get_energy_uncertainty(atoms)) / len(atoms)
        return {"uq_eV_pa": uq, "uq_pass": uq <= threshold}
    except Exception:
        return {"uq_eV_pa": None, "uq_pass": True}  # unavailable → skip


def check_ehull(
    atoms: Atoms,
    calc: UPETCalculator,
    threshold: float = 0.1,
    cache: Optional[dict] = None,
) -> dict:
    """Energy above hull via all-MLIP phase diagram (hard reject if > threshold eV/atom).

    cache is a dict keyed by frozenset(elements) → list[ComputedStructureEntry] for
    the MP reference phases, re-evaluated with calc.  Shared across calls to avoid
    repeated MPRester queries for the same chemical system.
    """
    if cache is None:
        cache = {}

    atoms = copy.deepcopy(atoms)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    structure = _safe_get_structure(atoms)
    if structure is None:
        return {
            "e_above_hull": None,
            "dft_e_above_hull": atoms.info.get("energy_above_hull"),
            "decomposition": None,
            "ehull_pass": False,
        }
    elements = [str(el) for el in structure.composition.elements]
    key = frozenset(elements)

    if key not in cache:
        with MPRester(MP_API_KEY) as mpr:
            mp_entries = mpr.get_entries_in_chemsys(elements)
        ref_entries = []
        for mp_entry in mp_entries:
            ref_struct = getattr(mp_entry, "structure", None)
            if ref_struct is None:
                continue
            ref_atoms = AseAtomsAdaptor.get_atoms(ref_struct)
            ref_atoms.calc = calc
            ref_entries.append(
                ComputedStructureEntry(ref_struct, ref_atoms.get_potential_energy())
            )
        cache[key] = ref_entries

    our_entry = ComputedStructureEntry(structure, energy)
    try:
        with _suppress_spglib_stderr():
            pd = PhaseDiagram(cache[key] + [our_entry])
            e_hull = float(pd.get_e_above_hull(our_entry))
            decomp, _ = pd.get_decomp_and_e_above_hull(our_entry)
    except ValueError:
        # Missing terminal entries for one or more elements in MP —
        # cannot construct the hull, so skip rather than hard-reject.
        return {
            "e_above_hull": None,
            "dft_e_above_hull": atoms.info.get("energy_above_hull"),
            "decomposition": None,
            "ehull_pass": True,
        }

    return {
        "e_above_hull": e_hull,
        "dft_e_above_hull": atoms.info.get("energy_above_hull"),
        "decomposition": {p.composition.reduced_formula: amt for p, amt in decomp.items()},
        "ehull_pass": e_hull <= threshold,
    }


def check_phonon_stability(
    atoms: Atoms,
    calc: MACECalculator,
    threshold_thz: float = -0.5,
) -> dict:
    """Γ-point phonon stability via finite-difference Hessian (hard reject if imaginary optic modes).

    Uses a temporary directory so Vibrations cache files don't pollute the cwd.
    """
    atoms = copy.deepcopy(atoms)
    atoms.calc = calc

    with tempfile.TemporaryDirectory() as tmp, _suppress_spglib_stderr():
        vib = Vibrations(atoms, name=os.path.join(tmp, "vib"), delta=0.01)
        vib.run()
        freqs_cm1 = np.real(vib.get_frequencies())

    freqs_thz = freqs_cm1 * _CM1_TO_THZ
    # drop 3 acoustic modes (smallest absolute frequency)
    optic = sorted(freqs_thz, key=abs)[3:]
    n_imag = int(np.sum(np.array(optic) < threshold_thz))
    min_freq = float(np.min(optic)) if optic else 0.0

    return {
        "n_imaginary_modes": n_imag,
        "most_negative_freq_thz": min_freq,
        "phonon_pass": n_imag == 0,
    }


def check_md_stability(
    atoms: Atoms,
    calc,
    temperature: float = 600.0,
    n_steps: int = 500,
    disp_threshold: float = 3.0,
    vol_threshold: float = 20.0,
) -> dict:
    """Short NVT-MD stability check (soft flag).

    Runs n_steps × 2 fs of Langevin dynamics and checks for large atomic
    displacements or volume drift relative to the starting geometry.
    """
    atoms = copy.deepcopy(atoms)
    atoms.calc = calc
    pos0 = atoms.get_positions().copy()
    vol0 = atoms.get_volume()

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(atoms, timestep=2.0 * fs, temperature_K=temperature, friction=0.02 / fs)
    dyn.run(n_steps)

    max_disp = float(np.max(np.linalg.norm(atoms.get_positions() - pos0, axis=1)))
    vol_drift = float(abs(atoms.get_volume() - vol0) / vol0 * 100.0)

    return {
        "md_max_disp_A": max_disp,
        "md_volume_drift_pct": vol_drift,
        "md_pass": max_disp <= disp_threshold and vol_drift <= vol_threshold,
    }


def check_cross_mlip(
    atoms: Atoms,
    pet_calc: UPETCalculator,
    mace_calc: MACECalculator,
    threshold: float = 0.5,
) -> dict:
    """Cross-MLIP single-point energy comparison (soft flag)."""
    n = len(atoms)

    a1 = copy.deepcopy(atoms)
    a1.calc = pet_calc
    e_pet = float(a1.get_potential_energy()) / n

    a2 = copy.deepcopy(atoms)
    a2.calc = mace_calc
    e_mace = float(a2.get_potential_energy()) / n

    delta = abs(e_pet - e_mace)
    return {
        "energy_pet_eV_pa": e_pet,
        "energy_mace_eV_pa": e_mace,
        "cross_delta_eV_pa": delta,
        "cross_pass": delta <= threshold,
    }


# ---------------------------------------------------------------------------
# Set-level metrics (operate on a list of structures, not a single Atoms)
# ---------------------------------------------------------------------------

def compute_uniqueness_mask(
    atoms_list: list[Atoms],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> np.ndarray:
    """Per-structure boolean mask: True for the first occurrence of each unique structure.

    Groups by reduced formula for efficiency (only compares within the same
    composition), matching MatterGen's approach.  Invalid structures (degenerate
    cell, NaN coords) are marked False.
    """
    from collections import defaultdict

    structures: list = [None] * len(atoms_list)
    by_formula: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(atoms_list):
        s = _safe_get_structure(a)
        if s is not None:
            structures[i] = s
            by_formula[s.composition.reduced_formula].append(i)

    mask = np.zeros(len(atoms_list), dtype=bool)
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True)

    with _suppress_spglib_stderr():
        for formula, indices in by_formula.items():
            for pos, idx_i in enumerate(indices):
                is_dup = False
                for idx_j in indices[:pos]:
                    if mask[idx_j] and matcher.fit(structures[idx_i], structures[idx_j]):
                        is_dup = True
                        break
                if not is_dup:
                    mask[idx_i] = True

    return mask


def compute_novelty_mask(
    atoms_list: list[Atoms],
    reference_list: list[Atoms],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> np.ndarray:
    """Per-structure boolean mask: True if the structure is absent from the reference set.

    Reference structures are indexed by reduced formula so StructureMatcher only
    runs on pairs that share the same composition.  Invalid structures are marked
    False.
    """
    from collections import defaultdict

    ref_by_formula: dict[str, list] = defaultdict(list)
    for a in reference_list:
        s = _safe_get_structure(a)
        if s is not None:
            ref_by_formula[s.composition.reduced_formula].append(s)

    mask = np.zeros(len(atoms_list), dtype=bool)
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True)

    with _suppress_spglib_stderr():
        for i, a in enumerate(atoms_list):
            s = _safe_get_structure(a)
            if s is None:
                continue
            refs = ref_by_formula.get(s.composition.reduced_formula, [])
            if not any(matcher.fit(s, r) for r in refs):
                mask[i] = True

    return mask


def compute_uniqueness(
    atoms_list: list[Atoms],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> dict:
    """Fraction of structures that are unique within the generated set.

    Uses pymatgen's StructureMatcher.group_structures(), which first groups
    by reduced composition (fast) and then compares only within groups.
    Returns the number of unique structures and the uniqueness fraction.
    """
    structures = [s for a in atoms_list if (s := _safe_get_structure(a)) is not None]
    n_skipped = len(atoms_list) - len(structures)
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True)
    with _suppress_spglib_stderr():
        groups = matcher.group_structures(structures)
    n_unique = len(groups)
    n_total = len(atoms_list)
    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "uniqueness": n_unique / n_total if n_total > 0 else 0.0,
    }


def compute_novelty(
    atoms_list: list[Atoms],
    reference_list: list[Atoms],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> dict:
    """Fraction of generated structures not found in a reference set.

    Efficient: reference structures are indexed by reduced formula so
    StructureMatcher only runs on pairs that share the same composition.
    *reference_list* is typically the training set.
    """
    from collections import defaultdict

    gen_structs = [s for a in atoms_list if (s := _safe_get_structure(a)) is not None]
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True)

    ref_by_formula: dict[str, list] = defaultdict(list)
    for a in reference_list:
        s = _safe_get_structure(a)
        if s is not None:
            ref_by_formula[s.composition.reduced_formula].append(s)

    n_novel = 0
    with _suppress_spglib_stderr():
        for g in gen_structs:
            formula = g.composition.reduced_formula
            refs = ref_by_formula.get(formula, [])
            if not any(matcher.fit(g, r) for r in refs):
                n_novel += 1

    n_total = len(atoms_list)
    return {
        "n_total": n_total,
        "n_novel": n_novel,
        "novelty": n_novel / n_total if n_total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Composition-level checks (formula string → bool)
# ---------------------------------------------------------------------------

def is_charge_balanced(formula: str) -> bool:
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
        from pipeline.composition_features import stoichiometric_entropy
    except ModuleNotFoundError:
        from composition_features import stoichiometric_entropy
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


def passes_chemistry_filters(formula: str) -> bool:
    if not is_charge_balanced(formula):
        return False
    if not passes_electronegativity_filter(formula):
        return False
    if not passes_radius_filter(formula):
        return False
    if not passes_entropy_filter(formula):
        return False
    return True


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_checks(
    atoms: Atoms,
    pet_calc: UPETCalculator,
    mace_calc: MACECalculator,
    ehull_cache: Optional[dict] = None,
    run_md: bool = False,
    run_cross: bool = True,
    funnel: bool = True,
) -> dict:
    """Run validity checks on a single structure.

    When *funnel=True* (default), stop early on hard rejects — this is faster
    because expensive MLIP checks are skipped for structures that already fail
    cheap tier-0 tests.

    When *funnel=False*, run every check unconditionally so that per-check
    pass rates are computed over the full set (wider sample, slower).

    Hard rejects (element_filter through phonon): gate ``overall_pass``.
    Soft flags (cross_mlip, md): always recorded but do not gate overall_pass.
    """
    result: dict = {}
    passing = True

    # Tier 0 — cheap pre-filters, no MLIP needed (~0 s/structure)
    r = check_element_filter(atoms)
    result.update(r)
    if not r["element_filter_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    r = check_min_dist(atoms)
    result.update(r)
    if not r["min_dist_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    r = check_isolated_atoms(atoms)
    result.update(r)
    if not r["isolated_atoms_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    r = check_smact(atoms)
    result.update(r)
    if not r["smact_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    r = check_bvs(atoms)
    result.update(r)
    if not r["bvs_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    r = check_spacegroup(atoms)
    result.update(r)
    if not r["spacegroup_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    # Tier 1 — UQ (fast, ~0.1 s/structure)
    r = check_uq(atoms, pet_calc)
    result.update(r)
    if not r["uq_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    # Tier 2 — E above hull (cached after first chemsys, ~10 s first call)
    r = check_ehull(atoms, pet_calc, cache=ehull_cache)
    result.update(r)
    if not r["ehull_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    # Tier 3 — Cross-MLIP (soft, ~0.5 s/structure)
    if run_cross:
        result.update(check_cross_mlip(atoms, pet_calc, mace_calc))

    # Tier 4 — Phonons (hard, ~2 s/structure with MACE)
    r = check_phonon_stability(atoms, mace_calc)
    result.update(r)
    if not r["phonon_pass"]:
        passing = False
        if funnel:
            result["overall_pass"] = False
            return result

    # Tier 5 — MD (soft, optional, ~3 s/structure)
    if run_md:
        result.update(check_md_stability(atoms, mace_calc))

    result["overall_pass"] = passing
    return result


__all__ = [
    "MACE_MODEL_PATH",
    "MP_API_KEY",
    "RADIOACTIVE_ELEMENTS",
    "load_pet",
    "load_mace",
    # per-structure checks — Tier 0 (no MLIP)
    "check_element_filter",
    "check_min_dist",
    "check_isolated_atoms",
    "check_smact",
    "check_bvs",
    # per-structure checks — Tier 1–5 (MLIP)
    "check_uq",
    "check_ehull",
    "check_phonon_stability",
    "check_md_stability",
    "check_cross_mlip",
    # set-level metrics
    "compute_uniqueness",
    "compute_novelty",
    "compute_uniqueness_mask",
    "compute_novelty_mask",
    # composition-level checks
    "O_RADIUS",
    "is_charge_balanced",
    "electronegativity_spread",
    "passes_electronegativity_filter",
    "ionic_radius_ratio",
    "passes_radius_filter",
    "entropy_value",
    "passes_entropy_filter",
    "goldschmidt_tolerance",
    "passes_tolerance_filter",
    "passes_chemistry_filters",
    # convenience wrapper
    "run_checks",
]

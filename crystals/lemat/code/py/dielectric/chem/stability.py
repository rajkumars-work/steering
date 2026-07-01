"""Energy-above-hull (e_hull) for ASE Atoms via MLIP-evaluated phase diagram.

Hull baseline: MP entries fetched once per chemical system, evaluated with the
same MLIP, and cached on disk per chemsys. Switching MLIPs requires deleting
the cache directory — entries from a different potential give wrong hull
energies.

Public API:
    load_stability_calc(...)          -> ASE-compatible MLIP calculator
    compute_e_above_hull(atoms, ...)  -> dict with e_above_hull and pass flag
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
import pickle
from typing import Optional

import numpy as np
import torch
from ase import Atoms

from chem._timeout import CheckTimeout, run_with_timeout
from chem.validity import _has_bad_coords, _safe_get_structure, _suppress_spglib_stderr

# ---------------------------------------------------------------------------
# Compatibility shim: e3nn >=0.5.7 changed CodeGenMixin.__setstate__ format.
# Older MACE checkpoints stored TorchScript bytes directly; new e3nn expects
# (buffer_type, buffer). Patch once at import so torch.load deserialises both.
# ---------------------------------------------------------------------------
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


DEFAULT_MACE_PATH = "/data/assets/checkpoints/mace/mace-mp-0b3-medium.model"
DEFAULT_EHULL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".ehull_cache")
DEFAULT_MP_API_KEY = "TnTrt09078le8A5ssIq5ZOn05hxOKdCO"

# Cap reference entries per chemsys to bound first-call latency
_MAX_REF_ENTRIES = 150


def load_mace(device: str = "cuda", mlip_path: str = DEFAULT_MACE_PATH):
    """Load a MACE calculator with the e3nn compatibility patch applied."""
    from mace.calculators import MACECalculator

    import chem.props.mace_compat  # noqa: F401  (apply __setstate__ patch)
    from chem.props.mace_compat import patch_mace_model

    calc = MACECalculator(model_paths=[mlip_path], device=device, default_dtype="float64")
    patch_mace_model(calc)
    return calc


def load_stability_calc(device: str = "cuda", mlip_path: str = DEFAULT_MACE_PATH):
    """Load the MLIP calculator used for stability / e_hull (alias for load_mace)."""
    return load_mace(device=device, mlip_path=mlip_path)


def _ehull_cache_path(cache_dir: str, elements: frozenset) -> str:
    return os.path.join(cache_dir, f"{'-'.join(sorted(elements))}.pkl")


def _load_disk_cache(cache_dir: str, elements: frozenset) -> Optional[list]:
    path = _ehull_cache_path(cache_dir, elements)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_disk_cache(cache_dir: str, elements: frozenset, entries: list) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(_ehull_cache_path(cache_dir, elements), "wb") as f:
            pickle.dump(entries, f)
    except Exception:
        pass


class StabilityError(RuntimeError):
    """Raised when e_above_hull cannot be computed due to an INFRASTRUCTURE failure
    (missing dependency, MP fetch/auth/network error) — as opposed to a structure that
    is legitimately un-scorable. Surfaced loudly so a broken reference fetch can never
    silently zero out SUN/MSUN/Combined (as happened when emmet-core was missing)."""


def _fetch_and_evaluate_refs(elements: list, calc, mp_api_key: str) -> list:
    """Pull MP entries for *elements* and re-evaluate energies with *calc*."""
    import requests
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.ext.matproj import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    with MPRester(mp_api_key) as mpr:
        for attr in ("session", "_session"):
            sess = getattr(mpr, attr, None)
            if sess is not None:
                sess.timeout = 30
        mp_entries = mpr.get_entries_in_chemsys(elements)

    if len(mp_entries) > _MAX_REF_ENTRIES:
        mp_entries = mp_entries[:_MAX_REF_ENTRIES]

    ref_entries = []
    for mp_entry in mp_entries:
        ref_struct = getattr(mp_entry, "structure", None)
        if ref_struct is None:
            continue
        ref_atoms = AseAtomsAdaptor.get_atoms(ref_struct)
        ref_atoms.calc = calc
        ref_entries.append(
            ComputedStructureEntry(ref_struct, ref_atoms.get_potential_energy()))
    return ref_entries


def _compute_e_above_hull_impl(
    atoms: Atoms,
    calc,
    cache_dir: str,
    mem_cache: dict,
    threshold: float,
    mp_api_key: str,
    skip_on_fetch_error: bool = False,
) -> dict:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedStructureEntry

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

    if key not in mem_cache:
        disk = _load_disk_cache(cache_dir, key)
        if disk is not None:
            mem_cache[key] = disk
        else:
            try:
                mem_cache[key] = _fetch_and_evaluate_refs(elements, calc, mp_api_key)
            except (ImportError, ModuleNotFoundError) as e:
                # Environment/dependency failure (e.g. missing emmet-core). This breaks
                # EVERY hull build, so never swallow it — surface loudly and immediately,
                # regardless of skip_on_fetch_error.
                raise StabilityError(
                    f"stability dependency missing ({e}) — cannot deserialize MP "
                    f"reference entries; install the missing package"
                ) from e
            except Exception as e:
                # MP-side / network / auth failure. Raise by default so a broken
                # reference fetch can't silently zero out Combined; callers that want
                # best-effort scoring can pass skip_on_fetch_error=True.
                if not skip_on_fetch_error:
                    raise StabilityError(
                        f"MP reference fetch failed for chemsys {sorted(elements)}: {e}"
                    ) from e
                return {
                    "e_above_hull": None,
                    "dft_e_above_hull": atoms.info.get("energy_above_hull"),
                    "decomposition": None,
                    "ehull_pass": False,
                    "ehull_error": f"MP fetch failed: {e}",
                }
            _save_disk_cache(cache_dir, key, mem_cache[key])

    our_entry = ComputedStructureEntry(structure, energy)
    try:
        with _suppress_spglib_stderr():
            pd = PhaseDiagram(mem_cache[key] + [our_entry])
            e_hull = float(pd.get_e_above_hull(our_entry))
            decomp, _ = pd.get_decomp_and_e_above_hull(our_entry)
    except ValueError as e:
        # Legitimately un-scorable: novel chemsys whose MP terminal entries are missing,
        # so no hull can be constructed. This is a real scientific case (common for the
        # novel-chemistry pools), NOT an infrastructure bug — so skip rather than raise,
        # but ehull_pass=False (a structure we couldn't evaluate never reads as "passed").
        return {
            "e_above_hull": None,
            "dft_e_above_hull": atoms.info.get("energy_above_hull"),
            "decomposition": None,
            "ehull_pass": False,
            "ehull_error": f"no hull (missing terminal entries): {e}",
        }

    return {
        "e_above_hull": e_hull,
        "dft_e_above_hull": atoms.info.get("energy_above_hull"),
        "decomposition": {p.composition.reduced_formula: amt for p, amt in decomp.items()},
        "ehull_pass": e_hull <= threshold,
    }


def compute_e_above_hull(
    atoms: Atoms,
    calc=None,
    mlip_path: str = DEFAULT_MACE_PATH,
    cache_dir: str = DEFAULT_EHULL_CACHE_DIR,
    mem_cache: Optional[dict] = None,
    threshold: float = 0.1,
    mp_api_key: str = DEFAULT_MP_API_KEY,
    timeout: float = 30,
    skip_on_fetch_error: bool = False,
) -> dict:
    """E_above_hull via MLIP-evaluated phase diagram.

    Args:
        atoms: ASE structure to evaluate.
        calc: pre-loaded ASE calculator. If None, ``load_stability_calc(mlip_path=...)``
            is called once. Passing in a shared calc avoids per-call MLIP load.
        mlip_path: MACE checkpoint, used only when ``calc`` is None.
        cache_dir: per-chemsys reference-entry pickle cache.
        mem_cache: optional in-process dict to share across calls.
        threshold: ehull_pass threshold in eV/atom.
        mp_api_key: Materials Project API key.
        timeout: hard SIGALRM wall-clock budget. On timeout returns
            ``{"e_above_hull": None, "ehull_pass": False, "ehull_timed_out": True}``.
        skip_on_fetch_error: if False (default) an MP reference-fetch failure raises
            ``StabilityError`` rather than silently returning None — so a broken fetch
            can't quietly zero out Combined. Set True for best-effort batch scoring.
            Dependency errors (e.g. missing emmet-core) ALWAYS raise regardless.

    Returns dict with keys ``e_above_hull``, ``dft_e_above_hull``,
    ``decomposition``, ``ehull_pass``. Raises ``StabilityError`` on infrastructure
    failures (see ``skip_on_fetch_error``).
    """
    if calc is None:
        calc = load_stability_calc(mlip_path=mlip_path)
    if mem_cache is None:
        mem_cache = {}
    try:
        return run_with_timeout(
            _compute_e_above_hull_impl, atoms, calc, cache_dir, mem_cache,
            threshold, mp_api_key, skip_on_fetch_error, timeout=timeout)
    except CheckTimeout:
        return {
            "e_above_hull": None,
            "dft_e_above_hull": atoms.info.get("energy_above_hull"),
            "decomposition": None,
            "ehull_pass": False,
            "ehull_timed_out": True,
        }


# Backward-compatible alias for callers using the old (atoms, calc) signature
def check_ehull(
    atoms: Atoms,
    calc,
    threshold: float = 0.1,
    cache: Optional[dict] = None,
) -> dict:
    """Legacy positional wrapper. Prefer ``compute_e_above_hull``."""
    return compute_e_above_hull(
        atoms, calc=calc, mem_cache=cache, threshold=threshold)

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import smact
import spglib
from ase import Atoms
from ase.data import chemical_symbols
from ase.geometry import cellpar_to_cell
from ase.neighborlist import neighbor_list as ase_neighbor_list
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact.screening import pauling_test

from .formatter import FieldFormatter
from .reader import parse_target
from .versions import (
    S_ANON_FORMULA_ELEMENTS,
    S_CRYSTAL_SYS_SG,
    S_DENSITY,
    S_EREF,
    S_NATOMS,
    S_PROP_LABELS,
    T_ANON_ELEMENTS,
    T_ATOMS,
    T_CN,
    T_EF,
    T_FORMULA,
    T_LATTICE,
    T_NN,
    T_OX,
    T_SCRATCHPAD,
    T_SG_INFO,
    T_SG_INFO_SYS,
    T_WP,
    get_version,
)


NOBLE_ELEMENTS: frozenset[str] = frozenset(["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"])
RADIOACTIVE_ELEMENTS: frozenset[str] = frozenset([
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
])


@dataclass
class ValidationConfig:
    max_natoms: int = 256
    min_dist_cutoff: float = 0.5
    isolated_cutoff: float = 5.0
    run_smact: bool = False
    run_bvs: bool = False
    run_spacegroup: bool = False
    run_site_consistency: bool = False
    fail_on_smact: bool = False
    fail_on_noble_elements: bool = True
    fail_on_radioactive_elements: bool = True
    fail_on_spacegroup_mismatch: bool = False
    fail_on_site_consistency: bool = False
    fail_on_bvs: bool = False
    tokenizer_model: Optional[str] = None
    max_source_tokens: Optional[int] = None
    max_target_tokens: Optional[int] = None


@dataclass
class ValidationResult:
    row_id: str
    origin: str
    passed: bool
    failures: list[str]
    warnings: list[str]
    details: dict[str, object]

    def primary_flag(self) -> str:
        return self.failures[0] if self.failures else ""


def _split_segments(text: str) -> list[str]:
    return [part.strip() for part in text.split("|")]


def _split_flag_tokens(flag: str) -> list[str]:
    return [token.strip() for token in flag.split(";") if token.strip()]


def _field_map(version_id: str, origin: str, source: bool) -> dict[str, str]:
    spec = get_version(version_id)
    fields = spec.source_segments if source else spec.target_segments
    values = fields.get(origin, fields["mp"])
    return {name: segments_idx for segments_idx, name in enumerate(values)}


def _segments_to_named_map(text: str, version_id: str, origin: str, source: bool) -> tuple[dict[str, str], bool]:
    spec = get_version(version_id)
    fields = spec.source_segments if source else spec.target_segments
    expected = fields.get(origin, fields["mp"])
    segments = _split_segments(text)
    if len(segments) != len(expected):
        return {}, False
    return {field: segments[i] for i, field in enumerate(expected)}, True


def _parse_anon_elements(segment: str) -> tuple[Optional[str], list[str]]:
    if not segment:
        return None, []
    parts = segment.split()
    if not parts:
        return None, []
    anon_formula = parts[0]
    elements = parts[1:]
    return anon_formula, elements


def _parse_int_field(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except Exception:
        return None


def _parse_sg_number(text: str) -> Optional[int]:
    parts = text.split()
    for token in parts:
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _parse_prefixed_numeric_list(text: str, prefix: str, cast=float) -> list[float]:
    if not text or not text.startswith(prefix):
        return []
    payload = text[len(prefix):]
    if not payload:
        return []
    vals = []
    for token in payload.split(";"):
        if not token:
            continue
        try:
            vals.append(cast(token))
        except Exception:
            return []
    return vals


def _parse_wp_counts(text: str) -> list[int]:
    if not text or not text.startswith("WP:"):
        return []
    counts = []
    for token in text[3:].split(";"):
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            return []
        try:
            counts.append(int(parts[1]))
        except Exception:
            return []
    return counts


def _parse_scratchpad(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in text.split():
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        out[key] = value
    return out


def _valid_symbols(symbols: list[str]) -> bool:
    known = set(chemical_symbols[1:])
    return all(sym in known for sym in symbols)


def _safe_get_structure(atoms: Atoms):
    try:
        return AseAtomsAdaptor.get_structure(atoms)
    except Exception:
        return None


def _check_min_dist(atoms: Atoms, cutoff: float) -> tuple[bool, Optional[float]]:
    if not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(atoms.cell.array)):
        return False, None
    try:
        vol = float(atoms.get_volume())
    except Exception:
        return False, None
    if vol < 0.1:
        return False, None
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(dists.min())
    return min_dist >= cutoff, min_dist


def _check_isolated_atoms(atoms: Atoms, cutoff: float) -> tuple[bool, int]:
    if np.linalg.matrix_rank(atoms.cell.array) < 3:
        return False, len(atoms)
    i_idx = ase_neighbor_list("i", atoms, cutoff)
    connected = set(i_idx.tolist())
    n_isolated = sum(1 for i in range(len(atoms)) if i not in connected)
    return n_isolated == 0, n_isolated


def _check_smact(atoms: Atoms) -> bool:
    elem_counter = Counter(atoms.get_chemical_symbols())
    elems = tuple(elem_counter.keys())
    if len(set(elems)) == 1:
        return True
    if all(e in smact.metals for e in elems):
        return True
    counts = np.array([elem_counter[e] for e in elems], dtype=int)
    counts = counts // np.gcd.reduce(counts)
    space = smact.element_dictionary(elems)
    smact_elems = [v for v in space.values()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if any(ox is None or len(ox) == 0 for ox in ox_combos):
        return False
    threshold = int(counts.max())
    for ox_states in __import__("itertools").product(*ox_combos):
        stoichs = [(int(c),) for c in counts]
        cn_e, _ = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            try:
                if pauling_test(ox_states, electronegs):
                    return True
            except TypeError:
                return True
    return False


def _check_bvs(atoms: Atoms, max_deviation: float = 0.5) -> tuple[bool, Optional[float]]:
    structure = _safe_get_structure(atoms)
    if structure is None:
        return False, None
    try:
        valences = BVAnalyzer().get_valences(structure)
    except Exception:
        return False, None
    max_dev = float(max(abs(v - round(v)) for v in valences))
    return max_dev <= max_deviation, max_dev


def _check_spacegroup(atoms: Atoms, target_sg: Optional[int], symprec: float = 0.1) -> tuple[bool, Optional[int]]:
    structure = _safe_get_structure(atoms)
    if structure is None:
        return False, None
    try:
        computed = int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        return False, None
    if target_sg is None:
        return True, computed
    return computed == target_sg, computed


def _tokenizer(model_path: Optional[str]):
    if not model_path:
        return None
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


class DatasetRowValidator:
    def __init__(self, version_id: str, config: Optional[ValidationConfig] = None):
        self.version_id = version_id
        self.config = config or ValidationConfig()
        self.formatter = FieldFormatter()
        self.sp = _tokenizer(self.config.tokenizer_model)

    def validate_row(self, row_id: str, origin: str, source: str, target: str) -> ValidationResult:
        failures: list[str] = []
        warnings: list[str] = []
        details: dict[str, object] = {}

        source_map, source_ok = _segments_to_named_map(source, self.version_id, origin, source=True)
        target_map, target_ok = _segments_to_named_map(target, self.version_id, origin, source=False)

        if not source_ok:
            failures.append("source_segments")
        if not target_ok:
            failures.append("target_segments")

        atoms = parse_target(target, self.version_id, origin)
        if atoms is None:
            failures.append("target_parse")
            return ValidationResult(row_id, origin, False, failures, warnings, details)

        symbols = atoms.get_chemical_symbols()
        symbol_set = set(symbols)
        details["natoms"] = len(atoms)
        details["elements"] = sorted(symbol_set)

        if not _valid_symbols(symbols):
            failures.append("unknown_elements")

        radioactive = sorted(symbol_set & RADIOACTIVE_ELEMENTS)
        noble = sorted(symbol_set & NOBLE_ELEMENTS)
        if radioactive:
            details["radioactive_elements"] = radioactive
            self._record_issue("radioactive_elements", failures, warnings, self.config.fail_on_radioactive_elements)
        if noble:
            details["noble_elements"] = noble
            self._record_issue("noble_elements", failures, warnings, self.config.fail_on_noble_elements)

        if len(atoms) > self.config.max_natoms:
            failures.append(f"large_structure:{len(atoms)}")

        min_dist_ok, min_dist = _check_min_dist(atoms, self.config.min_dist_cutoff)
        details["min_dist_A"] = min_dist
        if not min_dist_ok:
            failures.append("min_dist")

        isolated_ok, n_isolated = _check_isolated_atoms(atoms, self.config.isolated_cutoff)
        details["n_isolated_atoms"] = n_isolated
        if not isolated_ok:
            failures.append("isolated_atoms")

        if self.config.run_smact and not _check_smact(atoms):
            self._record_issue("smact", failures, warnings, self.config.fail_on_smact)

        if self.config.run_bvs:
            bvs_ok, bvs_dev = _check_bvs(atoms)
            details["bvs_max_deviation"] = bvs_dev
            if not bvs_ok:
                self._record_issue("bvs", failures, warnings, self.config.fail_on_bvs)

        target_formula = target_map.get(T_FORMULA) if target_map else None
        if target_formula:
            try:
                if Composition(target_formula).reduced_formula != Composition(atoms.get_chemical_formula()).reduced_formula:
                    failures.append("formula_mismatch")
            except Exception:
                failures.append("formula_mismatch")

        if source_map:
            natoms_source = source_map.get(S_NATOMS)
            if natoms_source:
                parsed = _parse_int_field(natoms_source)
                if parsed != len(atoms):
                    failures.append("source_natoms_mismatch")

            anon_field = source_map.get(S_ANON_FORMULA_ELEMENTS)
            if anon_field:
                _, source_elements = _parse_anon_elements(anon_field)
                if source_elements and set(source_elements) != symbol_set:
                    failures.append("source_elements_mismatch")

        if target_map:
            anon_field = target_map.get(T_ANON_ELEMENTS)
            if anon_field:
                _, target_elements = _parse_anon_elements(anon_field)
                if target_elements and set(target_elements) != symbol_set:
                    failures.append("target_elements_mismatch")

            target_sg = None
            sg_field = target_map.get(T_SG_INFO) or target_map.get(T_SG_INFO_SYS)
            if sg_field:
                target_sg = _parse_sg_number(sg_field)

            if self.config.run_spacegroup:
                sg_ok, computed_sg = _check_spacegroup(atoms, target_sg)
                details["computed_sg"] = computed_sg
                if not sg_ok:
                    self._record_issue("spacegroup_mismatch", failures, warnings, self.config.fail_on_spacegroup_mismatch)

            if self.config.run_site_consistency:
                self._check_site_consistency(atoms, target_map, failures, warnings, details)

        if self.sp is not None:
            src_tokens = len(self.sp.encode(source, out_type=int))
            tgt_tokens = len(self.sp.encode(target, out_type=int))
            details["source_tokens"] = src_tokens
            details["target_tokens"] = tgt_tokens
            if self.config.max_source_tokens is not None and src_tokens > self.config.max_source_tokens:
                failures.append(f"source_tokens:{src_tokens}")
            if self.config.max_target_tokens is not None and tgt_tokens > self.config.max_target_tokens:
                failures.append(f"target_tokens:{tgt_tokens}")

        return ValidationResult(row_id, origin, not failures, failures, warnings, details)

    def _record_issue(self, issue: str, failures: list[str], warnings: list[str], fail: bool) -> None:
        if fail:
            failures.append(issue)
        else:
            warnings.append(issue)

    def _check_site_consistency(
        self,
        atoms: Atoms,
        target_map: dict[str, str],
        failures: list[str],
        warnings: list[str],
        details: dict[str, object],
    ) -> None:
        wyckoff = atoms.info.get("wyckoff_letters")
        if not wyckoff:
            self._record_issue("wyckoff_missing", failures, warnings, self.config.fail_on_site_consistency)
            return

        site_info = self.formatter.get_scratchpad_fields(atoms, wyckoff, include_ox=T_OX in target_map or T_SCRATCHPAD in target_map)
        wp_counts = _parse_wp_counts(target_map.get(T_WP, ""))
        if not wp_counts and T_SCRATCHPAD in target_map:
            scratch = _parse_scratchpad(target_map[T_SCRATCHPAD])
            wp_counts = _parse_wp_counts(f"WP:{scratch['WP']}") if "WP" in scratch else []
            cn_vals = _parse_prefixed_numeric_list(f"CN:{scratch['CN']}", "CN:", cast=float) if "CN" in scratch else []
            nn_vals = _parse_prefixed_numeric_list(f"NN:{scratch['NN']}", "NN:", cast=float) if "NN" in scratch else []
        else:
            cn_vals = _parse_prefixed_numeric_list(target_map.get(T_CN, ""), "CN:", cast=float)
            nn_vals = _parse_prefixed_numeric_list(target_map.get(T_NN, ""), "NN:", cast=float)

        if wp_counts and sum(wp_counts) != len(atoms):
            self._record_issue("wp_sum_mismatch", failures, warnings, self.config.fail_on_site_consistency)

        expected_wp = _parse_wp_counts(f"WP:{site_info.get('WP', '')}") if site_info.get("WP") else []
        expected_cn = _parse_prefixed_numeric_list(f"CN:{site_info.get('CN', '')}", "CN:", cast=float) if site_info.get("CN") else []
        expected_nn = _parse_prefixed_numeric_list(f"NN:{site_info.get('NN', '')}", "NN:", cast=float) if site_info.get("NN") else []

        if wp_counts and expected_wp and sorted(wp_counts) != sorted(expected_wp):
            self._record_issue("wp_mismatch", failures, warnings, self.config.fail_on_site_consistency)
        if cn_vals and expected_cn and len(cn_vals) != len(expected_cn):
            self._record_issue("cn_length_mismatch", failures, warnings, self.config.fail_on_site_consistency)
        if nn_vals and expected_nn and len(nn_vals) != len(expected_nn):
            self._record_issue("nn_length_mismatch", failures, warnings, self.config.fail_on_site_consistency)


def validate_dataset(
    csv_path: str,
    version_id: str,
    config: Optional[ValidationConfig] = None,
    max_rows: Optional[int] = None,
) -> tuple[list[ValidationResult], Counter]:
    validator = DatasetRowValidator(version_id, config=config)
    results: list[ValidationResult] = []
    summary = Counter()
    seen_ids: set[str] = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            row_id = row.get("id", f"row:{idx}")
            if row_id in seen_ids:
                result = ValidationResult(row_id, row.get("origin", ""), False, ["duplicate_id"], [], {})
            else:
                seen_ids.add(row_id)
                result = validator.validate_row(row_id, row.get("origin", ""), row["source"], row["target"])
            results.append(result)
            if result.passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
                for failure in result.failures:
                    summary[failure] += 1
            for warning in result.warnings:
                summary[f"warning:{warning}"] += 1
    summary["rows"] = len(results)
    return results, summary


def upsert_master_flags(
    master_ids_path: str,
    flag_updates: dict[str, tuple[str, str]],
    overwrite_existing: bool = False,
) -> None:
    from dielectric_data.master_version import read_master, rewrite

    rows = read_master(master_ids_path)

    by_id = {row["id"]: row for row in rows}
    for row_id, (origin, flag) in flag_updates.items():
        if row_id in by_id:
            existing_flag = (by_id[row_id].get("flag") or "").strip()
            if overwrite_existing or not existing_flag:
                by_id[row_id]["flag"] = flag
            elif flag not in _split_flag_tokens(existing_flag):
                by_id[row_id]["flag"] = ";".join([flag, existing_flag])
        else:
            rows.append({
                "id": row_id,
                "origin": origin,
                "label": "",
                "flag": flag,
            })
            by_id[row_id] = rows[-1]

    rewrite(
        master_ids_path,
        rows,
        reason=f"upsert_master_flags: {len(flag_updates)} flag updates",
        rows_modified=len(flag_updates),
    )


__all__ = [
    "DatasetRowValidator",
    "NOBLE_ELEMENTS",
    "RADIOACTIVE_ELEMENTS",
    "ValidationConfig",
    "ValidationResult",
    "upsert_master_flags",
    "validate_dataset",
]

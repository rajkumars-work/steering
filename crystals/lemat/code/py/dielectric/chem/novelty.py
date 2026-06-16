"""Lemat-novelty: StructureMatcher comparison against the LeMat-Bulk dataset.

Uses composition-first filtering then pairwise StructureMatcher comparison,
matching LeMat-GenBench's methodology.

Two backends, both memory-bounded:

* **SQLite** (preferred when ``composition_index.sqlite`` is present): per-call
  formula→row-indices lookup against a small on-disk index. Structures are
  fetched on demand via PyArrow row-group reads, so peak RAM stays well under
  100 MB regardless of dataset size.

* **Pandas** (fallback): a pickled composition→indices index loaded once into
  memory; per-formula structures are materialised lazily from the parquet
  DataFrame. Needs the full DataFrame in RAM (~15 GB on LeMat-Bulk) — only use
  this when SQLite is unavailable and you have plenty of memory.

Public API:
    LematNoveltyChecker.from_assets(...)     -> reusable checker
    check_lemat_novelty(atoms_or_list, ...)  -> bool | list[bool]  (singleton)
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from chem._timeout import CheckTimeout, run_with_timeout

DEFAULT_LEMAT_DIR = "/data/assets/datasets/lematbulk"
DEFAULT_LEMAT_PARQUET = f"{DEFAULT_LEMAT_DIR}/lematbulk_compatible_pbe.parquet"
DEFAULT_LEMAT_SQLITE = f"{DEFAULT_LEMAT_DIR}/composition_index.sqlite"

# Cap on per-call structural comparisons (keeps wall-clock and RAM bounded
# when a popular formula has thousands of polymorphs in the reference set).
DEFAULT_MAX_CANDIDATES = 50


# ---------------------------------------------------------------------------
# SQLite + PyArrow backend (memory-bounded, default)
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    """Disk-only state: SQLite for formula lookup, PyArrow for per-row reads."""

    def __init__(self, parquet_path: str, sqlite_path: str):
        self.parquet_path = parquet_path
        self.sqlite_path = sqlite_path
        self._pq_file = None
        self._rg_offsets: list[tuple[int, int, int]] = []

    def _open_parquet(self):
        if self._pq_file is None:
            import pyarrow.parquet as pq
            self._pq_file = pq.ParquetFile(self.parquet_path)
            cum = 0
            for i in range(self._pq_file.num_row_groups):
                n = self._pq_file.metadata.row_group(i).num_rows
                self._rg_offsets.append((cum, cum + n, i))
                cum += n
        return self._pq_file

    def lookup_indices(self, reduced_formula: str) -> list[int]:
        with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT indices_json FROM composition_index WHERE reduced_formula = ?",
                (reduced_formula,),
            ).fetchone()
        return json.loads(row[0]) if row else []

    def iter_structures(self, indices: list[int]):
        """Stream Structures one row-group at a time so the caller can break
        out (e.g. on time budget) without forcing a full read."""
        if not indices:
            return
        f = self._open_parquet()
        # Maintain insertion order across row-groups: walk row-groups in the
        # order their first occurrence appears in *indices*.
        rg_order: list[int] = []
        rg_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pos, gi in enumerate(indices):
            for start, end, rg_id in self._rg_offsets:
                if start <= gi < end:
                    if rg_id not in rg_pairs:
                        rg_order.append(rg_id)
                    rg_pairs[rg_id].append((pos, gi - start))
                    break
        cols = ["lattice_vectors", "species_at_sites", "cartesian_site_positions"]
        for rg_id in rg_order:
            tbl = f.read_row_group(rg_id, columns=cols).to_pylist()
            for _pos, local_idx in rg_pairs[rg_id]:
                if not (0 <= local_idx < len(tbl)):
                    continue
                row = tbl[local_idx]
                try:
                    yield Structure(
                        Lattice(np.array(row["lattice_vectors"])),
                        list(row["species_at_sites"]),
                        np.array(row["cartesian_site_positions"]),
                        coords_are_cartesian=True,
                    )
                except Exception:
                    continue


# ---------------------------------------------------------------------------
# Pandas fallback (in-memory df, only when SQLite unavailable)
# ---------------------------------------------------------------------------

class _PandasBackend:
    """Composition-indexed pandas DataFrame, ~15 GB peak on LeMat-Bulk."""

    def __init__(self, parquet_path: str, ltol: float = 0.1, cache_dir: Optional[str] = None):
        self.parquet_path = parquet_path
        if cache_dir is None:
            cache_dir = os.path.dirname(parquet_path)
        self.cache_path = os.path.join(cache_dir, f".novelty_comp_index_tol{ltol}.pkl")
        self.composition_index: dict[str, list[int]] = {}
        self._df = None
        self._struct_cache: dict[str, list[Structure]] = {}
        self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.composition_index = data["composition_index"]
            return

        import pandas as pd
        df = pd.read_parquet(self.parquet_path)
        index: dict[str, list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            formula = row.get("chemical_formula_descriptive", "")
            if (row.get("lattice_vectors") is None
                    or row.get("species_at_sites") is None
                    or row.get("cartesian_site_positions") is None):
                continue
            try:
                reduced = Composition(formula).reduced_formula
            except Exception:
                reduced = formula
            index[reduced].append(idx)
        self.composition_index = dict(index)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"composition_index": self.composition_index}, f)
        self._df = df

    def lookup_indices(self, reduced_formula: str) -> list[int]:
        return self.composition_index.get(reduced_formula, [])

    def iter_structures(self, indices: list[int]):
        if not indices:
            return
        if self._df is None:
            import pandas as pd
            self._df = pd.read_parquet(self.parquet_path)
        for idx in indices:
            try:
                row = self._df.iloc[idx]
                yield Structure(
                    Lattice(np.array([np.array(v) for v in row["lattice_vectors"]])),
                    list(row["species_at_sites"]),
                    np.array([np.array(v) for v in row["cartesian_site_positions"]]),
                    coords_are_cartesian=True,
                )
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Unified checker
# ---------------------------------------------------------------------------

class LematNoveltyChecker:
    """Memory-bounded StructureMatcher novelty against LeMat-Bulk."""

    def __init__(self, backend, matcher: StructureMatcher,
                 max_candidates: int = DEFAULT_MAX_CANDIDATES,
                 time_budget_s: Optional[float] = None):
        self.backend = backend
        self.matcher = matcher
        self.max_candidates = max_candidates
        # Per-call wall-clock cap on the structural-match loop. Defaults to
        # None (use only max_candidates). Pass a number to bound wall-clock
        # independently of N.
        self.time_budget_s = time_budget_s

    @classmethod
    def from_assets(
        cls,
        parquet_path: str = DEFAULT_LEMAT_PARQUET,
        sqlite_path: Optional[str] = DEFAULT_LEMAT_SQLITE,
        ltol: float = 0.1,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        time_budget_s: Optional[float] = None,
    ) -> "LematNoveltyChecker":
        """Build a checker. Uses SQLite if available, else falls back to pandas.

        Args:
            parquet_path: LeMat-Bulk parquet (always required).
            sqlite_path: composition_index.sqlite. Pass None to force pandas.
            ltol: StructureMatcher length tolerance (matches GenBench at 0.1).
            max_candidates: cap on structural comparisons per call.
            time_budget_s: optional wall-clock cap on the match loop.
        """
        matcher = StructureMatcher(ltol=ltol)
        if sqlite_path and Path(sqlite_path).exists():
            backend = _SQLiteBackend(parquet_path, sqlite_path)
        else:
            backend = _PandasBackend(parquet_path, ltol=ltol)
        return cls(backend, matcher, max_candidates=max_candidates,
                   time_budget_s=time_budget_s)

    # Back-compat
    @classmethod
    def from_parquet(cls, parquet_path: str = DEFAULT_LEMAT_PARQUET,
                     ltol: float = 0.1, cache_dir: Optional[str] = None) -> "LematNoveltyChecker":
        return cls.from_assets(parquet_path=parquet_path,
                               sqlite_path=DEFAULT_LEMAT_SQLITE, ltol=ltol)

    def is_novel(self, atoms: Atoms) -> bool:
        """True if no StructureMatcher hit found in the reference set.

        Iterates candidates in their stored order, stopping early on first
        match. Per-call work is bounded by ``max_candidates`` and (if set)
        ``time_budget_s``. If neither limit is hit and no match is found,
        the structure is genuinely novel for this dataset.
        """
        try:
            struct = AseAtomsAdaptor.get_structure(atoms)
            reduced = struct.composition.reduced_formula
        except Exception:
            return True

        indices = self.backend.lookup_indices(reduced)
        if not indices:
            return True
        if len(indices) > self.max_candidates:
            indices = indices[:self.max_candidates]

        deadline = (time.monotonic() + self.time_budget_s) if self.time_budget_s else None
        for ref in self.backend.iter_structures(indices):
            if deadline is not None and time.monotonic() > deadline:
                return True  # ran out of time — fail-open (novelty bias)
            try:
                if self.matcher.fit(struct, ref):
                    return False
            except Exception:
                continue
        return True

    def check_batch(self, atoms_list: list, verbose: bool = False) -> list[bool]:
        results = []
        n_novel = 0
        t0 = time.time()
        for i, atoms in enumerate(atoms_list):
            novel = self.is_novel(atoms)
            results.append(novel)
            n_novel += int(novel)
            if verbose and (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(atoms_list)}] novel={n_novel}/{i+1} "
                      f"({100*n_novel/(i+1):.0f}%) ({elapsed:.0f}s)", flush=True)
        return results


# ---------------------------------------------------------------------------
# Singleton convenience
# ---------------------------------------------------------------------------

_lematbulk_checker: Optional[LematNoveltyChecker] = None


def get_lematbulk_checker(
    parquet_path: str = DEFAULT_LEMAT_PARQUET,
    sqlite_path: Optional[str] = DEFAULT_LEMAT_SQLITE,
    ltol: float = 0.1,
) -> LematNoveltyChecker:
    """Process-wide singleton checker for the default LeMat-Bulk assets."""
    global _lematbulk_checker
    if _lematbulk_checker is None:
        _lematbulk_checker = LematNoveltyChecker.from_assets(
            parquet_path=parquet_path, sqlite_path=sqlite_path, ltol=ltol)
    return _lematbulk_checker


def check_lemat_novelty(
    atoms_or_list: Union[Atoms, list],
    parquet_path: str = DEFAULT_LEMAT_PARQUET,
    sqlite_path: Optional[str] = DEFAULT_LEMAT_SQLITE,
    ltol: float = 0.1,
    timeout: float = 30,
) -> Union[bool, list]:
    """Lemat-novelty for one Atoms or a list. Fail-open on timeout (returns True)."""
    checker = get_lematbulk_checker(parquet_path=parquet_path,
                                    sqlite_path=sqlite_path, ltol=ltol)
    if isinstance(atoms_or_list, Atoms):
        try:
            return run_with_timeout(checker.is_novel, atoms_or_list, timeout=timeout)
        except CheckTimeout:
            return True
    return checker.check_batch(atoms_or_list)

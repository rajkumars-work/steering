"""
pipeline.py
===========

Stage-based pipeline for composition discovery.

Stage 1:
  - Pick two compositions from each cluster with randomness:
      * one far from cluster center (distance_score-weighted)
      * one with high bandgap + eps_0 (z-score-weighted)

Stage 2:
  - For each selected composition, call expand_many(return_pareto=True)
  - Save expanded Pareto compositions with properties to CSV

Stage 3:
  - Take stage-2 output, compute Pareto front ranks
  - Keep only rank-0 (front) and rank-1 compositions

Stage 6:
  - Relax stage-5 structures with MLIP
  - Compute dielectric-related properties using MLIP surrogates/phonons

This file is designed to be extended with more stages later.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ase import Atoms as ASEAtoms
from ase.data import chemical_symbols
from ase.io import write as ase_write

from chem.gen.expander import expand_many
from chem.pareto_front import ParetoFront
from pipeline.filters import (
    filter_candidates_by_training_species,
    filter_rows_by_chemistry,
    stage3_filter,
    stage5_filter_energy,
)
from chem.surrogates.soap import load_structures_from_extxyz, predict_properties
from chem.gen.structures import (
    generate_structures_for_compositions_topk,
    load_dataset,
)


@dataclass(frozen=True)
class SeedSelection:
    cluster: str
    cluster_label: str
    role: str  # "center" or "high"
    formula: str
    bandgap: Optional[float]
    eps_0: Optional[float]


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN check
        return None
    return v


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clusters file not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _write_csv(path: str, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cluster_name(cluster: str, cluster_label: str) -> str:
    label = (cluster_label or "").strip()
    if label:
        return label
    return f"cluster_{cluster}"


def stage1_select_seeds(
    cluster_rows: List[Dict[str, str]], seed: Optional[int] = None
) -> List[SeedSelection]:
    required_cols = {"formula", "cluster"}
    missing = [c for c in required_cols if c not in cluster_rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in cluster CSV: {missing}")

    rng = np.random.default_rng(seed)

    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in cluster_rows:
        groups.setdefault(str(row.get("cluster", "")), []).append(row)

    seeds: List[SeedSelection] = []
    for cluster_id, rows in sorted(groups.items(), key=lambda x: x[0]):
        if not rows:
            continue

        def distance_score(r: Dict[str, str]) -> float:
            v = _to_float(r.get("distance_score"))
            return v if v is not None else 0.0

        def _weighted_choice(
            rows_local: List[Dict[str, str]], weights: np.ndarray
        ) -> Dict[str, str]:
            weights = np.asarray(weights, dtype=float)
            weights[~np.isfinite(weights)] = 0.0
            total = float(np.sum(weights))
            if total <= 0.0:
                idx = rng.integers(0, len(rows_local))
                return rows_local[int(idx)]
            probs = weights / total
            idx = rng.choice(len(rows_local), p=probs)
            return rows_local[int(idx)]

        # Far-from-center selection: weighted by distance_score.
        dist_weights = np.array([distance_score(r) for r in rows], dtype=float)
        center_row = _weighted_choice(rows, dist_weights)

        # High bandgap + eps_0 selection: weighted by z-score combination.
        bandgaps = np.array(
            [_to_float(r.get("dft_band_gap")) for r in rows], dtype=float
        )
        eps0s = np.array(
            [_to_float(r.get("dft_eps_0")) for r in rows], dtype=float
        )
        valid = np.isfinite(bandgaps) & np.isfinite(eps0s)
        bg_mean = float(np.mean(bandgaps[valid])) if np.any(valid) else 0.0
        eps_mean = float(np.mean(eps0s[valid])) if np.any(valid) else 0.0
        bg_std = float(np.std(bandgaps[valid])) if np.any(valid) else 0.0
        eps_std = float(np.std(eps0s[valid])) if np.any(valid) else 0.0
        if bg_std <= 0.0:
            bg_std = 1.0
        if eps_std <= 0.0:
            eps_std = 1.0
        z_bg = (bandgaps - bg_mean) / bg_std
        z_eps = (eps0s - eps_mean) / eps_std
        combined = z_bg + z_eps
        combined[~valid] = float("-inf")
        if np.all(~np.isfinite(combined)):
            prop_weights = np.ones(len(rows), dtype=float)
        else:
            max_c = float(np.nanmax(combined))
            prop_weights = np.exp(combined - max_c)

        # Avoid picking the same row twice when possible.
        if len(rows) > 1:
            try:
                center_idx = rows.index(center_row)
            except ValueError:
                center_idx = -1
            if 0 <= center_idx < len(rows):
                prop_weights[center_idx] = 0.0

        high_row = _weighted_choice(rows, prop_weights)

        cluster_label = str(center_row.get("cluster_label", ""))

        seeds.append(
            SeedSelection(
                cluster=cluster_id,
                cluster_label=cluster_label,
                role="center",
                formula=str(center_row.get("formula", "")),
                bandgap=_to_float(center_row.get("dft_band_gap")),
                eps_0=_to_float(center_row.get("dft_eps_0")),
            )
        )
        seeds.append(
            SeedSelection(
                cluster=cluster_id,
                cluster_label=cluster_label,
                role="high",
                formula=str(high_row.get("formula", "")),
                bandgap=_to_float(high_row.get("dft_band_gap")),
                eps_0=_to_float(high_row.get("dft_eps_0")),
            )
        )

    return seeds


def stage2_expand(seeds: List[SeedSelection], per_seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        if not seed.formula:
            continue

        pareto = expand_many([seed.formula], per_seed=per_seed, return_pareto=True)
        cluster_name = _cluster_name(seed.cluster, seed.cluster_label)

        for formula, props in pareto.items():
            bandgap, eps_0 = props
            rows.append(
                {
                    "cluster": seed.cluster,
                    "cluster_name": cluster_name,
                    "seed_role": seed.role,
                    "seed_formula": seed.formula,
                    "composition_formula": formula,
                    "bandgap": bandgap,
                    "eps_0": eps_0,
                }
            )

    return rows


def _pmg_to_ase(structure) -> ASEAtoms:
    return ASEAtoms(
        symbols=[str(site.specie.symbol) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )


def _percentile_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([1.0])
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return (ranks - 1) / (len(values) - 1)


def _z_scores(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _generate_raw_candidates(
    stage3_rows: List[Dict[str, object]],
    extxyz_path: str,
    per_comp: int,
    n_jobs: Optional[int],
    use_gpu: bool,
) -> Tuple[Dict[str, Dict[str, object]], List[List[Dict[str, object]]]]:
    comp_rows = filter_rows_by_chemistry(stage3_rows)
    targets = list(comp_rows.keys())
    dataset = load_dataset(extxyz_path)
    pmg_structures = [pmg for _, pmg in dataset]

    topk = generate_structures_for_compositions_topk(
        targets,
        pmg_structures,
        k=per_comp,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
    )
    return comp_rows, topk


def _build_candidate_lists(
    comp_rows: Dict[str, Dict[str, object]],
    topk: List[List[Dict[str, object]]],
) -> Tuple[List[ASEAtoms], List[Dict[str, object]], Dict[str, List[int]]]:
    targets = list(comp_rows.keys())
    candidate_meta: List[Dict[str, object]] = []
    candidate_atoms: List[ASEAtoms] = []
    comp_to_indices: Dict[str, List[int]] = {}

    for i, formula in enumerate(targets):
        candidates = topk[i]
        for cand in candidates:
            struct = cand["structure"]
            if struct is None:
                continue
            atoms = _pmg_to_ase(struct)
            idx = len(candidate_atoms)
            candidate_atoms.append(atoms)
            candidate_meta.append(
                {
                    "composition_formula": formula,
                    "prototype_formula": cand["prototype_formula"],
                    "prototype_distance": cand["distance"],
                    "structure": struct,
                    "source_row": comp_rows[formula],
                    "candidate_index": idx,
                }
            )
            comp_to_indices.setdefault(formula, []).append(idx)

    return candidate_atoms, candidate_meta, comp_to_indices


def _filter_candidates(
    candidate_atoms: List[ASEAtoms],
    candidate_meta: List[Dict[str, object]],
    comp_to_indices: Dict[str, List[int]],
    extxyz_path: str,
) -> Tuple[List[ASEAtoms], List[Dict[str, object]], Dict[str, List[int]]]:
    training_structures = load_structures_from_extxyz(extxyz_path)
    training_species = {
        int(Z) for atoms in training_structures for Z in atoms.get_atomic_numbers()
    }

    candidate_atoms, candidate_meta, comp_to_indices, unknown_species = (
        filter_candidates_by_training_species(
            candidate_atoms,
            candidate_meta,
            comp_to_indices,
            training_species,
        )
    )

    if unknown_species:
        symbols = [
            chemical_symbols[z] if z < len(chemical_symbols) else str(z)
            for z in sorted(unknown_species)
        ]
        print(
            "Skipping candidates with elements not in SOAP training set: "
            f"{sorted(unknown_species)} ({', '.join(symbols)})"
        )

    return candidate_atoms, candidate_meta, comp_to_indices


def _predict_and_attach_properties(
    candidate_atoms: List[ASEAtoms],
    candidate_meta: List[Dict[str, object]],
    extxyz_path: str,
    n_jobs: Optional[int],
) -> None:
    predictions = predict_properties(
        candidate_atoms,
        model_names=("dft_band_gap", "dft_eps_0"),
        extxyz_path=extxyz_path,
        n_jobs=n_jobs,
    )

    bandgap_pred = predictions["dft_band_gap"]
    eps0_pred = predictions["dft_eps_0"]

    for meta in candidate_meta:
        idx = int(meta["candidate_index"])
        meta["pred_bandgap"] = float(bandgap_pred[idx])
        meta["pred_eps_0"] = float(eps0_pred[idx])


def _select_candidates_for_comp(
    cand_list: List[Dict[str, object]],
    keep_per_comp: int,
) -> List[Dict[str, object]]:
    ranker = ParetoFront(maximize_bandgap=True, maximize_eps_0=True)
    for c in cand_list:
        ranker.add(
            str(c["candidate_index"]),
            float(c["pred_bandgap"]),
            float(c["pred_eps_0"]),
        )

    front = ranker.pareto()
    rank1 = ranker.pareto(1)

    pool = list(front.keys())
    if len(pool) < keep_per_comp:
        pool.extend([k for k in rank1.keys() if k not in pool])

    bg_vals = np.array([c["pred_bandgap"] for c in cand_list], dtype=float)
    eps_vals = np.array([c["pred_eps_0"] for c in cand_list], dtype=float)
    bg_pct = _percentile_scores(bg_vals)
    eps_pct = _percentile_scores(eps_vals)
    bg_z = _z_scores(bg_vals)
    eps_z = _z_scores(eps_vals)

    # Re-map index to position in local list for scoring
    pos_map = {int(c["candidate_index"]): pos for pos, c in enumerate(cand_list)}

    scores = {}
    for c in cand_list:
        idx = int(c["candidate_index"])
        pos = pos_map[idx]
        pct_score = 0.5 * (bg_pct[pos] + eps_pct[pos])
        z_score = bg_z[pos] + eps_z[pos]
        scores[str(idx)] = (pct_score, z_score)

    def sort_key(c):
        idx = str(c["candidate_index"])
        _, z_score = scores[idx]
        return z_score

    pool_candidates = [c for c in cand_list if str(c["candidate_index"]) in pool]
    pool_candidates.sort(key=sort_key, reverse=True)
    return pool_candidates[:keep_per_comp]


def _process_selections(
    comp_to_indices: Dict[str, List[int]],
    candidate_meta: List[Dict[str, object]],
    candidate_atoms: List[ASEAtoms],
    keep_per_comp: int,
    output_dir: str,
) -> Tuple[List[Dict[str, object]], List[ASEAtoms]]:
    os.makedirs(output_dir, exist_ok=True)
    selected_rows: List[Dict[str, object]] = []
    selected_atoms: List[ASEAtoms] = []

    for formula, idxs in comp_to_indices.items():
        cand_list = [candidate_meta[i] for i in idxs]
        if not cand_list:
            continue

        selected = _select_candidates_for_comp(cand_list, keep_per_comp)

        for rank, c in enumerate(selected, start=1):
            atoms = candidate_atoms[int(c["candidate_index"])]
            output_file = os.path.join(
                output_dir,
                f"{formula}_s{rank}.extxyz",
            )
            ase_write(output_file, atoms)

            src = c["source_row"]
            row_out = {
                "cluster": src.get("cluster"),
                "cluster_name": src.get("cluster_name"),
                "seed_role": src.get("seed_role"),
                "seed_formula": src.get("seed_formula"),
                "composition_formula": formula,
                "pareto_rank": src.get("pareto_rank"),
                "structure_rank": rank,
                "structure_path": output_file,
                "prototype_formula": c.get("prototype_formula"),
                "prototype_distance": c.get("prototype_distance"),
                "pred_bandgap": c.get("pred_bandgap"),
                "pred_eps_0": c.get("pred_eps_0"),
            }
            selected_rows.append(row_out)
            selected_atoms.append(atoms)

    return selected_rows, selected_atoms


def stage4_generate_structures(
    stage3_rows: List[Dict[str, object]],
    extxyz_path: str,
    per_comp: int = 10,
    keep_per_comp: int = 2,
    output_dir: str = "pipeline_stage4_structures",
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
) -> Tuple[List[Dict[str, object]], List[ASEAtoms]]:
    if not stage3_rows:
        return [], []

    comp_rows, topk = _generate_raw_candidates(
        stage3_rows, extxyz_path, per_comp, n_jobs, use_gpu
    )
    candidate_atoms, candidate_meta, comp_to_indices = _build_candidate_lists(
        comp_rows, topk
    )

    if not candidate_atoms:
        return [], []

    candidate_atoms, candidate_meta, comp_to_indices = _filter_candidates(
        candidate_atoms, candidate_meta, comp_to_indices, extxyz_path
    )

    if not candidate_atoms:
        return [], []

    _predict_and_attach_properties(candidate_atoms, candidate_meta, extxyz_path, n_jobs)

    return _process_selections(
        comp_to_indices, candidate_meta, candidate_atoms, keep_per_comp, output_dir
    )


def run_pipeline(args: argparse.Namespace) -> None:
    cluster_rows = _read_csv_rows(args.clusters)
    if not cluster_rows:
        raise ValueError(f"No rows found in cluster CSV: {args.clusters}")

    # Stage 1: select seed compositions.
    seeds = stage1_select_seeds(
        cluster_rows, seed=args.seed_selection_seed
    )
    _write_csv(
        args.stage1_out,
        [
            {
                "cluster": s.cluster,
                "cluster_label": s.cluster_label,
                "role": s.role,
                "formula": s.formula,
                "bandgap": s.bandgap,
                "eps_0": s.eps_0,
            }
            for s in seeds
        ],
        ["cluster", "cluster_label", "role", "formula", "bandgap", "eps_0"],
    )

    # Stage 2: expand seeds and collect Pareto compositions.
    expanded = stage2_expand(seeds, per_seed=args.per_seed)
    _write_csv(
        args.stage2_out,
        expanded,
        [
            "cluster",
            "cluster_name",
            "seed_role",
            "seed_formula",
            "composition_formula",
            "bandgap",
            "eps_0",
        ],
    )

    # Stage 3: keep Pareto front + rank-1.
    stage3 = stage3_filter(expanded)
    _write_csv(
        args.stage3_out,
        stage3,
        [
            "cluster",
            "cluster_name",
            "seed_role",
            "seed_formula",
            "composition_formula",
            "bandgap",
            "eps_0",
            "pareto_rank",
        ],
    )

    # Stage 4: generate prototype-matched structures and select top candidates.
    stage4_rows, stage4_atoms = stage4_generate_structures(
        stage3,
        extxyz_path=args.extxyz_path,
        per_comp=args.per_comp_structures,
        keep_per_comp=args.keep_per_comp,
        output_dir=args.stage4_dir,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
    )
    _write_csv(
        args.stage4_out,
        stage4_rows,
        [
            "cluster",
            "cluster_name",
            "seed_role",
            "seed_formula",
            "composition_formula",
            "pareto_rank",
            "structure_rank",
            "structure_path",
            "prototype_formula",
            "prototype_distance",
            "pred_bandgap",
            "pred_eps_0",
        ],
    )

    # Stage 5: filter by predicted energy above hull.
    stage5_rows = stage5_filter_energy(
        stage4_rows,
        stage4_atoms,
        extxyz_path=args.extxyz_path,
        max_energy_per_atom=args.energy_threshold,
        n_jobs=args.n_jobs,
    )
    stage5_fields = [
        "cluster",
        "cluster_name",
        "seed_role",
        "seed_formula",
        "composition_formula",
        "pareto_rank",
        "structure_rank",
        "structure_path",
        "prototype_formula",
        "prototype_distance",
        "pred_bandgap",
        "pred_eps_0",
        "pred_energy_above_hull",
        "pred_energy_above_hull_per_atom",
        "n_atoms",
    ]
    _write_csv(
        args.stage5_out,
        stage5_rows,
        stage5_fields,
    )

    print(f"Stage 1 seeds: {len(seeds)} written to {args.stage1_out}")
    print(f"Stage 2 expanded: {len(expanded)} written to {args.stage2_out}")
    print(f"Stage 3 filtered: {len(stage3)} written to {args.stage3_out}")
    print(f"Stage 4 structures: {len(stage4_rows)} written to {args.stage4_out}")
    print(f"Stage 5 filtered: {len(stage5_rows)} written to {args.stage5_out}")

    # Stage 6 disabled (removed from pipeline execution).


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-based composition pipeline")
    parser.add_argument(
        "--clusters",
        default=os.path.join("cluster_out", "topq_clusters.csv"),
        help="Path to cluster CSV (from cluster_topq.py)",
    )
    parser.add_argument(
        "--per-seed",
        type=int,
        default=30,
        help="Number of expanded candidates per seed formula",
    )
    parser.add_argument(
        "--stage1-out",
        default="pipeline_stage1_seeds.csv",
        help="Output CSV for stage-1 seed selections",
    )
    parser.add_argument(
        "--stage2-out",
        default="pipeline_stage2_expanded.csv",
        help="Output CSV for stage-2 expanded Pareto compositions",
    )
    parser.add_argument(
        "--stage3-out",
        default="pipeline_stage3_frontier.csv",
        help="Output CSV for stage-3 Pareto front + rank-1 compositions",
    )
    parser.add_argument(
        "--extxyz-path",
        default="/data/assets/datasets/dielectric/mp/*.extxyz",
        help="Path to dataset (glob pattern) for prototypes + SOAP species",
    )
    parser.add_argument(
        "--per-comp-structures",
        type=int,
        default=10,
        help="Number of structures to generate per composition",
    )
    parser.add_argument(
        "--keep-per-comp",
        type=int,
        default=2,
        help="Number of structures to keep per composition after Pareto selection",
    )
    parser.add_argument(
        "--stage4-out",
        default="pipeline_stage4_structures.csv",
        help="Output CSV for stage-4 structure selection",
    )
    parser.add_argument(
        "--stage4-dir",
        default="pipeline_stage4_structures",
        help="Output directory for stage-4 structure files",
    )
    parser.add_argument(
        "--stage5-out",
        default="pipeline_stage5_filtered.csv",
        help="Output CSV for stage-5 energy-above-hull filtering",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.2,
        help="Maximum predicted energy_above_hull per atom to keep",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="CPU workers for SOAP/structure generation (default: all cores)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for nearest-prototype search (requires torch)",
    )
    parser.add_argument(
        "--seed-selection-seed",
        type=int,
        default=None,
        help="Seed for random selection in stage-1 (default: random each run)",
    )

    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()

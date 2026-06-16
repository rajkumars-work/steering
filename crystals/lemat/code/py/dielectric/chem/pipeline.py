"""
pipeline.py
===========

Stage-based pipeline for composition discovery.

Stage 1:
  - Pick two compositions from each cluster:
      * one far from cluster center (max distance_score)
      * one with high bandgap + eps_0 (min property_score if present)

Stage 2:
  - For each selected composition, call expand_many(return_pareto=True)
  - Save expanded Pareto compositions with properties to CSV

Stage 3:
  - Take stage-2 output, compute Pareto front ranks
  - Keep only rank-0 (front) and rank-1 compositions

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

from composition_expander import expand_many
from pareto_front import ParetoFront
from filters import (
    filter_candidates_by_training_species,
    filter_rows_by_chemistry,
    stage3_filter,
    stage5_filter_energy,
)
from soap import load_structures_from_extxyz, predict_properties
from structure_generator import (
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


def stage1_select_seeds(cluster_rows: List[Dict[str, str]]) -> List[SeedSelection]:
    required_cols = {"formula", "cluster"}
    missing = [c for c in required_cols if c not in cluster_rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in cluster CSV: {missing}")

    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in cluster_rows:
        groups.setdefault(str(row.get("cluster", "")), []).append(row)

    seeds: List[SeedSelection] = []
    for cluster_id, rows in sorted(groups.items(), key=lambda x: x[0]):
        if not rows:
            continue

        def distance_score(r: Dict[str, str]) -> float:
            v = _to_float(r.get("distance_score"))
            return v if v is not None else float("inf")

        def property_score(r: Dict[str, str]) -> float:
            v = _to_float(r.get("property_score"))
            return v if v is not None else float("inf")

        def combined_score(r: Dict[str, str]) -> float:
            bg = _to_float(r.get("dft_band_gap"))
            eps = _to_float(r.get("dft_eps_0"))
            if bg is None or eps is None:
                return float("-inf")
            return bg + eps

        center_row = max(rows, key=distance_score)

        has_property_score = any(r.get("property_score") not in (None, "") for r in rows)
        if has_property_score:
            ranked_high = sorted(rows, key=property_score)
        else:
            ranked_high = sorted(rows, key=combined_score, reverse=True)

        high_row = ranked_high[0]
        if high_row is center_row and len(ranked_high) > 1:
            high_row = ranked_high[1]

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

    if not candidate_atoms:
        return [], []

    # Filter out candidates with elements not present in the SOAP training set.
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

    if not candidate_atoms:
        return [], []

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

    os.makedirs(output_dir, exist_ok=True)
    selected_rows: List[Dict[str, object]] = []
    selected_atoms: List[ASEAtoms] = []

    for formula, idxs in comp_to_indices.items():
        cand_list = [candidate_meta[i] for i in idxs]
        if not cand_list:
            continue

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

        pos_map = {idx: pos for pos, idx in enumerate(idxs)}

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
        selected = pool_candidates[:keep_per_comp]

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


def run_pipeline(args: argparse.Namespace) -> None:
    cluster_rows = _read_csv_rows(args.clusters)
    if not cluster_rows:
        raise ValueError(f"No rows found in cluster CSV: {args.clusters}")

    # Stage 1: select seed compositions.
    seeds = stage1_select_seeds(cluster_rows)
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
    _write_csv(
        args.stage5_out,
        stage5_rows,
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
            "pred_energy_above_hull",
            "pred_energy_above_hull_per_atom",
            "n_atoms",
        ],
    )

    print(f"Stage 1 seeds: {len(seeds)} written to {args.stage1_out}")
    print(f"Stage 2 expanded: {len(expanded)} written to {args.stage2_out}")
    print(f"Stage 3 filtered: {len(stage3)} written to {args.stage3_out}")
    print(f"Stage 4 structures: {len(stage4_rows)} written to {args.stage4_out}")
    print(f"Stage 5 filtered: {len(stage5_rows)} written to {args.stage5_out}")


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

    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()

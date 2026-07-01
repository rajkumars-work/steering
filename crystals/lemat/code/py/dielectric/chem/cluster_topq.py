#!/usr/bin/env python3
"""
Cluster top-quadrant dielectric compositions using magpie + custom features.

Workflow:
1) Load structures from extxyz files.
2) Filter to top-q: >= median dft_band_gap and >= median dft_eps_0.
3) Compute features: magpie + electronegativity range + stoich entropy + anion flags.
4) Scale -> PCA(10) -> KMeans(k=25).
5) Optionally sample N items per cluster.
"""

from __future__ import annotations

import argparse
import csv
import os
from glob import glob
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ase.io import read
from pymatgen.core import Composition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from composition_features import (
    compute_magpie_and_custom,
    dominant_chemistry,
    get_magpie_featurizer,
)


def load_structures(extxyz_glob: str) -> List:
    files = glob(extxyz_glob)
    structures: List = []
    for file in files:
        atoms_list = read(file, index=":")
        if isinstance(atoms_list, list):
            structures.extend(atoms_list)
        else:
            structures.append(atoms_list)
    return structures


def deduplicate_by_composition(structures: Iterable) -> List:
    comp_map: Dict[str, object] = {}
    for atoms in structures:
        comp_str = atoms.get_chemical_formula()
        energy = atoms.info.get("energy_above_hull", float("inf"))
        if comp_str not in comp_map:
            comp_map[comp_str] = atoms
        else:
            existing_energy = comp_map[comp_str].info.get(
                "energy_above_hull", float("inf")
            )
            if energy < existing_energy:
                comp_map[comp_str] = atoms
    return list(comp_map.values())


def build_topq(structures: List) -> Tuple[List, float, float]:
    bandgaps = []
    eps0s = []
    valid = []
    for atoms in structures:
        bg = atoms.info.get("dft_band_gap")
        eps0 = atoms.info.get("dft_eps_0")
        if bg is None or eps0 is None:
            continue
        if not np.isfinite(bg) or not np.isfinite(eps0):
            continue
        bandgaps.append(float(bg))
        eps0s.append(float(eps0))
        valid.append(atoms)

    if not bandgaps:
        raise ValueError("No valid structures with dft_band_gap and dft_eps_0")

    bg_median = float(np.median(bandgaps))
    eps0_median = float(np.median(eps0s))

    topq = []
    for atoms in valid:
        if atoms.info.get("dft_band_gap") >= bg_median and atoms.info.get(
            "dft_eps_0"
        ) >= eps0_median:
            topq.append(atoms)

    return topq, bg_median, eps0_median


def cluster_topq(
    structures: List,
    n_clusters: int,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    magpie = get_magpie_featurizer()

    feats = []
    labels: List[str] = []
    for i, atoms in enumerate(structures):
        comp = Composition(atoms.get_chemical_formula())
        f, l = compute_magpie_and_custom(comp, magpie)
        feats.append(f)
        if i == 0:
            labels = l

    X = np.vstack(feats)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    clusters = kmeans.fit_predict(X_pca)

    return X, X_pca, clusters, kmeans.cluster_centers_, labels


def _composition_fraction_dict(comp: Composition) -> Dict[str, float]:
    frac = comp.fractional_composition
    return {str(el): float(frac[el]) for el in frac.elements}


def compute_cluster_labels(
    structures: List,
    X_pca: np.ndarray,
    clusters: np.ndarray,
    centers: np.ndarray,
    top_n_elements: int,
    center_frac: float,
    center_min: int,
) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c in range(centers.shape[0]):
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            labels[c] = ""
            continue
        dists = np.linalg.norm(X_pca[idxs] - centers[c], axis=1)
        order = np.argsort(dists)
        n_center = max(center_min, int(np.ceil(center_frac * len(idxs))))
        n_center = min(n_center, len(idxs))
        chosen = idxs[order[:n_center]]
        formulas = [structures[i].get_chemical_formula() for i in chosen]
        chem_label = dominant_chemistry(formulas)
        if chem_label != "random mix of everything":
            labels[c] = chem_label
            continue

        elem_totals: Dict[str, float] = {}
        for i in chosen:
            comp = Composition(structures[i].get_chemical_formula())
            frac_dict = _composition_fraction_dict(comp)
            for el, frac in frac_dict.items():
                elem_totals[el] = elem_totals.get(el, 0.0) + frac
        if not elem_totals:
            labels[c] = chem_label
            continue
        for el in elem_totals:
            elem_totals[el] /= float(n_center)
        top_elements = sorted(
            elem_totals.items(), key=lambda x: x[1], reverse=True
        )[: max(1, top_n_elements)]
        labels[c] = "-".join(el for el, _ in top_elements)
    return labels


def compute_distance_scores(
    X_pca: np.ndarray, clusters: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    scores = np.zeros(X_pca.shape[0], dtype=float)
    for c in range(centers.shape[0]):
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            continue
        dists = np.linalg.norm(X_pca[idxs] - centers[c], axis=1)
        d_min = float(np.min(dists))
        d_max = float(np.max(dists))
        if np.isclose(d_max, d_min):
            scores[idxs] = 1.0
            continue
        d_norm = (dists - d_min) / (d_max - d_min)
        scores[idxs] = 1.0 + 99.0 * d_norm
    return scores


def compute_property_scores(structures: List) -> np.ndarray:
    bandgaps = np.array(
        [float(atoms.info.get("dft_band_gap")) for atoms in structures],
        dtype=float,
    )
    eps0s = np.array(
        [float(atoms.info.get("dft_eps_0")) for atoms in structures], dtype=float
    )
    bg_min, bg_max = float(np.min(bandgaps)), float(np.max(bandgaps))
    eps_min, eps_max = float(np.min(eps0s)), float(np.max(eps0s))
    if np.isclose(bg_max, bg_min):
        bg_norm = np.ones_like(bandgaps)
    else:
        bg_norm = (bandgaps - bg_min) / (bg_max - bg_min)
    if np.isclose(eps_max, eps_min):
        eps_norm = np.ones_like(eps0s)
    else:
        eps_norm = (eps0s - eps_min) / (eps_max - eps_min)
    combined = 0.5 * (bg_norm + eps_norm)
    scores = 1.0 + 99.0 * (1.0 - combined)
    return scores


def sample_per_cluster(
    structures: List,
    clusters: np.ndarray,
    n_per_cluster: int,
    random_state: int,
) -> List[int]:
    rng = np.random.default_rng(random_state)
    selected = []
    for c in range(clusters.max() + 1):
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) <= n_per_cluster:
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=n_per_cluster, replace=False)
        selected.extend(chosen.tolist())
    return selected


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-q clustering (magpie + custom)")
    parser.add_argument(
        "--extxyz",
        default="/data/assets/datasets/dielectric/mp/*.extxyz",
        help="Glob for extxyz files",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate by composition (keep lowest energy_above_hull)",
    )
    parser.add_argument("--k", type=int, default=25, help="Number of clusters")
    parser.add_argument(
        "--pca-dims", type=int, default=10, help="PCA components"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--n-per-cluster",
        type=int,
        default=0,
        help="If >0, sample N items from each cluster",
    )
    parser.add_argument(
        "--label-top-n-elements",
        type=int,
        default=3,
        help="Number of elements to include in cluster label",
    )
    parser.add_argument(
        "--label-center-frac",
        type=float,
        default=0.2,
        help="Fraction of closest-to-center compositions to use for cluster labels",
    )
    parser.add_argument(
        "--label-center-min",
        type=int,
        default=3,
        help="Minimum number of closest-to-center compositions to use for cluster labels",
    )
    parser.add_argument(
        "--out-dir",
        default="cluster_out",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    print("Loading structures...")
    structures = load_structures(args.extxyz)
    print(f"Loaded {len(structures)} structures")

    if args.dedup:
        print("Deduplicating by composition...")
        structures = deduplicate_by_composition(structures)
        print(f"After deduplication: {len(structures)} unique compositions")

    print("Selecting top-q by medians of dft_band_gap and dft_eps_0...")
    topq, bg_med, eps0_med = build_topq(structures)
    print(f"Median dft_band_gap = {bg_med:.4f}, dft_eps_0 = {eps0_med:.4f}")
    print(f"Top-q count: {len(topq)}")

    print("Computing features, PCA, and clustering...")
    _, X_pca, clusters, centers, _ = cluster_topq(
        topq, n_clusters=args.k, n_components=args.pca_dims, random_state=args.random_state
    )
    print("Computing cluster labels and scores...")
    cluster_labels = compute_cluster_labels(
        topq,
        X_pca,
        clusters,
        centers,
        top_n_elements=args.label_top_n_elements,
        center_frac=args.label_center_frac,
        center_min=args.label_center_min,
    )
    distance_scores = compute_distance_scores(X_pca, clusters, centers)
    property_scores = compute_property_scores(topq)

    # Build output rows
    rows = []
    for i, (atoms, c) in enumerate(zip(topq, clusters)):
        rows.append(
            {
                "material_id": atoms.info.get("material_id", ""),
                "formula": atoms.get_chemical_formula(),
                "dft_band_gap": atoms.info.get("dft_band_gap"),
                "dft_eps_0": atoms.info.get("dft_eps_0"),
                "cluster": int(c),
                "cluster_label": cluster_labels.get(int(c), ""),
                "distance_score": float(distance_scores[i]),
                "property_score": float(property_scores[i]),
            }
        )

    out_all = os.path.join(args.out_dir, "topq_clusters.csv")
    write_csv(
        out_all,
        rows,
        [
            "material_id",
            "formula",
            "dft_band_gap",
            "dft_eps_0",
            "cluster",
            "cluster_label",
            "distance_score",
            "property_score",
        ],
    )
    print(f"Wrote: {out_all}")

    if args.n_per_cluster and args.n_per_cluster > 0:
        selected = sample_per_cluster(
            topq, clusters, args.n_per_cluster, args.random_state
        )
        rows_sel = [rows[i] for i in selected]
        out_sel = os.path.join(
            args.out_dir, f"topq_sample_n{args.n_per_cluster}.csv"
        )
        write_csv(
            out_sel,
            rows_sel,
            [
                "material_id",
                "formula",
                "dft_band_gap",
                "dft_eps_0",
                "cluster",
                "cluster_label",
                "distance_score",
                "property_score",
            ],
        )
        print(
            f"Sampled {len(rows_sel)} items ({args.n_per_cluster} per cluster when possible)"
        )
        print(f"Wrote: {out_sel}")


if __name__ == "__main__":
    main()

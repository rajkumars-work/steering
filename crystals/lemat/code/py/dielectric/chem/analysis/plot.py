"""Plotting helpers for dielectric analysis.

This module provides a simple bandgap vs eps_0 plot with a 2-D histogram
background representing the training distribution, and an overlay of Pareto
front points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import csv
import math
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for batch plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from ase.io import read
from glob import glob


@dataclass(frozen=True)
class ParetoPoint:
    bandgap: float
    eps_0: float


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_points(points: Iterable[Sequence[float]] | None) -> np.ndarray:
    if points is None:
        return np.zeros((0, 2), dtype=float)
    rows = []
    for item in points:
        if isinstance(item, ParetoPoint):
            bg, eps = item.bandgap, item.eps_0
        else:
            if len(item) < 2:
                continue
            bg, eps = item[0], item[1]
        bg = _to_float(bg)
        eps = _to_float(eps)
        if bg is None or eps is None:
            continue
        if not (math.isfinite(bg) and math.isfinite(eps)):
            continue
        rows.append((bg, eps))
    if not rows:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(rows, dtype=float)


def _default_family_markers() -> dict[str, str]:
    return {
        "f_based": "o",
        "cl_based": "s",
        "br_based": "D",
        "i_based": "P",
        "mixed_halide": "X",
        "oxide": "^",
        "other": "v",
    }


def _default_structure_colors() -> dict[str, str]:
    return {
        "simple_binary": "#0f766e",
        "double_perovskite_like": "#d97706",
        "perovskite_like": "#b91c1c",
        "complex_oxide": "#7c3aed",
        "elpasolite_like": "#2563eb",
        "heavy_metal_halide": "#1d4ed8",
        "complex_halide": "#0ea5e9",
        "complex_salt": "#6b7280",
    }


def load_bandgap_eps0_from_csv(
    path: str | Path,
    bandgap_col: str = "bandgap",
    eps0_col: str = "eps_0",
) -> tuple[np.ndarray, np.ndarray]:
    """Load bandgap and eps_0 columns from a CSV file."""
    bandgaps = []
    eps0s = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bg = _to_float(row.get(bandgap_col))
            eps = _to_float(row.get(eps0_col))
            if bg is None or eps is None:
                continue
            if not (math.isfinite(bg) and math.isfinite(eps)):
                continue
            bandgaps.append(bg)
            eps0s.append(eps)
    return np.asarray(bandgaps, dtype=float), np.asarray(eps0s, dtype=float)


_TRAINING_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DEFAULT_CACHE_FILE = Path("data/training_bandgap_eps0_cache.npz")


def load_training_bandgap_eps0(
    extxyz_glob: str = "/data/assets/datasets/dielectric/mp/*.extxyz",
    bandgap_key: str = "dft_band_gap",
    eps0_key: str = "dft_eps_0",
    cache_path: str | Path | None = _DEFAULT_CACHE_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load training bandgap/eps_0 values from extxyz files (cached)."""
    cache_key = f"{extxyz_glob}|{bandgap_key}|{eps0_key}"
    cached = _TRAINING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path)
            bg_arr = np.asarray(data["bandgaps"], dtype=float)
            eps_arr = np.asarray(data["eps0s"], dtype=float)
            if bg_arr.size and eps_arr.size:
                _TRAINING_CACHE[cache_key] = (bg_arr, eps_arr)
                return bg_arr, eps_arr

    bandgaps: list[float] = []
    eps0s: list[float] = []
    files = glob(extxyz_glob)
    for path in files:
        atoms_list = read(path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for atoms in atoms_list:
            bg = _to_float(atoms.info.get(bandgap_key))
            eps = _to_float(atoms.info.get(eps0_key))
            if bg is None or eps is None:
                continue
            if not (math.isfinite(bg) and math.isfinite(eps)):
                continue
            bandgaps.append(bg)
            eps0s.append(eps)

    bg_arr = np.asarray(bandgaps, dtype=float)
    eps_arr = np.asarray(eps0s, dtype=float)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, bandgaps=bg_arr, eps0s=eps_arr)
    _TRAINING_CACHE[cache_key] = (bg_arr, eps_arr)
    return bg_arr, eps_arr


def _white_to_color_cmap(color: str = "#2a9d8f") -> LinearSegmentedColormap:
    base = to_rgb(color)
    return LinearSegmentedColormap.from_list("white_to_color", ["#ffffff", base])


def _resolve_cmap(cmap):
    if cmap is None:
        return _white_to_color_cmap()
    if isinstance(cmap, str):
        if cmap == "white_to_color":
            return _white_to_color_cmap()
        return plt.get_cmap(cmap)
    return cmap


def _density_to_rgba(
    counts: np.ndarray, base_color: str, log_scale: bool
) -> np.ndarray:
    base = np.array(to_rgb(base_color), dtype=float)
    rgba = np.zeros((counts.shape[0], counts.shape[1], 4), dtype=float)
    if log_scale:
        positive = counts[counts > 0]
        if positive.size == 0:
            rgba[..., :3] = 1.0
            rgba[..., 3] = 1.0
            return rgba
        vmin = max(1.0, float(positive.min()))
        vmax = float(positive.max())
        scale = np.zeros_like(counts, dtype=float)
        mask = counts > 0
        scale[mask] = (np.log10(counts[mask]) - np.log10(vmin)) / max(
            np.log10(vmax) - np.log10(vmin), 1e-9
        )
    else:
        vmax = float(np.max(counts)) if counts.size else 0.0
        if vmax <= 0:
            rgba[..., :3] = 1.0
            rgba[..., 3] = 1.0
            return rgba
        scale = counts / vmax
    scale = np.clip(scale, 0.0, 1.0)
    # Single-hue background: blend from white (low density) to base color (high).
    rgba[..., :3] = (1.0 - scale[..., None]) + scale[..., None] * base
    rgba[..., 3] = 1.0
    return rgba


def _plot_bandgap_eps0_distribution(
    bandgaps: Sequence[float],
    eps0s: Sequence[float],
    pareto_points: Iterable[Sequence[float]] | None = None,
    pareto_families: Sequence[str] | None = None,
    pareto_structures: Sequence[str] | None = None,
    out_path: str | Path = "/tmp/di_plot.png",
    bins: int | tuple[int, int] = 60,
    cmap=None,
    base_color: str = "#2a9d8f",
    log_scale: bool = True,
    y_scale: str = "auto",
    pareto_color: str = "black",
    pareto_size: float = 40.0,
    family_markers: Mapping[str, str] | None = None,
    structure_colors: Mapping[str, str] | None = None,
    x_label: str = "Bandgap (eV)",
    y_label: str = "eps_0 (static)",
    title: str | None = "Dielectrics",
) -> None:
    """Plot bandgap vs eps_0 with a 2-D histogram background.

    The background intensity is proportional to the count of training points in
    each histogram bucket. Pareto points are overlaid in a distinct color.
    """
    bg = np.asarray(bandgaps, dtype=float)
    eps = np.asarray(eps0s, dtype=float)

    valid = np.isfinite(bg) & np.isfinite(eps)
    bg = bg[valid]
    eps = eps[valid]
    if bg.size == 0 or eps.size == 0:
        raise ValueError("No valid bandgap/eps_0 values provided for plotting.")

    counts, xedges, yedges = np.histogram2d(bg, eps, bins=bins)
    counts = counts.T  # transpose so y-axis is vertical

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rgba = _density_to_rgba(counts, base_color, log_scale)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )

    pareto = _coerce_points(pareto_points)
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    legend_handles.append(Patch(facecolor=base_color, edgecolor="none"))
    legend_labels.append("materials-project dielectrics")
    # Auto-select a y-scale when Pareto points span orders of magnitude beyond
    # the training distribution. This keeps the background readable.
    if y_scale == "auto":
        y_candidates = [eps]
        if pareto.size:
            y_candidates.append(pareto[:, 1])
        y_all = np.concatenate(y_candidates)
        y_all = y_all[np.isfinite(y_all)]
        y_all = y_all[y_all > 0]
        if y_all.size:
            y_min = float(y_all.min())
            y_max = float(y_all.max())
            if y_max / max(y_min, 1e-12) > 200.0:
                y_scale = "log"
            else:
                y_scale = "linear"
        else:
            y_scale = "linear"
    if pareto.size:
        family_markers = dict(_default_family_markers()) | dict(family_markers or {})
        structure_colors = dict(_default_structure_colors()) | dict(
            structure_colors or {}
        )
        families = (
            list(pareto_families)
            if pareto_families is not None
            else ["unknown"] * pareto.shape[0]
        )
        structures = (
            list(pareto_structures)
            if pareto_structures is not None
            else ["unknown"] * pareto.shape[0]
        )
        if len(families) != pareto.shape[0] or len(structures) != pareto.shape[0]:
            raise ValueError("Pareto family/structure metadata length mismatch.")
        families_arr = np.asarray(families, dtype=object)
        structures_arr = np.asarray(structures, dtype=object)
        unique_families = list(dict.fromkeys(families_arr.tolist()))
        unique_structures = list(dict.fromkeys(structures_arr.tolist()))

        for fam in unique_families:
            mask = families_arr == fam
            if not np.any(mask):
                continue
            colors = [
                structure_colors.get(struct, pareto_color)
                for struct in structures_arr[mask].tolist()
            ]
            ax.scatter(
                pareto[mask, 0],
                pareto[mask, 1],
                s=pareto_size,
                c=colors,
                marker=family_markers.get(str(fam), "o"),
                edgecolors="white",
                linewidths=0.4,
            )

        for struct in unique_structures:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=structure_colors.get(str(struct), pareto_color),
                    markeredgecolor="white",
                    markersize=6,
                )
            )
            legend_labels.append(str(struct))

        for fam in unique_families:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=family_markers.get(str(fam), "o"),
                    color="none",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markersize=6,
                )
            )
            legend_labels.append(str(fam))

    ax.legend(legend_handles, legend_labels, frameon=False, loc="best")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_scale in {"linear", "log"}:
        ax.set_yscale(y_scale)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_pareto_triples_on_training(
    training_bandgaps: Sequence[float],
    training_eps0s: Sequence[float],
    triples: Iterable[Sequence[object]],
    out_path: str | Path = "/tmp/di_plot.png",
    bins: int | tuple[int, int] = 60,
    cmap=None,
    base_color: str = "#2a9d8f",
    log_scale: bool = True,
    y_scale: str = "auto",
    pareto_color: str = "black",
    pareto_size: float = 45.0,
    x_label: str = "Bandgap (eV)",
    y_label: str = "eps_0 (static)",
    title: str | None = "Dielectrics",
) -> None:
    """Plot (formula, bandgap, eps_0) triples over training distribution.

    The background is the 2-D histogram of the training data. Pareto points are
    plotted with labels derived from the formula field.
    """
    pareto_points = []
    labels = []
    for item in triples:
        if len(item) < 3:
            continue
        formula, bandgap, eps0 = item[0], item[1], item[2]
        bg = _to_float(bandgap)
        eps = _to_float(eps0)
        if bg is None or eps is None:
            continue
        if not (math.isfinite(bg) and math.isfinite(eps)):
            continue
        pareto_points.append((bg, eps))
        labels.append(str(formula))

    pareto_points = np.asarray(pareto_points, dtype=float)

    # Plot background + Pareto points
    _plot_bandgap_eps0_distribution(
        training_bandgaps,
        training_eps0s,
        pareto_points=pareto_points,
        pareto_families=None,
        pareto_structures=None,
        out_path=out_path,
        bins=bins,
        cmap=cmap,
        base_color=base_color,
        log_scale=log_scale,
        y_scale=y_scale,
        pareto_color=pareto_color,
        pareto_size=pareto_size,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )


def plot_pareto_points(
    points: list[dict[str, object]],
    label: bool = False,
    out_path: str | Path = "/tmp/di_plot.png",
    extxyz_glob: str = "/data/assets/datasets/dielectric/mp/*.extxyz",
    bandgap_key: str = "dft_band_gap",
    eps0_key: str = "dft_eps_0",
    cache_path: str | Path | None = _DEFAULT_CACHE_FILE,
    bins: int | tuple[int, int] = 60,
    cmap=None,
    base_color: str = "#2a9d8f",
    log_scale: bool = True,
    y_scale: str = "auto",
    pareto_color: str = "black",
    pareto_size: float = 45.0,
    family_markers: Mapping[str, str] | None = None,
    structure_colors: Mapping[str, str] | None = None,
    x_label: str = "Bandgap (eV)",
    y_label: str = "eps_0 (static)",
    title: str | None = "Dielectrics",
) -> None:
    """Plot Pareto dict points over training distribution.

    Example:
        points = [
            {
                "composition": "Na2BeGaF7",
                "bandgap": 5.116933,
                "eps_0": 316.069572,
                "family": "f_based",
                "structure": "heavy_metal_halide",
            },
        ]
        plot_pareto_points(points, label=True)
    """
    training_bandgaps, training_eps0s = load_training_bandgap_eps0(
        extxyz_glob=extxyz_glob,
        bandgap_key=bandgap_key,
        eps0_key=eps0_key,
        cache_path=cache_path,
    )

    pareto_points = []
    labels = []
    families: list[str] = []
    structures: list[str] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        bg = _to_float(item.get("bandgap"))
        eps = _to_float(item.get("eps_0"))
        if bg is None or eps is None:
            continue
        if not (math.isfinite(bg) and math.isfinite(eps)):
            continue
        pareto_points.append((bg, eps))
        labels.append(str(item.get("composition", "")))
        families.append(str(item.get("family", "unknown")))
        structures.append(str(item.get("structure", "unknown")))

    pareto_points = np.asarray(pareto_points, dtype=float)

    # Background plot
    _plot_bandgap_eps0_distribution(
        training_bandgaps,
        training_eps0s,
        pareto_points=pareto_points,
        pareto_families=families,
        pareto_structures=structures,
        out_path=out_path,
        bins=bins,
        cmap=cmap,
        base_color=base_color,
        log_scale=log_scale,
        y_scale=y_scale,
        pareto_color=pareto_color,
        pareto_size=pareto_size,
        family_markers=family_markers,
        structure_colors=structure_colors,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )

    if label and pareto_points.size:
        # Re-render with annotations so labels are included in the saved file.
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        counts, xedges, yedges = np.histogram2d(
            training_bandgaps, training_eps0s, bins=bins
        )
        counts = counts.T
        rgba = _density_to_rgba(counts, base_color, log_scale)
        ax.imshow(
            rgba,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
        )
        ax.set_facecolor("white")
        family_markers = dict(_default_family_markers()) | dict(family_markers or {})
        structure_colors = dict(_default_structure_colors()) | dict(
            structure_colors or {}
        )
        families_arr = np.asarray(families, dtype=object)
        structures_arr = np.asarray(structures, dtype=object)
        unique_families = list(dict.fromkeys(families_arr.tolist()))
        unique_structures = list(dict.fromkeys(structures_arr.tolist()))
        for fam in unique_families:
            mask = families_arr == fam
            if not np.any(mask):
                continue
            colors = [
                structure_colors.get(struct, pareto_color)
                for struct in structures_arr[mask].tolist()
            ]
            ax.scatter(
                pareto_points[mask, 0],
                pareto_points[mask, 1],
                s=pareto_size,
                c=colors,
                marker=family_markers.get(str(fam), "o"),
                edgecolors="white",
                linewidths=0.4,
            )
        legend_handles = [Patch(facecolor=base_color, edgecolor="none")]
        legend_labels = ["materials-project dielectrics"]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=pareto_color,
                markeredgecolor="white",
                markersize=6,
            )
        )
        legend_labels.append("Pareto")
        ax.legend(legend_handles, legend_labels, frameon=False, loc="best")
        if y_scale == "auto":
            y_candidates = [np.asarray(training_eps0s, dtype=float)]
            if pareto_points.size:
                y_candidates.append(pareto_points[:, 1])
            y_all = np.concatenate(y_candidates)
            y_all = y_all[np.isfinite(y_all)]
            y_all = y_all[y_all > 0]
            if y_all.size:
                y_min = float(y_all.min())
                y_max = float(y_all.max())
                if y_max / max(y_min, 1e-12) > 200.0:
                    y_scale = "log"
                else:
                    y_scale = "linear"
            else:
                y_scale = "linear"
        if y_scale in {"linear", "log"}:
            ax.set_yscale(y_scale)
        for (x, y), text in zip(pareto_points, labels):
            ax.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        legend_handles = [Patch(facecolor=base_color, edgecolor="none")]
        legend_labels = ["materials-project dielectrics"]
        for struct in unique_structures:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=structure_colors.get(str(struct), pareto_color),
                    markeredgecolor="white",
                    markersize=6,
                )
            )
            legend_labels.append(str(struct))
        for fam in unique_families:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=family_markers.get(str(fam), "o"),
                    color="none",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markersize=6,
                )
            )
            legend_labels.append(str(fam))
        ax.legend(legend_handles, legend_labels, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


__all__ = [
    "ParetoPoint",
    "load_bandgap_eps0_from_csv",
    "load_training_bandgap_eps0",
    "plot_pareto_points",
]


if __name__ == "__main__":
    points = [
        {
            "composition": "AlBiF4",
            "bandgap": 5.26,
            "eps_0": 8.38,
            "family": "f_based",
            "structure": "heavy_metal_halide",
        },
        {
            "composition": "Li2GaPbF7",
            "bandgap": 3.66,
            "eps_0": 15.35,
            "family": "f_based",
            "structure": "heavy_metal_halide",
        },
        {
            "composition": "RbPb2Cl5",
            "bandgap": 2.91,
            "eps_0": 20.07,
            "family": "cl_based",
            "structure": "heavy_metal_halide",
        },
        {
            "composition": "Ca2ZrTiO6",
            "bandgap": 2.84,
            "eps_0": 48.96,
            "family": "oxide",
            "structure": "double_perovskite_like",
        },
        {
            "composition": "GaBiF4",
            "bandgap": 0.23,
            "eps_0": 190.49,
            "family": "f_based",
            "structure": "heavy_metal_halide",
        },
        {
            "composition": "Na2BeGaF7",
            "bandgap": 5.12,
            "eps_0": 61.7,
            "family": "f_based",
            "structure": "complex_halide",
        },
    ]

    plot_pareto_points(points, label=True)

"""Automated property binner.

Given an array of float values for a property, produces bins that respect
the minimum-support constraint implied by the training-set size and the
nested source-dropout scheme. Emits a uniform, property-agnostic label
vocabulary so no per-property naming intervention is needed.

Derivation (see docs/NestedPropDropoutPlan.md and the approved binning plan):

    retention = (1 - p_segment) * (1 - p_label)
    min_support M = ceil(target_updates / (epochs * retention))

With the current defaults (p_segment=0.2, p_label=0.2, epochs=150,
target_updates=200_000) → M ≈ 2083, rounded to 2000.

Labels follow `<prop_tag>-<tier>`, where `tier` is drawn symmetrically from
the canonical scale TIER_SCALE. For K bins, TIER_FOR_K[K] gives the exact
tier names used (centered on `mid`).

All functions are pure — no file I/O, no RNG. The CLI driver does I/O.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tier vocabulary
# ---------------------------------------------------------------------------
TIER_SCALE: tuple[str, ...] = (
    "ulow", "vlow", "low", "mid", "high", "vhigh", "uhigh",
)

# Which tiers to use when the binner selects K bins. Symmetric about mid.
TIER_FOR_K: dict[int, tuple[str, ...]] = {
    1: ("mid",),
    2: ("low", "high"),
    3: ("low", "mid", "high"),
    4: ("vlow", "low", "high", "vhigh"),
    5: ("vlow", "low", "mid", "high", "vhigh"),
    6: ("ulow", "vlow", "low", "high", "vhigh", "uhigh"),
    7: ("ulow", "vlow", "low", "mid", "high", "vhigh", "uhigh"),
}

DEFAULT_P_SEGMENT = 0.2
DEFAULT_P_LABEL = 0.2
DEFAULT_EPOCHS = 150
DEFAULT_TARGET_UPDATES = 200_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BinnerConfig:
    """Configuration for `auto_bin`.

    Supply either `min_support` directly, or the training-dynamics parameters
    (`p_segment`, `p_label`, `epochs`, `target_updates`) from which it is
    derived at `resolve_support` time.
    """
    prop_tag: str
    k_max: int = 5
    min_support: Optional[int] = None
    p_segment: float = DEFAULT_P_SEGMENT
    p_label: float = DEFAULT_P_LABEL
    epochs: int = DEFAULT_EPOCHS
    target_updates: int = DEFAULT_TARGET_UPDATES


@dataclass
class Binning:
    prop_tag: str
    edges: list[float]              # K-1 interior edges, ascending
    labels: list[str]               # K tier-labeled strings
    counts: list[int]               # rows per bin
    effective_support: int          # min(counts)
    tiers_used: list[str]           # subset of TIER_SCALE, ordered
    min_support: int                # threshold that was honored
    n_values: int                   # total finite values considered
    n_skipped: int = 0              # non-finite values skipped
    snapped_edges: list[float] = field(default_factory=list)  # pre-merge record

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_support(cfg: BinnerConfig) -> int:
    """Return the min_support threshold M.

    If `cfg.min_support` is set, use it directly. Otherwise derive it from
    the dropout retention formula.
    """
    if cfg.min_support is not None:
        return int(cfg.min_support)

    retention = (1.0 - cfg.p_segment) * (1.0 - cfg.p_label)
    if retention <= 0 or cfg.epochs <= 0:
        raise ValueError(
            f"Cannot derive min_support from p_segment={cfg.p_segment}, "
            f"p_label={cfg.p_label}, epochs={cfg.epochs}")
    return int(math.ceil(cfg.target_updates / (cfg.epochs * retention)))


def auto_bin(values: Sequence[float], cfg: BinnerConfig) -> Binning:
    """Produce bins for `values` under the support constraint implied by cfg.

    Algorithm:
      1. K = min(k_max, floor(N_finite / M))
      2. Start from K equal-count quantile edges.
      3. Snap each interior edge to a "round" value at the property's
         natural scale (snap_edge).
      4. Recount; if any bin < M, merge with its smaller neighbor.
      5. Emit tier-labeled result.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    n_skipped = int(arr.size - n)
    if n == 0:
        raise ValueError(f"No finite values for prop_tag={cfg.prop_tag!r}")

    M = resolve_support(cfg)

    # K bounded by data and by k_max; must be in 1..7 (TIER_FOR_K coverage)
    k = min(cfg.k_max, max(1, n // M))
    k = max(1, min(k, 7))

    # Sort once — quantile edges are O(K) after this.
    sorted_vals = np.sort(finite)

    # Initial edges: K-1 equal-count quantile cuts.
    if k == 1:
        raw_edges: list[float] = []
    else:
        quantiles = np.linspace(0, 1, k + 1)[1:-1]
        # np.quantile on a pre-sorted array
        idx = (quantiles * (n - 1)).astype(int)
        raw_edges = [float(sorted_vals[i]) for i in idx]

    snapped = [_snap_edge(e) for e in raw_edges]

    # Ensure strictly ascending after snapping (collapse duplicates upward)
    snapped = _dedup_ascending(snapped)

    # Count rows per bin given current edges
    counts = _bin_counts(sorted_vals, snapped)

    # Merge under-supported bins until all >= M or we're down to 1 bin
    while len(counts) > 1 and min(counts) < M:
        i_min = int(np.argmin(counts))
        # Merge with the smaller-centered neighbor (prefer lower-index side
        # if at boundary)
        if i_min == 0:
            merge_right = True
        elif i_min == len(counts) - 1:
            merge_right = False
        else:
            merge_right = counts[i_min - 1] >= counts[i_min + 1]

        # Remove the appropriate interior edge
        edge_idx = i_min if merge_right else i_min - 1
        del snapped[edge_idx]
        counts = _bin_counts(sorted_vals, snapped)

    k = len(counts)
    tiers = list(TIER_FOR_K[k])
    labels = [f"{cfg.prop_tag}-{t}" for t in tiers]
    eff_support = int(min(counts)) if counts else 0

    return Binning(
        prop_tag=cfg.prop_tag,
        edges=[float(e) for e in snapped],
        labels=labels,
        counts=[int(c) for c in counts],
        effective_support=eff_support,
        tiers_used=tiers,
        min_support=int(M),
        n_values=n,
        n_skipped=n_skipped,
        snapped_edges=[float(e) for e in _dedup_ascending([_snap_edge(e) for e in raw_edges])],
    )


def label_value(value: float, binning: Binning) -> Optional[str]:
    """Return the tier-labeled string for `value` given a fitted Binning.

    Returns None for non-finite values.
    """
    if value is None or not (isinstance(value, (int, float)) and math.isfinite(float(value))):
        return None
    v = float(value)
    # edges[i] is the upper boundary of bin i (open on the right)
    for i, edge in enumerate(binning.edges):
        if v < edge:
            return binning.labels[i]
    return binning.labels[-1]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _snap_edge(edge: float) -> float:
    """Snap an edge to a human-readable nearby value.

    Strategy: round to 1 significant figure when |edge| < 10 (decimal scale),
    otherwise round to the nearest integer multiple of a power of 10 sized
    to ~one order of magnitude below the edge. Keeps 0.3 → 0.3, 2.84 → 3,
    12.7 → 13, 134.2 → 130, etc.
    """
    if edge == 0.0:
        return 0.0
    sign = 1.0 if edge > 0 else -1.0
    mag = abs(edge)
    if mag < 1.0:
        # 1 sig fig
        exp = math.floor(math.log10(mag))
        step = 10 ** exp
        return sign * round(mag / step) * step
    elif mag < 10.0:
        return sign * round(mag)
    else:
        # Round to two significant figures so 134→130, 12.7→13, 287→290
        exp = math.floor(math.log10(mag)) - 1
        step = 10 ** exp
        return sign * round(mag / step) * step


def _dedup_ascending(edges: list[float]) -> list[float]:
    """Remove duplicates while preserving ascending order.

    Two edges that snap to the same value would produce an empty bin; we
    keep the first occurrence only.
    """
    seen = set()
    out: list[float] = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return sorted(out)


def _bin_counts(sorted_vals: np.ndarray, edges: list[float]) -> list[int]:
    """Count values per bin given ascending interior edges. Bins are
    half-open on the right (value < edge goes left)."""
    if not edges:
        return [int(sorted_vals.size)]
    # np.searchsorted returns indices such that edges split sorted_vals
    # into (-inf, edges[0]), [edges[0], edges[1]), ..., [edges[-1], +inf)
    idx = np.searchsorted(sorted_vals, edges, side="left")
    counts: list[int] = []
    prev = 0
    for i in idx:
        counts.append(int(i - prev))
        prev = int(i)
    counts.append(int(sorted_vals.size - prev))
    return counts


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

# Canonical sidecar location for the production training set. When the
# training CSV changes the dataset's `VersionSpec.binning_spec_path` overrides
# this; consumers that don't know which dataset is in play (e.g., an
# inference script) can fall back to this path.
DEFAULT_BINNING_SIDECAR = (
    "/data/rkumar/code/py/dielectric/data/big_d15_autolabel.binning.json"
)


def load_binnings(path: Optional[str] = None) -> dict[str, "Binning"]:
    """Load all Binning objects from a sidecar JSON produced by
    `chem.auto_bin_cli`.

    Returns {property_name: Binning}. Property names are the full names
    used at bin-time (e.g., "band_gap", "nequip_eps_0", "stability",
    "density"), not the short prop_tag.

    If `path` is None, uses `DEFAULT_BINNING_SIDECAR`.
    """
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path) if path is not None else _Path(DEFAULT_BINNING_SIDECAR)
    raw = _json.loads(p.read_text())
    out: dict[str, Binning] = {}
    for prop_name, d in raw["properties"].items():
        out[prop_name] = Binning(
            prop_tag=d["prop_tag"],
            edges=list(map(float, d["edges"])),
            labels=list(d["labels"]),
            counts=list(d["counts"]),
            effective_support=int(d["effective_support"]),
            tiers_used=list(d["tiers_used"]),
            min_support=int(d["min_support"]),
            n_values=int(d["n_values"]),
            n_skipped=int(d.get("n_skipped", 0)),
            snapped_edges=list(map(float, d.get("snapped_edges", []))),
        )
    return out


__all__ = [
    "TIER_SCALE",
    "TIER_FOR_K",
    "BinnerConfig",
    "Binning",
    "auto_bin",
    "resolve_support",
    "label_value",
    "load_binnings",
    "DEFAULT_BINNING_SIDECAR",
]

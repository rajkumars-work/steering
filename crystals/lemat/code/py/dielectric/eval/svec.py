"""Axis-agnostic S_vec metric.

S_vec = max(0, 1 − D̄ / D_chance), where:
  D̄          = mean per-axis bin-distance d_a, averaged over axes (weighted) and structures
  D_chance   = expected D̄ under independent uniform marginals (per-axis chance)
  d_a ∈ [0,1] is per-axis: 0 = perfect, 1 = max-wrong.

The metric is **defined by the list of axes you register**. Add an axis with
one of the constructors below (or a custom Axis) and compute over any subset.

Usage:
    from eval.svec import ordered_axis, binary_axis, jaccard_axis, compute_svec

    axes = [
        ordered_axis("bg",   ["vlow","low","mid","high","vhigh"],
                     target_key="bg_target",  pred_key="bg_pred_bin"),
        ordered_axis("rho",  ["ulow","vlow","low","mid","high","vhigh","uhigh"],
                     target_key="rho_target", pred_key="rho_pred_bin"),
        binary_axis("novel", target_key="novel_target", pred_key="novel_pred"),
        jaccard_axis("elements", target_key="elem_target", pred_key="elem_pred"),
    ]
    out = compute_svec(per_structure_entries, axes)
    print(out["S_vec"], out["per_axis_d_bar"])

Add a new axis later (e.g., a continuous property after binning):
    axes.append(ordered_axis("magnetization",
        ["vlow","low","mid","high","vhigh"],
        target_key="mag_target", pred_key="mag_pred_bin"))
    out2 = compute_svec(entries, axes)   # nothing else changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


# ---------------------------------------------------------------------------
# Axis spec — the only thing a metric instance knows
# ---------------------------------------------------------------------------

@dataclass
class Axis:
    name: str
    distance: Callable[[Any, Any], float | None]   # (target, pred) -> [0,1] or None
    chance_d: float                                # E[d] under independent uniform marginals
    target_key: str                                # key into per-structure entry for target value
    pred_key: str                                  # key into entry for predicted value
    weight: float = 1.0                            # relative weight in the aggregate


# ---------------------------------------------------------------------------
# Convenience constructors for the common axis types
# ---------------------------------------------------------------------------

def _ordered_d(tier_order: list[str]) -> Callable:
    K = len(tier_order)
    idx = {t: i for i, t in enumerate(tier_order)}
    def d(target, pred):
        if target is None or pred is None: return None
        if target not in idx or pred not in idx: return None
        return abs(idx[target] - idx[pred]) / (K - 1) if K > 1 else 0.0
    return d


def _ordered_chance(K: int) -> float:
    """E[|i-j|/(K-1)] for i,j ~ Uniform({0..K-1}) independent. Closed form: (K+1)/(3K)."""
    return (K + 1) / (3 * K) if K > 1 else 0.0


def ordered_axis(name: str, tier_order: list[str],
                 target_key: str, pred_key: str, weight: float = 1.0) -> Axis:
    """Ordered K-bin axis using bin-index distance:
        d = |idx(target) − idx(pred)| / (K − 1)
    Both target and pred are bin LABELS. Penalizes "barely-crossed-boundary"
    cases the same as "deep in the wrong bin"; use quantile_axis when the
    continuous value is available."""
    return Axis(name=name,
                distance=_ordered_d(tier_order),
                chance_d=_ordered_chance(len(tier_order)),
                target_key=target_key, pred_key=pred_key, weight=weight)


def _quantile_d(tier_order: list[str], interior_edges: list[float]) -> Callable:
    """Build a distance fn that takes (target_bin_name, predicted_value)
    and returns the quantile-space gap from the target bin's range.

    interior_edges: K-1 sorted floats (autobin sidecar `edges` field). Bin i
    has value-range (edges[i-1], edges[i]). The first (bin 0) and last
    (bin K-1) bins are open-ended; within those, quantile is approximated as
    the bin midpoint (no within-bin granularity in the tails)."""
    K = len(tier_order)
    if len(interior_edges) != K - 1:
        raise ValueError(f"need K-1={K-1} interior edges, got {len(interior_edges)}")
    idx = {t: i for i, t in enumerate(tier_order)}

    def value_to_quantile(v: float) -> float:
        # Locate which bin v falls in.
        bin_idx = 0
        for e in interior_edges:
            if v < e:
                break
            bin_idx += 1
        # bin_idx now in {0, ..., K-1}
        if bin_idx == 0:
            return 0.5 / K              # midpoint of open-ended first bin
        if bin_idx == K - 1:
            return (K - 0.5) / K        # midpoint of open-ended last bin
        lo = interior_edges[bin_idx - 1]
        hi = interior_edges[bin_idx]
        frac = 0.5 if hi == lo else (v - lo) / (hi - lo)
        # Clamp to [0, 1] in case of pathological values right at edges:
        frac = min(1.0, max(0.0, frac))
        return (bin_idx + frac) / K

    def d(target_bin: str, pred_value) -> float | None:
        if target_bin is None or pred_value is None: return None
        if target_bin not in idx: return None
        try:
            q_pred = value_to_quantile(float(pred_value))
        except (TypeError, ValueError):
            return None
        t = idx[target_bin]
        q_lo, q_hi = t / K, (t + 1) / K
        if q_pred < q_lo:  return q_lo - q_pred
        if q_pred > q_hi:  return q_pred - q_hi
        return 0.0
    return d


def _quantile_chance(K: int) -> float:
    """E[d_q] for uniform target bin, uniform q_pred over [0,1]."""
    if K < 2:
        return 0.0
    total = 0.0
    for t in range(K):
        a, b = t / K, 1.0 - (t + 1) / K
        total += (a * a + b * b) / 2
    return total / K


def quantile_axis(name: str, tier_order: list[str], interior_edges: list[float],
                  target_key: str, pred_value_key: str, weight: float = 1.0) -> Axis:
    """Ordered K-bin axis with continuous predicted values, using
    quantile-space distance. Boundary-aware: predictions inside the target
    bin's quantile range have d=0 regardless of where in the bin they land;
    crossings start with the tiny shortfall beyond the boundary.

    interior_edges: K-1 sorted floats from the autobin sidecar (`edges` field).
    The first/last bins are open-ended (no within-bin gradient in the tails)."""
    return Axis(name=name,
                distance=_quantile_d(tier_order, interior_edges),
                chance_d=_quantile_chance(len(tier_order)),
                target_key=target_key, pred_key=pred_value_key, weight=weight)


def binary_axis(name: str, target_key: str, pred_key: str, weight: float = 1.0,
                p_positive: float = 0.5) -> Axis:
    """Binary axis (e.g., novelty yes/no, validity pass/fail).
    chance_d = 2·p(1−p); 0.5 at p=0.5, lower as marginal becomes skewed."""
    def d(t, p):
        if t is None or p is None: return None
        return 0.0 if bool(t) == bool(p) else 1.0
    return Axis(name=name, distance=d, chance_d=2 * p_positive * (1 - p_positive),
                target_key=target_key, pred_key=pred_key, weight=weight)


def categorical_axis(name: str, classes: list[Any],
                     target_key: str, pred_key: str, weight: float = 1.0) -> Axis:
    """Unordered categorical (e.g., space-group). 0/1 distance.
    chance_d = (K−1)/K under uniform marginals."""
    K = len(classes); allowed = set(classes)
    def d(t, p):
        if t is None or p is None: return None
        if t not in allowed or p not in allowed: return None
        return 0.0 if t == p else 1.0
    return Axis(name=name, distance=d, chance_d=(K-1)/K if K >= 1 else 0.0,
                target_key=target_key, pred_key=pred_key, weight=weight)


def jaccard_axis(name: str, target_key: str, pred_key: str,
                 chance_d: float = 0.85, weight: float = 1.0) -> Axis:
    """Unordered-set axis (e.g., element set). Jaccard distance.
    chance_d is data-dependent — default 0.85 from prior measurements on dielectric
    eval; override with --chance_d for a specific corpus."""
    def d(t, p):
        if t is None or p is None: return None
        try:
            T = set(t); P = set(p)
        except TypeError:
            return None
        if not T and not P: return 0.0
        u = T | P
        return 1.0 - len(T & P) / len(u) if u else 0.0
    return Axis(name=name, distance=d, chance_d=chance_d,
                target_key=target_key, pred_key=pred_key, weight=weight)


def integer_axis(name: str, target_key: str, pred_key: str,
                 chance_d: float = 0.5, weight: float = 1.0) -> Axis:
    """Integer-valued axis (e.g., natoms). Relative gap, capped at 1.
    chance_d is empirical (default 0.5 on our prompt distribution)."""
    def d(t, p):
        if t is None or p is None: return None
        if not isinstance(t, (int, float)) or not isinstance(p, (int, float)): return None
        return min(1.0, abs(t - p) / max(1, abs(t)))
    return Axis(name=name, distance=d, chance_d=chance_d,
                target_key=target_key, pred_key=pred_key, weight=weight)


# ---------------------------------------------------------------------------
# The metric itself — axis-list in, S_vec out
# ---------------------------------------------------------------------------

def _empirical_chance_for_axis(a: Axis, entries: list[dict]) -> float | None:
    """Empirical chance-d for axis a under the cell's actual prompt distribution.

    Procedure: take all (target, pred) pairs that are valid; then for each
    target, compute E[d(target, pred')] where pred' is drawn uniformly from
    the empirical prediction distribution in the cell. Average across targets.

    This is the conservative "what would the chance match rate look like
    if you shuffled predictions among prompts?" baseline."""
    targets, preds = [], []
    for e in entries:
        t = e.get(a.target_key); p = e.get(a.pred_key)
        if t is None or p is None: continue
        if a.distance(t, p) is None: continue
        targets.append(t); preds.append(p)
    if len(targets) < 2:
        return None
    # For tractability, average over all (t, p') pairs (n^2 if small;
    # otherwise random subsample).
    import random as _random
    rng = _random.Random(42)
    n = len(targets)
    if n <= 200:
        total = cnt = 0
        for t in targets:
            for p in preds:
                d = a.distance(t, p)
                if d is not None:
                    total += d; cnt += 1
        return total / cnt if cnt > 0 else None
    # Subsample: 200 targets × 200 preds → 40k pairs
    sample_t = rng.sample(targets, 200)
    sample_p = rng.sample(preds, 200)
    total = cnt = 0
    for t in sample_t:
        for p in sample_p:
            d = a.distance(t, p)
            if d is not None:
                total += d; cnt += 1
    return total / cnt if cnt > 0 else None


def compute_svec(entries: list[dict], axes: list[Axis],
                 tau_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
                 chance_mode: str = "uniform",
                 bootstrap_B: int = 0,
                 bootstrap_seed: int = 1234,
                 ) -> dict:
    """Compute S_vec on a list of per-structure dicts.

    Each entry must expose `axis.target_key` and `axis.pred_key` for each axis
    used (None means "axis inactive for this structure").

    Args:
        entries: per-structure dicts.
        axes: list of Axis specs.
        tau_grid: cumulative thresholds for F(τ).
        chance_mode: "uniform" (D_chance from each axis's analytical chance_d
                    under uniform marginals) or "empirical" (chance computed
                    from this cell's actual prompt-target and prediction
                    distributions — corrects for prompt bin imbalance).
        bootstrap_B: if > 0, percentile-bootstrap (resample structures with
                    replacement) B times for 95% CI on S_vec and per-axis d̄.
                    Returns CIs in the output dict.
        bootstrap_seed: RNG seed for bootstrap.

    Returns:
      n              number of structures contributing
      D_bar          population mean weighted per-axis distance
      D_chance       weighted chance baseline
      S_vec          max(0, 1 - D_bar / D_chance)
      per_axis_d_bar mean d_a per axis
      F_curve        cumulative fraction of structures with d_bar <= tau
      active_axes    list of axis names actually used
      S_vec_ci_95    [lo, hi] from bootstrap if requested
      per_axis_d_bar_ci  per-axis [lo, hi] from bootstrap if requested
    """
    if not entries or not axes:
        return {"n": 0}

    per_struct_d = []
    per_axis_acc = {a.name: [] for a in axes}
    # Per-structure per-axis distance (needed for bootstrap):
    per_struct_axis_d = []

    for e in entries:
        active_w, active_d = 0.0, 0.0
        struct_axis_d = {}
        for a in axes:
            t = e.get(a.target_key)
            p = e.get(a.pred_key)
            d = a.distance(t, p)
            if d is None:
                continue
            per_axis_acc[a.name].append(d)
            struct_axis_d[a.name] = (d, a.weight)
            active_w += a.weight
            active_d += a.weight * d
        if active_w > 0:
            per_struct_d.append(active_d / active_w)
            per_struct_axis_d.append(struct_axis_d)

    if not per_struct_d:
        return {"n": 0}

    n = len(per_struct_d)
    D_bar = sum(per_struct_d) / n

    active_axes = [a for a in axes if per_axis_acc[a.name]]

    # Per-axis chance baseline:
    chance_per_axis = {}
    for a in active_axes:
        if chance_mode == "empirical":
            emp = _empirical_chance_for_axis(a, entries)
            chance_per_axis[a.name] = emp if emp is not None else a.chance_d
        else:
            chance_per_axis[a.name] = a.chance_d

    total_w = sum(a.weight for a in active_axes) or 1.0
    D_chance = sum(a.weight * chance_per_axis[a.name] for a in active_axes) / total_w
    S_vec = max(0.0, 1.0 - D_bar / D_chance) if D_chance > 0 else 0.0

    per_axis_mean = {n_: (sum(v) / len(v)) if v else None
                     for n_, v in per_axis_acc.items()}

    F = {tau: sum(1 for d in per_struct_d if d <= tau + 1e-9) / n for tau in tau_grid}

    out = {
        "n": n, "D_bar": D_bar, "D_chance": D_chance, "S_vec": S_vec,
        "per_axis_d_bar": per_axis_mean,
        "F_curve": F,
        "active_axes": [a.name for a in active_axes],
        "per_axis_chance": chance_per_axis,
        "weights": {a.name: a.weight for a in active_axes},
        "chance_mode": chance_mode,
    }

    if bootstrap_B and bootstrap_B > 0:
        import random as _random
        rng = _random.Random(bootstrap_seed)
        s_vec_samples = []
        axis_d_samples = {a.name: [] for a in active_axes}
        idxs = list(range(n))
        for _ in range(bootstrap_B):
            resample = [idxs[rng.randrange(n)] for _ in range(n)]
            d_resample = [per_struct_d[i] for i in resample]
            D_bar_b = sum(d_resample) / n
            S_vec_b = max(0.0, 1.0 - D_bar_b / D_chance) if D_chance > 0 else 0.0
            s_vec_samples.append(S_vec_b)
            for a in active_axes:
                vals = [per_struct_axis_d[i].get(a.name, (None,))[0] for i in resample]
                vals = [v for v in vals if v is not None]
                if vals:
                    axis_d_samples[a.name].append(sum(vals) / len(vals))
        s_vec_samples.sort()
        lo = s_vec_samples[int(0.025 * len(s_vec_samples))]
        hi = s_vec_samples[int(0.975 * len(s_vec_samples))]
        out["S_vec_ci_95"] = [lo, hi]
        ci_axis = {}
        for ax, vals in axis_d_samples.items():
            if not vals:
                ci_axis[ax] = None; continue
            vals_sorted = sorted(vals)
            ci_axis[ax] = [vals_sorted[int(0.025 * len(vals_sorted))],
                            vals_sorted[int(0.975 * len(vals_sorted))]]
        out["per_axis_d_bar_ci_95"] = ci_axis

    return out


# ---------------------------------------------------------------------------
# Convenience: the canonical axis-set for current RECAST k7 prompts.
# Use this as a starting point and add/remove axes as you go.
# ---------------------------------------------------------------------------

DEFAULT_BINNING_SIDECAR = "/data/rkumar/code/py/dielectric/data/big_d15_autolabel_binrho_k7.binning.json"


def recast_k7_axes(include: tuple[str, ...] | None = None,
                    mode: str = "quantile",
                    binning_sidecar: str | None = None) -> list[Axis]:
    """Build the standard 5- or 6-axis list for RECAST k7 prompts.

    DEFAULT is mode="quantile" (boundary-aware; needs continuous predicted
    values). Use mode="index" if you only have bin labels.

    Per-structure entries expose target/pred keys per axis:
      mode="index":    {axis}_target (bin label),  {axis}_pred (bin label)
      mode="quantile": {axis}_target (bin label),  {axis}_pred_value (continuous)

    Args:
        include: subset of axis names to keep; None = all 6.
        mode:
            "quantile" — quantile-space distance (default; boundary-aware).
            "index"    — bin-index distance (bin labels only).
        binning_sidecar: path to autobin sidecar JSON; defaults to the
                         RECAST k7 sidecar shipped in dielectric/data/.
    """
    if binning_sidecar is None:
        binning_sidecar = DEFAULT_BINNING_SIDECAR
    tier_orders = {
        "bg":   ["vlow", "low", "mid", "high", "vhigh"],
        "eps":  ["vlow", "low", "mid", "high", "vhigh"],
        "hull": ["vlow", "low", "high", "vhigh"],
        "rho":  ["ulow", "vlow", "low", "mid", "high", "vhigh", "uhigh"],
    }

    if mode == "index":
        ordered = [
            ordered_axis("bg",   tier_orders["bg"],   "bg_target",   "bg_pred"),
            ordered_axis("eps",  tier_orders["eps"],  "eps_target",  "eps_pred"),
            ordered_axis("hull", tier_orders["hull"], "hull_target", "hull_pred"),
            ordered_axis("rho",  tier_orders["rho"],  "rho_target",  "rho_pred"),
        ]
    elif mode == "quantile":
        import json as _json
        with open(binning_sidecar) as f:
            sidecar = _json.load(f)
        sidecar_keys = {"bg": "band_gap", "eps": "nequip_eps_0",
                        "hull": "stability", "rho": "density"}
        ordered = []
        for name in ("bg", "eps", "hull", "rho"):
            edges = sidecar["properties"][sidecar_keys[name]]["edges"]
            ordered.append(quantile_axis(
                name, tier_orders[name], edges,
                target_key=f"{name}_target",
                pred_value_key=f"{name}_pred_value",
            ))
    else:
        raise ValueError(f"mode must be 'index' or 'quantile', got {mode!r}")

    all_axes = [
        jaccard_axis("elements", "elem_target",   "elem_pred",   chance_d=0.85),
        integer_axis("natoms",   "natoms_target", "natoms_pred", chance_d=0.50),
        *ordered,
    ]
    if include is None:
        return all_axes
    keep = set(include)
    return [a for a in all_axes if a.name in keep]

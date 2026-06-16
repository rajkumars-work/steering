"""Steerability measurement for text-to-structure models.

Quantifies how well a model can be steered toward target property bins on
three axes — bg (5 bins), eps_0/k (5 bins), rho (7 bins). Single-axis and
pairwise multi-axis scores. See docs/Steerability.md (or
/data/rkumar/.claude/plans/i-d-like-to-design-bubbly-pretzel.md) for the
metric definitions.

Public entry points:
    build_prompt_set(...)     # sanity-protocol prompt sampler
    score_generations(...)    # per-generation MLIP+rho eval, bin assignment
    aggregate_single_axis(...) # c_k, S_p, R_p
    aggregate_pair(...)        # c_kj, S_pq, Delta_p, Delta_q
    measure_ceiling(...)       # MLIP self-consistency on training samples
    bootstrap_ci(...)          # resampling CIs

This module is pure compute — I/O lives in scripts/run_steerability.py.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ed/ and dielectric/ live as siblings; both must be on path to import the
# screening, parser, and binning helpers.
_DIEL = Path("/data/rkumar/code/py/dielectric")
_ED = Path("/data/rkumar/code/py/ed")
for _p in (_DIEL, _ED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chem.auto_bin import label_value, load_binnings, Binning  # noqa: E402

DEFAULT_BINNING_JSON = str(
    _DIEL / "data" / "big_d15_autolabel_binrho_k7.binning.json"
)
DEFAULT_TRAIN_CSV = str(
    _DIEL / "data" / "big_d15_autolabel_binrho_k7.csv"
)
DEFAULT_VERSION_ID = "d15_binrho_k7"

# Map our short axis name → property name in the binning sidecar
AXIS_TO_BIN_PROP = {
    "bg":  "band_gap",
    "eps": "nequip_eps_0",
    "rho": "density",
}

# Map axis → token prefix in the source labels segment
AXIS_TO_TAG_PREFIX = {
    "bg": "bg-",
    "eps": "k-",
    "rho": "rho-",
    "hull": "hull-",
}


# ---------------------------------------------------------------------------
# Prompt-set construction (sanity protocol)
# ---------------------------------------------------------------------------

@dataclass
class TrainRow:
    src: str
    elements: str    # e.g. "Al Be N Si"
    natoms: int
    labels: dict[str, str]   # {"bg": "bg-vhigh", "hull": "hull-low", "k": "k-low", "rho": "rho-mid"}
    origin: str
    split: str       # train / eval / test_1k / test_100


def _parse_label_segment(seg: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in seg.split():
        for axis, pref in AXIS_TO_TAG_PREFIX.items():
            if tok.startswith(pref):
                out[axis] = tok
                break
    return out


def parse_source_prompt(src: str) -> Optional[dict]:
    """Parse a 3-segment autobin source prompt.

    Format: "elem1 elem2 ... | natoms | bg-X hull-X k-X rho-X"

    Returns dict with: elements (set[str]), natoms (int), label dict
    keyed by axis (bg/hull/eps/rho), or None if unparseable.

    Shared by `scripts/score_steerability.py` (post-screen scorer) and
    `eval.steerability.load_train_index` (sanity-protocol prompt builder).
    """
    if not isinstance(src, str):
        return None
    parts = [p.strip() for p in src.split("|")]
    if len(parts) != 3:
        return None
    elems_str, natoms_str, labels_str = parts
    try:
        natoms = int(natoms_str.split()[0])
    except (ValueError, IndexError):
        return None
    return {
        "elements": set(elems_str.split()),
        "natoms": natoms,
        "labels": _parse_label_segment(labels_str),
    }


def load_train_index(csv_path: str = DEFAULT_TRAIN_CSV) -> list[TrainRow]:
    """Load every row of the training CSV into TrainRow objects.

    Source format must be 3-segment: 'elements | natoms | bg-X hull-X k-X rho-X'.
    Rows that don't parse cleanly are dropped (no exception).
    """
    out: list[TrainRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "")
            parts = [s.strip() for s in src.split("|")]
            if len(parts) != 3:
                continue
            elems_seg, natoms_seg, label_seg = parts
            try:
                natoms = int(natoms_seg)
            except ValueError:
                continue
            labels = _parse_label_segment(label_seg)
            if not all(k in labels for k in ("bg", "hull", "eps", "rho")):
                continue
            out.append(TrainRow(
                src=src,
                elements=elems_seg,
                natoms=natoms,
                labels=labels,
                origin=row.get("origin", "mp"),
                split=row.get("label", ""),
            ))
    return out


def _build_src(row: TrainRow, override: dict[str, str]) -> str:
    """Rebuild the 3-segment source replacing labels in `override`."""
    merged = {**row.labels, **override}
    label_seg = " ".join([merged["bg"], merged["hull"], merged["eps"], merged["rho"]])
    return f"{row.elements} | {row.natoms} | {label_seg}"


def build_single_axis_prompts(
    rows: list[TrainRow],
    axis: str,
    bin_labels: list[str],
    n_per_bin: int = 25,
    seed: int = 42,
    splits: tuple[str, ...] = ("train", "eval", "test_1k", "test_100"),
) -> list[dict]:
    """Sanity-protocol prompts for a single axis.

    For each target bin t_k of `axis`, sample n_per_bin rows whose label on
    that axis is t_k. Use the row's natural labels for the other axes.

    Returns list of {prompt, origin, target_bin, axis, source_row_idx} dicts.
    """
    rng = random.Random(seed)
    pool = [r for r in rows if r.split in splits]
    by_bin: dict[str, list[TrainRow]] = {b: [] for b in bin_labels}
    for r in pool:
        if r.labels.get(axis) in by_bin:
            by_bin[r.labels[axis]].append(r)

    prompts: list[dict] = []
    for tk in bin_labels:
        cands = by_bin[tk]
        if not cands:
            print(f"WARN: no training rows with {axis}={tk}; skipping bin")
            continue
        if len(cands) >= n_per_bin:
            picks = rng.sample(cands, n_per_bin)
        else:
            print(f"WARN: only {len(cands)} rows with {axis}={tk}; "
                  f"upsampling with replacement to n={n_per_bin}")
            picks = [rng.choice(cands) for _ in range(n_per_bin)]
        for r in picks:
            prompts.append({
                "prompt": r.src,   # row already has axis = tk
                "origin": r.origin,
                "target_bin": tk,
                "axis": axis,
                "elements": r.elements,
                "natoms": r.natoms,
                "natural_labels": dict(r.labels),
            })
    return prompts


def build_pair_prompts(
    rows: list[TrainRow],
    axis_p: str,
    axis_q: str,
    bin_labels_p: list[str],
    bin_labels_q: list[str],
    n_per_cell: int = 25,
    seed: int = 42,
    splits: tuple[str, ...] = ("train", "eval", "test_1k", "test_100"),
) -> list[dict]:
    """Sanity-protocol prompts for a (p, q) cell grid.

    For each (t_p, t_q), sample rows that already match BOTH labels.
    """
    rng = random.Random(seed)
    pool = [r for r in rows if r.split in splits]
    prompts: list[dict] = []
    for tp in bin_labels_p:
        for tq in bin_labels_q:
            cands = [r for r in pool
                     if r.labels.get(axis_p) == tp and r.labels.get(axis_q) == tq]
            if not cands:
                print(f"WARN: no rows with {axis_p}={tp} & {axis_q}={tq}")
                continue
            if len(cands) >= n_per_cell:
                picks = rng.sample(cands, n_per_cell)
            else:
                print(f"WARN: {axis_p}={tp} & {axis_q}={tq}: "
                      f"{len(cands)} cands, upsampling to {n_per_cell}")
                picks = [rng.choice(cands) for _ in range(n_per_cell)]
            for r in picks:
                prompts.append({
                    "prompt": r.src,
                    "origin": r.origin,
                    "target_bin_p": tp,
                    "target_bin_q": tq,
                    "axis_p": axis_p,
                    "axis_q": axis_q,
                    "elements": r.elements,
                    "natoms": r.natoms,
                    "natural_labels": dict(r.labels),
                })
    return prompts


# ---------------------------------------------------------------------------
# Property prediction on generated atoms
# ---------------------------------------------------------------------------

def density_g_per_cm3(atoms) -> float:
    """g/cm^3. Mirrors dielectric/validity/structural.py:197."""
    vol = float(atoms.get_volume())
    if vol <= 0:
        return float("nan")
    return float(np.sum(atoms.get_masses())) * 1.66054 / vol


def predict_bg_eps(atoms, bg_calc, eps_calc) -> tuple[float, float]:
    """Predict bandgap and dielectric via nequip MLIPs. Returns (bg, eps).

    NaN on any failure.
    """
    try:
        a1 = atoms.copy(); a1.calc = bg_calc
        bg = float(a1.get_potential_energy())
    except Exception:
        bg = float("nan")
    try:
        a2 = atoms.copy(); a2.calc = eps_calc
        eps = float(a2.get_potential_energy())
    except Exception:
        eps = float("nan")
    return bg, eps


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def steerability_score(macro_compliance: float, K: int) -> float:
    """S = max(0, (c_bar - 1/K) / (1 - 1/K))."""
    if K <= 1:
        return float("nan")
    return max(0.0, (macro_compliance - 1.0 / K) / (1.0 - 1.0 / K))


def normalized_mutual_information(
    pred_bins: list[Optional[str]],
    target_bins: list[str],
    bin_labels: list[str],
) -> float:
    """NMI(predicted; prompted) / log(K). 0 if either side has no entropy.

    Predictions equal to None or unknown labels are dropped.
    """
    pairs = [(p, t) for p, t in zip(pred_bins, target_bins)
             if p in bin_labels and t in bin_labels]
    if not pairs:
        return 0.0
    K = len(bin_labels)
    n = len(pairs)
    label_to_idx = {b: i for i, b in enumerate(bin_labels)}
    joint = np.zeros((K, K))
    for p, t in pairs:
        joint[label_to_idx[p], label_to_idx[t]] += 1
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    pxy = joint
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.where(pxy > 0, pxy / (px * py), 1.0))
    mi = float(np.sum(np.where(pxy > 0, pxy * log_term, 0.0)))
    if K <= 1 or mi <= 0:
        return 0.0
    return float(mi / math.log(K))


@dataclass
class SingleAxisAggregate:
    axis: str
    K: int
    bin_labels: list[str]
    n_total: int
    n_parsed: int
    parse_rate: float
    per_bin_compliance: dict[str, float]   # t_k → c_k
    per_bin_n: dict[str, int]
    macro_compliance: float
    steerability: float    # S_p
    resolution: float      # R_p
    confusion: list[list[int]] = field(default_factory=list)


def aggregate_single_axis(
    per_prompt: list[dict],
    axis: str,
    bin_labels: list[str],
) -> SingleAxisAggregate:
    """Aggregate a list of per-prompt dicts (output of score_generations)."""
    pred_key = f"pred_bin_{axis}"
    rows = [r for r in per_prompt if r.get("axis") == axis]
    n_total = len(rows)
    parsed = [r for r in rows if r.get("parse_ok")]
    n_parsed = len(parsed)
    K = len(bin_labels)

    per_bin_n: dict[str, int] = {b: 0 for b in bin_labels}
    per_bin_hit: dict[str, int] = {b: 0 for b in bin_labels}
    label_to_idx = {b: i for i, b in enumerate(bin_labels)}
    confusion = [[0] * K for _ in range(K)]

    pred_list: list[Optional[str]] = []
    target_list: list[str] = []
    for r in parsed:
        tk = r["target_bin"]
        pk = r.get(pred_key)
        per_bin_n[tk] += 1
        if pk == tk:
            per_bin_hit[tk] += 1
        if pk in label_to_idx and tk in label_to_idx:
            confusion[label_to_idx[pk]][label_to_idx[tk]] += 1
        pred_list.append(pk)
        target_list.append(tk)

    per_bin_c = {
        b: (per_bin_hit[b] / per_bin_n[b]) if per_bin_n[b] > 0 else float("nan")
        for b in bin_labels
    }
    valid = [c for c in per_bin_c.values() if not math.isnan(c)]
    macro = float(np.mean(valid)) if valid else float("nan")
    S = steerability_score(macro, K) if not math.isnan(macro) else float("nan")
    R = normalized_mutual_information(pred_list, target_list, bin_labels)

    return SingleAxisAggregate(
        axis=axis,
        K=K,
        bin_labels=list(bin_labels),
        n_total=n_total,
        n_parsed=n_parsed,
        parse_rate=(n_parsed / n_total) if n_total > 0 else 0.0,
        per_bin_compliance=per_bin_c,
        per_bin_n=per_bin_n,
        macro_compliance=macro,
        steerability=S,
        resolution=R,
        confusion=confusion,
    )


@dataclass
class PairAggregate:
    axis_p: str
    axis_q: str
    K_p: int
    K_q: int
    bin_labels_p: list[str]
    bin_labels_q: list[str]
    n_total: int
    n_parsed: int
    parse_rate: float
    cell_compliance: dict[str, dict[str, float]]   # tp → tq → c
    cell_n: dict[str, dict[str, int]]
    macro_compliance: float
    steerability: float
    macro_p_under_pair: float
    macro_q_under_pair: float


def aggregate_pair(
    per_prompt: list[dict],
    axis_p: str,
    axis_q: str,
    bin_labels_p: list[str],
    bin_labels_q: list[str],
) -> PairAggregate:
    rows = [r for r in per_prompt
            if r.get("axis_p") == axis_p and r.get("axis_q") == axis_q]
    n_total = len(rows)
    parsed = [r for r in rows if r.get("parse_ok")]
    n_parsed = len(parsed)

    K_p, K_q = len(bin_labels_p), len(bin_labels_q)
    cell_n: dict[str, dict[str, int]] = {tp: {tq: 0 for tq in bin_labels_q} for tp in bin_labels_p}
    cell_hit: dict[str, dict[str, int]] = {tp: {tq: 0 for tq in bin_labels_q} for tp in bin_labels_p}
    p_hit_total = 0
    q_hit_total = 0
    parsed_in_grid = 0

    for r in parsed:
        tp = r["target_bin_p"]
        tq = r["target_bin_q"]
        pp = r.get(f"pred_bin_{axis_p}")
        pq = r.get(f"pred_bin_{axis_q}")
        if tp not in cell_n or tq not in cell_n[tp]:
            continue
        parsed_in_grid += 1
        cell_n[tp][tq] += 1
        if pp == tp and pq == tq:
            cell_hit[tp][tq] += 1
        if pp == tp:
            p_hit_total += 1
        if pq == tq:
            q_hit_total += 1

    cell_c: dict[str, dict[str, float]] = {
        tp: {tq: (cell_hit[tp][tq] / cell_n[tp][tq]) if cell_n[tp][tq] > 0 else float("nan")
             for tq in bin_labels_q}
        for tp in bin_labels_p
    }
    valid_cells = [v for d in cell_c.values() for v in d.values() if not math.isnan(v)]
    macro = float(np.mean(valid_cells)) if valid_cells else float("nan")
    K = K_p * K_q
    S = steerability_score(macro, K) if not math.isnan(macro) else float("nan")
    macro_p = (p_hit_total / parsed_in_grid) if parsed_in_grid else float("nan")
    macro_q = (q_hit_total / parsed_in_grid) if parsed_in_grid else float("nan")

    return PairAggregate(
        axis_p=axis_p, axis_q=axis_q,
        K_p=K_p, K_q=K_q,
        bin_labels_p=list(bin_labels_p),
        bin_labels_q=list(bin_labels_q),
        n_total=n_total,
        n_parsed=n_parsed,
        parse_rate=(n_parsed / n_total) if n_total > 0 else 0.0,
        cell_compliance=cell_c,
        cell_n=cell_n,
        macro_compliance=macro,
        steerability=S,
        macro_p_under_pair=macro_p,
        macro_q_under_pair=macro_q,
    )


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_S_ci(
    per_prompt_for_axis: list[dict],
    axis: str,
    bin_labels: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Stratified-by-target-bin bootstrap of S_p. Returns (lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    by_bin: dict[str, list[dict]] = {b: [] for b in bin_labels}
    for r in per_prompt_for_axis:
        tk = r.get("target_bin")
        if tk in by_bin:
            by_bin[tk].append(r)

    K = len(bin_labels)
    samples = []
    for _ in range(n_boot):
        c_macro = []
        for b in bin_labels:
            pool = by_bin[b]
            if not pool:
                continue
            idx = rng.integers(0, len(pool), size=len(pool))
            hits = sum(1 for i in idx
                       if pool[i].get("parse_ok") and pool[i].get(f"pred_bin_{axis}") == b)
            c_macro.append(hits / len(pool))
        if c_macro:
            samples.append(steerability_score(float(np.mean(c_macro)), K))
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------------------------------------------------------------------------
# Intrinsic ceiling: training-row retrieval oracle through the same pipeline
# ---------------------------------------------------------------------------

def measure_ceiling_for_axis(
    rows: list[TrainRow],
    axis: str,
    bin_labels: list[str],
    n_per_bin: int,
    score_fn,
    seed: int = 7,
    splits: tuple[str, ...] = ("eval", "test_1k", "test_100"),
) -> dict:
    """For each bin, pull n_per_bin rows from eval/test, parse their target,
    and run the eval pipeline. Macro-compliance = ceiling.

    `score_fn(rows_to_score) -> list[per_prompt dict]` — same shape as
    score_generations, but reading the row's `target` from CSV not generated.
    """
    rng = random.Random(seed)
    pool = [r for r in rows if r.split in splits]
    by_bin: dict[str, list[TrainRow]] = {b: [] for b in bin_labels}
    for r in pool:
        if r.labels.get(axis) in by_bin:
            by_bin[r.labels[axis]].append(r)

    selected: list[TrainRow] = []
    sel_meta: list[dict] = []
    for tk in bin_labels:
        cands = by_bin[tk]
        if not cands:
            continue
        picks = (rng.sample(cands, n_per_bin) if len(cands) >= n_per_bin
                 else [rng.choice(cands) for _ in range(n_per_bin)])
        for r in picks:
            selected.append(r)
            sel_meta.append({"axis": axis, "target_bin": tk, "origin": r.origin})

    return {"selected": selected, "meta": sel_meta}


__all__ = [
    "AXIS_TO_BIN_PROP",
    "AXIS_TO_TAG_PREFIX",
    "TrainRow",
    "load_train_index",
    "parse_source_prompt",
    "build_single_axis_prompts",
    "build_pair_prompts",
    "density_g_per_cm3",
    "predict_bg_eps",
    "steerability_score",
    "normalized_mutual_information",
    "SingleAxisAggregate",
    "aggregate_single_axis",
    "PairAggregate",
    "aggregate_pair",
    "bootstrap_S_ci",
    "measure_ceiling_for_axis",
    "DEFAULT_BINNING_JSON",
    "DEFAULT_TRAIN_CSV",
    "DEFAULT_VERSION_ID",
]

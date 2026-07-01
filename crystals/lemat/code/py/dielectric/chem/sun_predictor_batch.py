"""Batch-level prompt inspector with OOD flag (Option A).

Given a batch of prompts (`sources.txt` style: `Elem1 Elem2 | natoms | ...`),
produce:

  1. Cell-aggregate predictions for SUN, stable, MSUN, novel-vs-LeMat
     using the calibrated single-feature heuristic
     (`SUN_Prediction.md` §3.1, lookup version).

  2. An OOD warning if the batch's `(n_elements, natoms)` joint
     distribution falls outside the calibration envelope. The flag
     compares each prompt against the 754-row calibration histogram;
     if too many prompts land in low-support bins, the predictions
     are likely to be off (e.g., the B.0 controlled holdout failed
     the formula because its `n_elements=2, natoms≈48` corner has
     ~zero calibration support).

  3. Per-prompt P(SUN/stable/...) values for inspection (with the
     standard "per-prompt variance is wide" disclaimer).

Calibration: `eval/sun_drivers/per_structure_features_v3.csv`
(754 rows used in Phase A2 fit + B.1.3 CV).

CLI:
    python -m chem.sun_predictor_batch <sources.txt>
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path("/data/rkumar/code/py/dielectric")
CALIBRATION_CSV = ROOT / "eval/sun_drivers/per_structure_features_v3.csv"
LEMAT_STATS_CACHE = ROOT / "eval/sun_drivers/lemat_element_stats.pkl"

# Coefficients from SUN_Prediction.md §3.1 (lookup version on log(1+n_lemat))
COEFS_LOOKUP = {
    "stable_or_meta": (-0.117, -1.102),
    "novel_vs_lemat": (+1.299, -0.582),
    "is_sun":         (-0.549, -1.416),
    "is_msun":        (-2.018, -0.405),
}

# Bin edges for the calibration histogram. natoms bins were chosen to
# carve out the failure corners surfaced by B.0 (small n_elements at
# large natoms — the 2-element 48-atom region had zero calibration support).
NATOMS_BINS = [0, 5, 10, 15, 20, 30, 40, 60, 1000]
NATOMS_LABELS = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-60", "60+"]
NELEM_VALUES = [1, 2, 3, 4, 5, 6, 7]   # >=7 lumped at the end


# ---------- prompt parsing ----------

def parse_prompt_line(line: str) -> tuple[frozenset[str], int] | None:
    """Parse `'Fe O | 8 | bg-low ...'` → (frozenset({'Fe','O'}), 8)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 2:
        return None
    elements = frozenset(parts[0].split())
    try:
        natoms = int(parts[1])
    except ValueError:
        return None
    if not elements or natoms < 1:
        return None
    return elements, natoms


def parse_prompts(path: str | Path) -> list[tuple[frozenset[str], int]]:
    out = []
    for ln in Path(path).read_text().splitlines():
        p = parse_prompt_line(ln)
        if p is not None:
            out.append(p)
    return out


# ---------- calibration histogram ----------

def _natoms_bin_idx(natoms: int) -> int:
    for i, edge in enumerate(NATOMS_BINS[1:]):
        if natoms <= edge:
            return i
    return len(NATOMS_LABELS) - 1


def _nelem_bin_idx(n_elem: int) -> int:
    if n_elem <= 1:
        return 0
    if n_elem >= 7:
        return 6
    return n_elem - 1


@dataclass
class CalibrationGrid:
    """2D histogram of (n_elements, natoms_bin) over the calibration set."""
    counts: np.ndarray              # shape (7, len(NATOMS_LABELS))
    n_total: int

    @classmethod
    def from_csv(cls, csv_path: Path = CALIBRATION_CSV) -> "CalibrationGrid":
        df = pd.read_csv(csv_path)
        # use only validity-passed rows (the predictor was fit on these)
        if "genbench_valid" in df.columns:
            df = df[df.genbench_valid == True].copy()
        counts = np.zeros((len(NELEM_VALUES), len(NATOMS_LABELS)), dtype=int)
        for _, row in df.iterrows():
            i = _nelem_bin_idx(int(row.n_elements))
            j = _natoms_bin_idx(int(row.natoms_prompt))
            counts[i, j] += 1
        return cls(counts=counts, n_total=int(counts.sum()))

    def support(self, n_elements: int, natoms: int) -> int:
        i = _nelem_bin_idx(n_elements)
        j = _natoms_bin_idx(natoms)
        return int(self.counts[i, j])

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.counts,
            index=[f"n_elem={v if v < 7 else '7+'}" for v in NELEM_VALUES],
            columns=NATOMS_LABELS,
        )


# ---------- OOD scoring ----------

# Thresholds calibrated against the two existing validation batches:
#   - n=1000 HIGH/MEDIUM/LOW (in-envelope, formula passed)
#   - B.0 (out-of-envelope, formula failed)
# A prompt's bin is "low-support" if calibration count <= LOW_SUPPORT_CUTOFF.
# A batch warns if too high a fraction of prompts land in low-support bins.
LOW_SUPPORT_CUTOFF = 5
WARN_FRAC_LOW_SUPPORT = 0.20      # >20% of batch in low-support bins → warn
WARN_FRAC_ZERO_SUPPORT = 0.05     # >5% in zero-support bins → warn (harder rule)


@dataclass
class BatchOODReport:
    n_prompts: int
    frac_low_support: float
    frac_zero_support: float
    bin_breakdown: pd.DataFrame
    warning: str | None             # None if in-envelope

    def is_in_envelope(self) -> bool:
        return self.warning is None


def score_batch_ood(prompts: list[tuple[frozenset[str], int]],
                    grid: CalibrationGrid) -> BatchOODReport:
    """Compute calibration support for each prompt in batch."""
    rows = []
    n_low = n_zero = 0
    for elems, natoms in prompts:
        s = grid.support(len(elems), natoms)
        rows.append({"n_elements": len(elems), "natoms": natoms, "calib_support": s})
        if s == 0:
            n_zero += 1
            n_low += 1
        elif s <= LOW_SUPPORT_CUTOFF:
            n_low += 1
    n = len(prompts)
    if n == 0:
        return BatchOODReport(0, 0.0, 0.0, pd.DataFrame(rows), "empty batch")

    frac_low = n_low / n
    frac_zero = n_zero / n
    df = pd.DataFrame(rows)

    warn = None
    if frac_zero > WARN_FRAC_ZERO_SUPPORT:
        warn = (
            f"OOD warning: {frac_zero:.0%} of batch lies in (n_elements, natoms) "
            f"bins with zero calibration support. Predictions for these prompts "
            f"are extrapolations; cell-aggregate Combined estimate may invert "
            f"(cf. B.0 holdout failure)."
        )
    elif frac_low > WARN_FRAC_LOW_SUPPORT:
        warn = (
            f"OOD warning: {frac_low:.0%} of batch lies in (n_elements, natoms) "
            f"bins with ≤{LOW_SUPPORT_CUTOFF} calibration rows. Cell-aggregate "
            f"prediction is unverified for this distribution."
        )
    return BatchOODReport(
        n_prompts=n, frac_low_support=frac_low, frac_zero_support=frac_zero,
        bin_breakdown=df, warning=warn,
    )


# ---------- batch predictions ----------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_lemat_stats = None
def _load_lemat_stats():
    global _lemat_stats
    if _lemat_stats is None:
        _lemat_stats = pickle.load(open(LEMAT_STATS_CACHE, "rb"))
    return _lemat_stats


def predict_targets_lookup(elems: frozenset[str], natoms: int) -> dict[str, float]:
    """Per-prompt P(target) using log(1 + n_lemat) lookup version."""
    stats = _load_lemat_stats()
    n_lemat = stats["elemset_natoms_count"].get((elems, int(natoms)), 0)
    x = math.log1p(n_lemat)
    return {t: _sigmoid(a + b * x) for t, (a, b) in COEFS_LOOKUP.items()}


def predict_batch(prompts: list[tuple[frozenset[str], int]]) -> pd.DataFrame:
    rows = []
    for elems, natoms in prompts:
        p = predict_targets_lookup(elems, natoms)
        rows.append({
            "n_elements": len(elems), "natoms": natoms,
            "P_SUN": p["is_sun"],
            "P_stable": p["stable_or_meta"],
            "P_MSUN": p["is_msun"],
            "P_novel_lemat": p["novel_vs_lemat"],
        })
    return pd.DataFrame(rows)


# ---------- top-level ----------

def inspect_batch(prompts: list[tuple[frozenset[str], int]]):
    grid = CalibrationGrid.from_csv()
    ood = score_batch_ood(prompts, grid)
    preds = predict_batch(prompts)
    cell_means = preds[["P_SUN", "P_stable", "P_MSUN", "P_novel_lemat"]].mean()
    return {"ood": ood, "preds": preds, "cell_means": cell_means, "grid": grid}


def _format_report(result, prompts_path: str | None = None) -> str:
    ood = result["ood"]
    cm = result["cell_means"]
    lines = []
    if prompts_path:
        lines.append(f"Batch: {prompts_path}")
    lines.append(f"n_prompts: {ood.n_prompts}")
    lines.append("")
    lines.append("Cell-aggregate predictions (caveat: per-prompt variance is wide,")
    lines.append("and these are valid only if the OOD flag below is clear):")
    lines.append(f"  P(SUN)            ≈ {cm.P_SUN:.3f}")
    lines.append(f"  P(stable+meta)    ≈ {cm.P_stable:.3f}")
    lines.append(f"  P(MSUN)           ≈ {cm.P_MSUN:.3f}")
    lines.append(f"  P(novel-vs-LeMat) ≈ {cm.P_novel_lemat:.3f}")
    lines.append("")
    lines.append(f"OOD check: frac_low_support={ood.frac_low_support:.1%}, "
                 f"frac_zero_support={ood.frac_zero_support:.1%}")
    if ood.warning:
        lines.append("  ⚠ " + ood.warning)
    else:
        lines.append("  ✓ batch in calibration envelope; predictions trustworthy "
                     "at the cell-aggregate level.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sources", help="path to sources.txt")
    ap.add_argument("--show-prompts", action="store_true",
                    help="also print per-prompt predictions")
    args = ap.parse_args(argv)
    prompts = parse_prompts(args.sources)
    if not prompts:
        print(f"no parseable prompts in {args.sources}", file=sys.stderr)
        return 1
    result = inspect_batch(prompts)
    print(_format_report(result, args.sources))
    if args.show_prompts:
        print("\nPer-prompt predictions:")
        print(result["preds"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SUN/Combined predictor from prompt features.

Calibrated 2026-05-03 against 9 cells (754 valid structures total). The
law is:

    P(Combined) ≈ A + B × ln( natoms / (elemset_natoms_count + 1) )

with A = -0.079, B = 0.232.

`elemset_natoms_count` = number of LeMat-Bulk PBE-compatible structures
whose element set is exactly `elements` and whose natoms is exactly
`natoms`. Looked up from
/data/assets/datasets/lematbulk/lematbulk_compatible_pbe.parquet via
the cached counter in eval/sun_drivers/lemat_element_stats.pkl.

LOO MAE: 0.049 across 9 cells; R² = 0.955. Predicts cell-mean Combined
within ±5 pp on average. Below this MAE threshold the law is treated
as a calibrated cell-mean predictor; for individual prompts, treat the
output as a probability with substantial per-prompt variance.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable

# Coefficients fit on cell-aggregate data (9 cells × ~90 prompts each)
_INTERCEPT = -0.079
_SLOPE_LOG_RATIO = 0.232

_STATS_CACHE = Path(
    "/data/rkumar/code/py/dielectric/eval/sun_drivers/lemat_element_stats.pkl"
)
_stats_loaded = None


def _load_stats():
    global _stats_loaded
    if _stats_loaded is None:
        _stats_loaded = pickle.load(open(_STATS_CACHE, "rb"))
    return _stats_loaded


def predict_combined(elements: Iterable[str], natoms: int) -> float:
    """Predicted P(Combined SUN+MSUN) for a single prompt.

    Parameters
    ----------
    elements : iterable of element symbols (e.g. ['Fe', 'O'])
    natoms   : int — total atoms in the prompt's target structure

    Returns
    -------
    float in [0, 1] — clipped predicted Combined rate.

    Notes
    -----
    This is a cell-mean predictor calibrated to within ±5 pp on average
    over 9 prompt cells (LOO MAE = 0.049). Per-prompt prediction
    variance is high — use only for population-level estimates.
    """
    stats = _load_stats()
    elemset = frozenset(elements)
    if natoms < 1:
        raise ValueError("natoms must be >= 1")

    elemset_natoms_count = stats["elemset_natoms_count"].get((elemset, int(natoms)), 0)
    log_ratio = math.log(int(natoms) / (elemset_natoms_count + 1))
    p = _INTERCEPT + _SLOPE_LOG_RATIO * log_ratio
    return max(0.0, min(1.0, p))


def predict_combined_batch(prompts):
    """Vectorised: prompts is iterable of (elements, natoms) pairs."""
    return [predict_combined(e, n) for e, n in prompts]


if __name__ == "__main__":
    # Demo
    cases = [
        (["Fe", "O"], 5),
        (["Fe", "O"], 20),
        (["Fe", "O", "Si"], 12),
        (["Fe", "Co", "Ni", "O"], 12),
        (["Sm", "Eu", "Gd", "Yb"], 16),  # rare lanthanides
    ]
    for elems, n in cases:
        p = predict_combined(elems, n)
        print(f"  elements={elems}, natoms={n} → P(Combined) ≈ {p:.3f}")

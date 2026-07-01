"""Rough predictor: given (elements, natoms), estimate the probability
that a generated structure will be stable, novel-vs-LeMat-Bulk, SUN, MSUN.

Built from per-structure features extracted in the SUN-drivers study
(eval/sun_drivers/predictability_study.json). Backed by cached
distribution stats:
  - eval/sun_drivers/train_chemistry_stats.pkl  (training-set coverage)
  - eval/sun_drivers/lemat_element_stats.pkl    (LeMat-Bulk coverage)

Usage:
    from chem.output_predictor import OutputPredictor
    p = OutputPredictor()
    print(p.predict(["Li","Co","O"], natoms=8))
    # {'stable': 0.18, 'novel_vs_lemat': 0.42, 'sun': 0.07, 'msun': 0.10}

Design notes:
- Causally-valid feature sets used per target:
    stable          ← train-side count features only
    novel_vs_lemat  ← lemat-side count features only
    sun, msun       ← both sides (predictive, not causal)
- Logistic-regression coefficients are fit once at module load on the
  per-structure dataframe; the same model that the analysis uses.
- Calibration: AUC ≈ 0.73 (stable), 0.72 (novel), 0.81 (sun)
  on leave-one-cell-out CV from 9 cells × ~90 structures each.

This is a rough estimator. Do NOT treat the probabilities as well-
calibrated for individual prompts — they're calibrated at the cell
(prompt-set) level (LOO MAE ≈ 0.10–0.17 on cell rates).
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SUN_DIR = ROOT / "eval/sun_drivers"
DF_CSV = SUN_DIR / "per_structure_features_v3.csv"
TRAIN_STATS = SUN_DIR / "train_chemistry_stats.pkl"
LEMAT_STATS = SUN_DIR / "lemat_element_stats.pkl"


# Feature sets (causally valid per target)
FEATURES_STABLE = [
    "natoms_prompt",
    "train_tup_count", "train_set_count",
    "train_set_stable_frac", "train_elem_stable_frac",
]
FEATURES_NOVEL = [
    "natoms_prompt",
    "prompt_geomean_element_freq",
    "prompt_log_sum_element_freq",
    "prompt_elemset_natoms_count",
]
FEATURES_SUN = [  # predictive; not strictly causal
    "natoms_prompt",
    "prompt_elemset_natoms_count",
    "prompt_log_sum_element_freq",
    "train_tup_count", "train_set_count",
    "train_elem_stable_frac",
]
FEATURES_MSUN = FEATURES_SUN  # same predictors

TARGETS = {
    "stable": ("stable_or_meta", FEATURES_STABLE),
    "novel_vs_lemat": ("novel_vs_lemat", FEATURES_NOVEL),
    "sun": ("is_sun", FEATURES_SUN),
    "msun": ("is_msun", FEATURES_MSUN),
}


def _safe_log1p(x):
    return math.log1p(max(0.0, float(x)))


class OutputPredictor:
    def __init__(self, stats_only: bool = False):
        self._train_stats = None
        self._lemat_stats = None
        self._models = None
        self._feature_medians = None  # for fallback when feature missing
        if not stats_only:
            self.fit_or_load()

    # -------------------- stats loaders --------------------
    @property
    def train_stats(self):
        if self._train_stats is None:
            with open(TRAIN_STATS, "rb") as f:
                self._train_stats = pickle.load(f)
        return self._train_stats

    @property
    def lemat_stats(self):
        if self._lemat_stats is None:
            with open(LEMAT_STATS, "rb") as f:
                self._lemat_stats = pickle.load(f)
        return self._lemat_stats

    # -------------------- prompt → features --------------------
    def features_from_prompt(self, elements: Iterable[str], natoms: int) -> dict[str, float]:
        elements = sorted(set(e for e in elements if e))
        elem_set = frozenset(elements)

        ts = self.train_stats
        ls = self.lemat_stats

        # Training-side features
        train_tup_count = ts["elemset_natoms_count_train"].get((elem_set, natoms), 0)
        train_set_count = ts["elemset_count_train"].get(elem_set, 0)
        n_train_rows = ts["n_train_rows"]

        # Stable fraction priors (with smoothing for sparse sets)
        train_set_stable = ts["elemset_stable_train"].get(elem_set, 0)
        train_set_stable_frac = (
            train_set_stable / train_set_count if train_set_count > 0 else 0.0
        )
        # Element-wise stable fraction prior (mean of per-element stable rates)
        elem_stable_fracs = []
        for e in elements:
            ec = ts["element_count_train"].get(e, 0)
            es = ts["element_stable_train"].get(e, 0)
            if ec > 0:
                elem_stable_fracs.append(es / ec)
        train_elem_stable_frac = (
            sum(elem_stable_fracs) / len(elem_stable_fracs) if elem_stable_fracs else 0.0
        )

        # LeMat-side features
        prompt_elemset_lemat_count = ls["elemset_count"].get(elem_set, 0)
        prompt_elemset_natoms_count = ls["elemset_natoms_count"].get((elem_set, natoms), 0)
        # Element-frequency aggregates
        elem_freqs = [ls["element_count"].get(e, 1) for e in elements]
        prompt_min_element_freq = min(elem_freqs) if elem_freqs else 1
        prompt_geomean_element_freq = (
            math.exp(sum(math.log(f) for f in elem_freqs) / len(elem_freqs))
            if elem_freqs else 1.0
        )
        prompt_log_sum_element_freq = sum(math.log(f) for f in elem_freqs) if elem_freqs else 0.0

        return {
            "natoms_prompt": natoms,
            "n_atoms_actual": natoms,  # not known until generation
            "train_tup_count": train_tup_count,
            "train_set_count": train_set_count,
            "train_set_stable_frac": train_set_stable_frac,
            "train_elem_stable_frac": train_elem_stable_frac,
            "prompt_min_element_freq": prompt_min_element_freq,
            "prompt_geomean_element_freq": prompt_geomean_element_freq,
            "prompt_log_sum_element_freq": prompt_log_sum_element_freq,
            "prompt_elemset_lemat_count": prompt_elemset_lemat_count,
            "prompt_elemset_natoms_count": prompt_elemset_natoms_count,
        }

    @staticmethod
    def _transform(feature_row: dict[str, float], feat_names: list[str]) -> np.ndarray:
        """Apply log1p to count-like features for stable fitting."""
        x = []
        for name in feat_names:
            v = float(feature_row.get(name, 0.0) or 0.0)
            if "count" in name or name == "natoms_prompt" or name == "n_atoms_actual" or "log_sum" in name:
                x.append(_safe_log1p(v))
            else:
                x.append(v)
        return np.array(x, dtype=float)

    # -------------------- fit / load models --------------------
    def fit_or_load(self):
        """Fit all four models from the cached dataframe. Cheap (<1s)."""
        df = pd.read_csv(DF_CSV)
        df = df[df["genbench_valid"].fillna(False).astype(bool)].copy()
        df = df.dropna(subset=[v[0] for v in TARGETS.values()])
        for tname, (col, _) in TARGETS.items():
            df[col] = df[col].astype(int)

        models = {}
        for tname, (col, feats) in TARGETS.items():
            # Build feature matrix with the same log1p transform
            X = np.array([
                self._transform(row.to_dict(), feats) for _, row in df[feats].iterrows()
            ])
            y = df[col].values
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
            models[tname] = {
                "feats": feats,
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
            }
        self._models = models
        self._feature_medians = {f: float(df[f].median()) for f in {f for _, fs in TARGETS.values() for f in fs}}

    # -------------------- predict --------------------
    def predict(self, elements: Iterable[str], natoms: int) -> dict[str, float]:
        if self._models is None:
            self.fit_or_load()
        feats = self.features_from_prompt(elements, natoms)
        out = {}
        for tname, info in self._models.items():
            x = self._transform(feats, info["feats"])
            logit = float(np.dot(x, info["coef"]) + info["intercept"])
            out[tname] = 1.0 / (1.0 + math.exp(-logit))
        return out

    def predict_with_features(self, elements: Iterable[str], natoms: int) -> dict:
        """Like predict() but also returns the feature row used."""
        feats = self.features_from_prompt(elements, natoms)
        preds = self.predict(elements, natoms)
        return {"prediction": preds, "features": feats}


def _demo():
    p = OutputPredictor()
    print("=" * 72)
    print("OutputPredictor demo")
    print("=" * 72)

    cases = [
        ("Li Co O",         16,  "common 3-element MP-style oxide"),
        ("K F",              4,  "small halide (WBM-style)"),
        ("Ce Os Rh Ge",     20,  "alex-style transition-metal alloy"),
        ("H Li N O",        12,  "common 4-element"),
        ("Cs Eu I O",       10,  "rare-earth chemistry"),
        ("Si",               8,  "single-element silicon"),
    ]
    print(f"\n{'elements':<22} {'natoms':>6}  {'stable':>7} {'novel':>7} {'sun':>6} {'msun':>6}  {'tup_lemat':>10} {'tup_train':>10}")
    print("-" * 100)
    for elements_str, n, note in cases:
        elements = elements_str.split()
        out = p.predict_with_features(elements, n)
        pred = out["prediction"]
        feats = out["features"]
        print(f"{elements_str:<22} {n:>6}  {pred['stable']:>7.3f} {pred['novel_vs_lemat']:>7.3f} "
              f"{pred['sun']:>6.3f} {pred['msun']:>6.3f}  "
              f"{int(feats['prompt_elemset_natoms_count']):>10} {int(feats['train_tup_count']):>10}  "
              f"{note}")


if __name__ == "__main__":
    _demo()

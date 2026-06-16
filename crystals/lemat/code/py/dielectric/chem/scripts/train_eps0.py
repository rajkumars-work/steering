#!/usr/bin/env python3
"""Train baseline and outlier-weighted eps_0 surrogates and compare performance."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from chem.surrogates.soap import (
    build_soap_descriptor,
    composition_string,
    featurize_structures,
    load_structures_from_extxyz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--extxyz",
        default="/data/assets/atlas/dielectrics/datasets/dft_bandgap_eps0_epsinf.extxyz",
        help="Training dataset (extxyz with dft_eps_0).",
    )
    p.add_argument(
        "--outdir",
        default="data/checkpoints/eps0_outlier_weighted",
        help="Output directory for models and metadata.",
    )
    p.add_argument(
        "--cache",
        default="data/soap_features_dft_eps_0.pkl",
        help="Feature cache path.",
    )
    p.add_argument("--n-jobs", type=int, default=os.cpu_count())
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature generation to limit memory usage.",
    )
    p.add_argument(
        "--feature-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Feature dtype to reduce memory footprint.",
    )
    p.add_argument("--soap-r-cut", type=float, default=4.0)
    p.add_argument("--soap-n-max", type=int, default=4)
    p.add_argument("--soap-l-max", type=int, default=3)
    p.add_argument(
        "--soap-sparse",
        action="store_true",
        help="Use sparse SOAP descriptor (lower memory, may be slower).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.01,
        help="Quantile for defining outliers (e.g. 0.01 = 1% tails).",
    )
    p.add_argument(
        "--outlier-weight",
        type=float,
        default=5.0,
        help="Sample weight for outliers.",
    )
    p.add_argument(
        "--log-target",
        action="store_true",
        help="Train on log1p(target) to reduce heavy-tail effects.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost device.",
    )
    p.add_argument(
        "--tree-method",
        default=None,
        help="XGBoost tree_method (e.g. hist, gpu_hist). Defaults based on --device.",
    )
    return p.parse_args()


def _meminfo_available_bytes() -> int | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    try:
        text = meminfo_path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def build_features(
    structures,
    cache_path: Path,
    n_jobs: int,
    batch_size: int,
    dtype: str,
    soap_r_cut: float,
    soap_n_max: int,
    soap_l_max: int,
    soap_sparse: bool,
) -> np.ndarray:
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            cache = None
        if isinstance(cache, dict) and "X" in cache:
            X = cache["X"]
            if hasattr(X, "shape") and X.shape[0] == len(structures):
                return X
        if isinstance(cache, dict) and "X_path" in cache:
            mmap_path = Path(cache["X_path"])
            if mmap_path.exists():
                shape = tuple(cache.get("shape", (len(structures), 0)))
                dtype = cache.get("dtype", dtype)
                if shape[0] == len(structures):
                    return np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)
    # Cache missing or invalid -> compute fresh.
    species = sorted(list({Z for atoms in structures for Z in atoms.get_atomic_numbers()}))
    soap = build_soap_descriptor(
        species,
        r_cut=soap_r_cut,
        n_max=soap_n_max,
        l_max=soap_l_max,
        sparse=soap_sparse,
    )
    mmap_path = cache_path.with_suffix(cache_path.suffix + ".mmap")
    X = featurize_structures(
        structures,
        soap,
        n_jobs=n_jobs,
        batch_size=batch_size,
        out=mmap_path,
        dtype=np.dtype(dtype),
        prefer="threads",
    )
    est_bytes = int(X.shape[0] * X.shape[1] * np.dtype(dtype).itemsize)
    avail_bytes = _meminfo_available_bytes()
    if avail_bytes is not None and est_bytes > int(avail_bytes * 0.8):
        raise MemoryError(
            "Feature matrix is too large for available memory. "
            "Try lowering --batch-size or using --feature-dtype float32."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {"X_path": str(mmap_path), "shape": X.shape, "dtype": str(X.dtype)}, f
        )
    return X


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "spearman": float(rho)}


def _make_xgb_params(device: str, tree_method: str | None) -> dict:
    if tree_method is None:
        tree_method = "gpu_hist" if device == "cuda" else "hist"
    return {"tree_method": tree_method, "device": device}


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    sample_weight,
    device: str,
    tree_method: str | None,
):
    xgb_params = _make_xgb_params(device, tree_method)
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method=xgb_params["tree_method"],
        device=xgb_params["device"],
        early_stopping_rounds=50,
        eval_metric=["rmse", "mae"],
    )

    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )
    except xgb.core.XGBoostError as exc:
        msg = str(exc)
        if "gpu_hist" in msg or "No visible GPU" in msg or "Device is changed" in msg:
            print("GPU unavailable for XGBoost; falling back to CPU hist.")
            model = xgb.XGBRegressor(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.5,
                tree_method="hist",
                device="cpu",
                early_stopping_rounds=50,
                eval_metric=["rmse", "mae"],
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=50,
            )
        else:
            raise
    pred = model.predict(X_test)
    metrics = compute_metrics(y_test, pred)
    return model, metrics, pred


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    cache_path = Path(args.cache)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to outdir {outdir}. Use --outdir to set a writable path."
        ) from exc

    print("Loading structures...")
    structures = load_structures_from_extxyz(args.extxyz)
    print(f"Loaded {len(structures)} structures")

    valid_structures = []
    y_values = []
    for atoms in structures:
        if "dft_eps_0" in atoms.info:
            valid_structures.append(atoms)
            y_values.append(atoms.info["dft_eps_0"])

    structures = valid_structures
    y = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(y)
    if not finite_mask.all():
        removed = int((~finite_mask).sum())
        print(f"Removing {removed} structures with non-finite dft_eps_0")
        y = y[finite_mask]
        structures = [s for s, keep in zip(structures, finite_mask) if keep]
    if args.log_target:
        y = np.log1p(y)

    print(f"Using {len(structures)} structures with dft_eps_0")

    # Compute features (uses on-disk memmap cache to limit RAM)
    print("Computing features (SOAP + composition)...")
    X = build_features(
        structures,
        cache_path,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        dtype=args.feature_dtype,
        soap_r_cut=args.soap_r_cut,
        soap_n_max=args.soap_n_max,
        soap_l_max=args.soap_l_max,
        soap_sparse=args.soap_sparse,
    )
    print("Feature matrix shape:", X.shape)

    groups = [composition_string(a) for a in structures]
    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Baseline
    print("\nTraining baseline model...")
    baseline_model, baseline_metrics, _ = train_model(
        X_train, y_train, X_test, y_test, None, args.device, args.tree_method
    )

    # Outlier-weighted
    q = args.outlier_quantile
    low, high = np.quantile(y_train, [q, 1.0 - q])
    weights = np.ones_like(y_train, dtype=float)
    weights[(y_train <= low) | (y_train >= high)] = args.outlier_weight

    print("\nTraining outlier-weighted model...")
    weighted_model, weighted_metrics, _ = train_model(
        X_train, y_train, X_test, y_test, weights, args.device, args.tree_method
    )

    # Save models
    baseline_path = outdir / "xgb_surrogate_dft_eps_0_baseline.json"
    weighted_path = outdir / "xgb_surrogate_dft_eps_0_outlier_weighted.json"
    baseline_model.save_model(baseline_path)
    weighted_model.save_model(weighted_path)

    summary = {
        "extxyz": args.extxyz,
        "n_structures": len(structures),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "outlier_quantile": q,
        "outlier_weight": args.outlier_weight,
        "log_target": bool(args.log_target),
        "outlier_low": float(low),
        "outlier_high": float(high),
        "baseline_metrics": baseline_metrics,
        "weighted_metrics": weighted_metrics,
        "models": {
            "baseline": str(baseline_path),
            "weighted": str(weighted_path),
        },
    }

    summary_path = outdir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train SOAP-based XGBoost surrogate models for materials properties.

Extracted from chem.surrogates.soap — this is the training pipeline only.
For inference, use chem.surrogates.soap.predict_properties().
"""

from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os
import pickle
import json

from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chem.surrogates.soap import (
    EXTXYZ_PATH,
    MODELS,
    MODEL_TO_PROPERTY,
    N_JOBS,
    TEST_SIZE,
    RANDOM_STATE,
    build_soap_descriptor,
    composition_features,
    composition_string,
    featurize_structures,
    load_structures_from_extxyz,
    test_no_info_leakage,
)


def train_models(extxyz_path=EXTXYZ_PATH, n_jobs=N_JOBS):
    print(f"Using {n_jobs} CPU cores for parallel processing")

    print("Loading structures...")
    structures = load_structures_from_extxyz(extxyz_path)
    print(f"Loaded {len(structures)} structures")

    all_structures = structures

    # =========================================================
    # DATA LEAKAGE CHECK (run once for all properties)
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECKING FOR DATA LEAKAGE")
    print("=" * 60)

    # Inspect what's in atoms.info for first few structures
    print("\nInspecting atoms.info contents from first 3 structures:")
    for i, atoms in enumerate(all_structures[:3]):
        print(f"\nStructure {i}:")
        print(f"  Formula: {atoms.get_chemical_formula()}")
        print(f"  info keys: {list(atoms.info.keys())}")
        print(f"  info values: {atoms.info}")

    # Check if atoms.arrays contains any suspicious data
    print("\nInspecting atoms.arrays from first structure:")
    print(f"  arrays keys: {list(all_structures[0].arrays.keys())}")

    # Check for other properties that might correlate with target properties
    print("\nChecking for potentially correlated properties in dataset:")
    common_correlated_props = [
        "eps_inf",
        "eps_0",
        "e_total",
        "bandgap",
        "energy",
        "energy_per_atom",
        "formation_energy",
        "gap",
        "band_gap",
        "homo",
        "lumo",
    ]

    found_suspicious = []
    for atoms in all_structures[:5]:
        for key in atoms.info.keys():
            if key.lower() in [p.lower() for p in common_correlated_props]:
                found_suspicious.append(key)

    print(f"Found properties in dataset: {set(found_suspicious)}")
    print("⚠ These properties should NOT be used in features (only as targets)!")

    print("=" * 60 + "\n")

    # Create cache directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    leakage_test_passed = test_no_info_leakage(all_structures)
    if not leakage_test_passed:
        print("⚠⚠⚠ CRITICAL: DATA LEAKAGE DETECTED! ⚠⚠⚠")
        print("Feature computation is accessing atoms.info!")
        raise RuntimeError("Data leakage detected in feature computation")

    # =========================================================
    # TRAIN MODELS FOR EACH PROPERTY
    # =========================================================

    results_summary = {}

    for MODEL_NAME in MODELS:
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL FOR: {MODEL_NAME}")
        print("=" * 80 + "\n")

        PROPERTY_KEY = MODEL_TO_PROPERTY[MODEL_NAME]

        valid_structures = []
        y_values = []
        for atoms in all_structures:
            if PROPERTY_KEY in atoms.info:
                valid_structures.append(atoms)
                y_values.append(atoms.info[PROPERTY_KEY])

        structures = valid_structures
        y = np.array(y_values)

        print(f"Using {len(structures)} structures with property '{PROPERTY_KEY}'")

        cache_file = f"data/soap_features_{MODEL_NAME}.pkl"

        if os.path.exists(cache_file):
            print(f"\nLoading cached features from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                X = cache_data["X"]
                y_cached = cache_data["y"]

                if len(y_cached) == len(y) and np.allclose(y_cached, y):
                    print(f"✓ Loaded cached features: {X.shape}")
                    skip_feature_computation = True
                else:
                    print("⚠ Cache mismatch detected, recomputing features...")
                    skip_feature_computation = False
        else:
            print(f"\nNo cache found, will compute features and save to {cache_file}")
            skip_feature_computation = False

        if not skip_feature_computation:
            print("Building composition features...")
            print("Initializing SOAP descriptor...")

            species = sorted(
                list({Z for atoms in structures for Z in atoms.get_atomic_numbers()})
            )

            soap = build_soap_descriptor(species)

            print("Computing features (parallel)...")
            X = featurize_structures(structures, soap, n_jobs=n_jobs)

            print("Feature matrix shape:", X.shape)

            print(f"Saving features to {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump({"X": X, "y": y}, f)
            print("✓ Features cached successfully")
        else:
            print("Using cached features")

        groups = [composition_string(a) for a in structures]
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("\n" + "=" * 60)
        print(f"TRAINING XGBOOST FOR {MODEL_NAME}")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print("Max iterations: 800")
        print("=" * 60 + "\n")

        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=50,
            eval_metric=["rmse", "mae"],
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )

        print(f"\n✓ Training completed after {model.best_iteration} iterations")
        print(f"✓ Best validation RMSE: {model.best_score:.4f}")

        print("\nEvaluating model...")

        pred = model.predict(X_test)

        rho, _ = spearmanr(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        print(f"\nSpearman correlation: {rho:.3f}")
        print(f"MAE: {mae:.3f}")

        k = max(1, int(0.1 * len(y_test)))

        top_true_idx = np.argsort(y_test)[-k:]
        top_pred_idx = np.argsort(pred)[-k:]

        precision = len(set(top_true_idx) & set(top_pred_idx)) / k

        print(f"Top-{k}% precision: {precision:.3f} (k={k})")

        print("\nPlotting feature importance...")

        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=20)
        plt.title(f"Feature Importance - {MODEL_NAME}")
        plt.tight_layout()
        importance_file = f"feature_importance_{MODEL_NAME}.png"
        plt.savefig(importance_file)
        print(f"Feature importance plot saved to {importance_file}")
        plt.close()

        print("\n" + "=" * 60)
        print(f"SAVING MODEL FOR {MODEL_NAME}")
        print("=" * 60)

        model_file = f"data/xgb_surrogate_{MODEL_NAME}.json"
        model.save_model(model_file)
        print(f"✓ Model saved to {model_file}")

        metadata = {
            "model_name": MODEL_NAME,
            "property": PROPERTY_KEY,
            "n_features": X.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "spearman": float(rho),
            "mae": float(mae),
            "top_k_precision": float(precision),
        }

        metadata_file = f"data/xgb_metadata_{MODEL_NAME}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file}")

        results_summary[MODEL_NAME] = {
            "property": PROPERTY_KEY,
            "spearman": rho,
            "mae": mae,
            "top_k_precision": precision,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        print("=" * 60)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80 + "\n")

    for model_name, results in results_summary.items():
        print(f"{model_name}:")
        print(f"  Source property: {results['property']}")
        print(f"  Spearman: {results['spearman']:.3f}")
        print(f"  MAE: {results['mae']:.3f}")
        print(f"  Top-10% precision: {results['top_k_precision']:.3f}")
        print(f"  Train/Test: {results['n_train']}/{results['n_test']}")
        print()

    print("Models saved:")
    for model_name in MODELS:
        print(f"  - data/xgb_surrogate_{model_name}.json")

    print("\nDONE.")


if __name__ == "__main__":
    train_models()

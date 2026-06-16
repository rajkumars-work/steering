"""
Pipeline evaluation utilities.

Runs Stage-6 (MLIP relaxation + property evaluation) on a Stage-5 CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ase.io import read as ase_read

from chem.props.dielectric import stats as default_training_stats
from chem.material_classifier import classify_material
from chem.pareto_front import ParetoFront
from chem.props.stage6 import stage6_fieldnames, stage6_relax_and_props
from chem.surrogates.soap import build_soap_descriptor, featurize_structures, load_structures_from_extxyz
import xgboost as xgb


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage-5 CSV not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _write_csv(path: str, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_stage6_from_csv(
    stage5_csv: str,
    stage6_out: str,
    stage6_dir: str,
    batch_size: int,
    checkpoints_dir: str | None,
    parallel_batches: int,
    num_gpus: int,
    keep_failed: bool,
    timeout_s: float,
    wait_interval_s: float,
    eps0_surrogate_model: str | None,
    eps0_surrogate_log_target: bool | None,
    eps0_surrogate_extxyz: str | None,
    eps0_soap_r_cut: float,
    eps0_soap_n_max: int,
    eps0_soap_l_max: int,
    eps0_soap_sparse: bool,
    eps0_n_jobs: int,
    eps0_batch_size: int,
) -> int:
    stage5_rows = _read_csv_rows(stage5_csv)
    if not stage5_rows:
        print(f"No rows found in {stage5_csv}")
        return 0

    stage6_rows = stage6_relax_and_props(
        stage5_rows,
        batch_size=batch_size,
        checkpoints_dir=checkpoints_dir,
        output_dir=stage6_dir,
        parallel_batches=parallel_batches,
        num_gpus=num_gpus,
        keep_failed=keep_failed,
        timeout_s=timeout_s,
        wait_interval_s=wait_interval_s,
    )

    if eps0_surrogate_model:
        if eps0_surrogate_log_target is None:
            eps0_surrogate_log_target = _load_surrogate_log_target(eps0_surrogate_model)
        stage6_rows = _override_eps0_with_surrogate(
            stage6_rows,
            model_path=eps0_surrogate_model,
            log_target=bool(eps0_surrogate_log_target),
            extxyz_path=eps0_surrogate_extxyz,
            soap_r_cut=eps0_soap_r_cut,
            soap_n_max=eps0_soap_n_max,
            soap_l_max=eps0_soap_l_max,
            soap_sparse=eps0_soap_sparse,
            n_jobs=eps0_n_jobs,
            batch_size=eps0_batch_size,
        )

    relaxed_total = len(stage6_rows)
    relaxed_with_path = sum(
        1 for row in stage6_rows if str(row.get("relaxed_structure_path", "")).strip()
    )
    eligible_total = _count_pareto_eligible(stage6_rows)
    front_rows = _pareto_front_rows(stage6_rows)

    fieldnames = stage6_fieldnames(list(stage5_rows[0].keys()))
    _write_csv(stage6_out, front_rows, fieldnames)
    print(f"Stage 6 properties: {len(front_rows)} written to {stage6_out}")
    print("\nStage-6 Summary")
    print(f"  input rows (pre-relaxation): {len(stage5_rows)}")
    print(f"  relaxed outputs: {relaxed_total}")
    print(f"  relaxed with paths: {relaxed_with_path}")
    print(f"  pareto-eligible rows: {eligible_total}")
    print(f"  pareto-front rows: {len(front_rows)}")
    _print_stats(front_rows)
    _print_pareto_front(front_rows)
    return len(front_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage-6 (relaxation + property evaluation) from a Stage-5 CSV"
    )
    parser.add_argument(
        "--stage5-in",
        default="pipeline_stage5_filtered.csv",
        help="Input Stage-5 CSV (must include structure_path)",
    )
    parser.add_argument(
        "--stage6-out",
        default="pipeline_stage6_properties.csv",
        help="Output CSV for Stage-6 MLIP relaxation + properties",
    )
    parser.add_argument(
        "--stage6-dir",
        default="/data/assets/runs/di",
        help="Output directory for Stage-6 relaxed extxyz files",
    )
    parser.add_argument(
        "--stage6-batch-size",
        type=int,
        default=4,
        help="Structures per MLIP batch for Stage-6",
    )
    parser.add_argument(
        "--stage6-parallel-batches",
        type=int,
        default=1,
        help="Parallel MLIP batches for Stage-6 (0 = auto)",
    )
    parser.add_argument(
        "--stage6-num-gpus",
        type=int,
        default=0,
        help="Number of GPUs for Stage-6 (0 = auto)",
    )
    parser.add_argument(
        "--stage6-checkpoints-dir",
        default="/data/assets/checkpoints",
        help="Path to MLIP checkpoints (ATLAS_CHECKPOINTS_DIR)",
    )
    parser.add_argument(
        "--stage6-keep-failed",
        action="store_true",
        help="Keep Stage-6 rows that failed MLIP compute with stage6_error",
    )
    parser.add_argument(
        "--stage6-timeout-s",
        type=float,
        default=0.0,
        help="Per-batch timeout in seconds for Stage-6 (0 disables timeout)",
    )
    parser.add_argument(
        "--stage6-wait-interval-s",
        type=float,
        default=30.0,
        help="Polling interval in seconds while waiting on Stage-6 Ray tasks",
    )
    parser.add_argument(
        "--eps0-surrogate-model",
        default="data/checkpoints/eps0_outlier_weighted/xgb_surrogate_dft_eps_0_outlier_weighted.json",
        help=(
            "XGBoost surrogate model for eps_0 (path). "
            "Defaults to data/checkpoints/eps0_outlier_weighted/xgb_surrogate_dft_eps_0_outlier_weighted.json."
        ),
    )
    parser.add_argument(
        "--eps0-surrogate-log-target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Surrogate model predicts log1p(eps_0); apply expm1 to outputs.",
    )
    parser.add_argument(
        "--eps0-surrogate-extxyz",
        default=None,
        help="Extxyz path to derive species list for SOAP (defaults to training_summary.json extxyz if available).",
    )
    parser.add_argument("--eps0-soap-r-cut", type=float, default=4.0)
    parser.add_argument("--eps0-soap-n-max", type=int, default=4)
    parser.add_argument("--eps0-soap-l-max", type=int, default=3)
    parser.add_argument("--eps0-soap-sparse", action="store_true")
    parser.add_argument("--eps0-n-jobs", type=int, default=4)
    parser.add_argument("--eps0-batch-size", type=int, default=32)

    args = parser.parse_args()

    run_stage6_from_csv(
        stage5_csv=args.stage5_in,
        stage6_out=args.stage6_out,
        stage6_dir=args.stage6_dir,
        batch_size=args.stage6_batch_size,
        checkpoints_dir=args.stage6_checkpoints_dir,
        parallel_batches=args.stage6_parallel_batches,
        num_gpus=args.stage6_num_gpus,
        keep_failed=args.stage6_keep_failed,
        timeout_s=args.stage6_timeout_s,
        wait_interval_s=args.stage6_wait_interval_s,
        eps0_surrogate_model=args.eps0_surrogate_model,
        eps0_surrogate_log_target=args.eps0_surrogate_log_target,
        eps0_surrogate_extxyz=args.eps0_surrogate_extxyz,
        eps0_soap_r_cut=args.eps0_soap_r_cut,
        eps0_soap_n_max=args.eps0_soap_n_max,
        eps0_soap_l_max=args.eps0_soap_l_max,
        eps0_soap_sparse=args.eps0_soap_sparse,
        eps0_n_jobs=args.eps0_n_jobs,
        eps0_batch_size=args.eps0_batch_size,
    )


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _get_value(row: Dict[str, object], keys: List[str]) -> Optional[float]:
    for key in keys:
        v = _to_float(row.get(key))
        if v is not None:
            return v
    return None


_BANDGAP_KEYS = ["mlip_bandgap", "bandgap", "pred_bandgap"]
_EPS0_KEYS = ["surrogate_eps_0", "eps_0", "pred_eps_0"]


def _count_pareto_eligible(rows: List[Dict[str, object]]) -> int:
    eligible = 0
    for row in rows:
        bandgap = _get_value(row, _BANDGAP_KEYS)
        eps_0 = _get_value(row, _EPS0_KEYS)
        if bandgap is None or eps_0 is None:
            continue
        eligible += 1
    return eligible


def _pareto_front_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []

    pf = ParetoFront(maximize_bandgap=True, maximize_eps_0=True)
    id_to_row: Dict[str, Dict[str, object]] = {}
    for idx, row in enumerate(rows):
        bandgap = _get_value(row, _BANDGAP_KEYS)
        eps_0 = _get_value(row, _EPS0_KEYS)
        if bandgap is None or eps_0 is None:
            continue
        item_id = f"row_{idx}"
        id_to_row[item_id] = row
        pf.add(item_id, bandgap, eps_0)

    front = pf.pareto()
    if not front:
        return []

    return [id_to_row[item_id] for item_id in front.keys()]


def _print_pareto_front(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print(json.dumps([], indent=2))
        return

    items: List[Tuple[str, float, float]] = []
    for row in rows:
        bandgap = _get_value(row, _BANDGAP_KEYS)
        eps_0 = _get_value(row, _EPS0_KEYS)
        if bandgap is None or eps_0 is None:
            continue
        composition = str(row.get("composition_formula", ""))
        items.append((composition, bandgap, eps_0))

    items.sort(key=lambda x: (-x[1], -x[2], x[0]))

    payload: List[Dict[str, object]] = []
    for composition, bandgap, eps_0 in items:
        classification = classify_material(composition) if composition else {}
        payload.append(
            {
                "composition": composition,
                "bandgap": round(float(bandgap), 2),
                "eps_0": round(float(eps_0), 2),
                "family": classification.get("family", ""),
                "structure": classification.get("structure_guess", ""),
            }
        )

    print(json.dumps(payload, indent=2))


def _collect_values(rows: List[Dict[str, object]], keys: List[str]) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        v = None
        for key in keys:
            v = _to_float(row.get(key))
            if v is not None:
                break
        if v is not None:
            values.append(v)
    return np.array(values, dtype=float)


def _load_surrogate_species(extxyz_path: str | None) -> List[int]:
    if extxyz_path:
        structures = load_structures_from_extxyz(extxyz_path)
        if not structures:
            raise ValueError(f"No structures found for extxyz: {extxyz_path}")
        return sorted(list({Z for atoms in structures for Z in atoms.get_atomic_numbers()}))

    summary_path = os.path.join("data", "checkpoints", "eps0_outlier_weighted", "training_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            extxyz = summary.get("extxyz")
            if extxyz:
                structures = load_structures_from_extxyz(extxyz)
                if structures:
                    return sorted(list({Z for atoms in structures for Z in atoms.get_atomic_numbers()}))
        except Exception as exc:
            print(f"Warning: failed to load surrogate species from summary: {exc}")

    raise ValueError(
        "Unable to derive SOAP species list. Provide --eps0-surrogate-extxyz."
    )


def _load_surrogate_log_target(model_path: str) -> bool:
    summary_path = os.path.join("data", "checkpoints", "eps0_outlier_weighted", "training_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            models = summary.get("models", {})
            if model_path in models.values():
                return bool(summary.get("log_target", False))
        except Exception as exc:
            print(f"Warning: failed to load surrogate log_target from summary: {exc}")
    return False


def _override_eps0_with_surrogate(
    rows: List[Dict[str, object]],
    model_path: str,
    log_target: bool,
    extxyz_path: str | None,
    soap_r_cut: float,
    soap_n_max: int,
    soap_l_max: int,
    soap_sparse: bool,
    n_jobs: int,
    batch_size: int,
) -> List[Dict[str, object]]:
    if not rows:
        return rows

    paths = []
    idxs = []
    for i, row in enumerate(rows):
        rel_path = str(row.get("relaxed_structure_path", "")).strip()
        if not rel_path or not os.path.exists(rel_path):
            continue
        paths.append(rel_path)
        idxs.append(i)

    if not paths:
        return rows

    species = _load_surrogate_species(extxyz_path)
    soap = build_soap_descriptor(
        species,
        r_cut=soap_r_cut,
        n_max=soap_n_max,
        l_max=soap_l_max,
        sparse=soap_sparse,
    )

    atoms_list = []
    for path in paths:
        atoms_list.append(ase_read(path))

    X = featurize_structures(
        atoms_list,
        soap,
        n_jobs=n_jobs,
        batch_size=batch_size,
        dtype=np.float32,
        prefer="threads",
    )

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    expected = int(model.get_booster().num_features())
    got = int(X.shape[1]) if X.ndim == 2 else int(X.shape[0])
    if expected != got:
        raise ValueError(
            "Feature shape mismatch for eps_0 surrogate: "
            f"model expects {expected} features, got {got}. "
            "Use a model trained with the same SOAP settings/species, "
            "or adjust --eps0-soap-* / --eps0-surrogate-extxyz to match."
        )
    pred = model.predict(X)
    if log_target:
        pred = np.expm1(pred)
    pred = np.maximum(pred, 1.0)

    for idx, value in zip(idxs, pred, strict=False):
        rows[idx]["surrogate_eps_0"] = float(value)
        rows[idx]["eps_0"] = float(value)
        rows[idx].pop("mlip_eps_0", None)

    return rows


def _summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _format_float(value: float, width: int = 8) -> str:
    if value != value:  # NaN
        return " " * (width - 3) + "n/a"
    return f"{value:>{width}.3f}"


def _training_stats(prop_key: str) -> Optional[Dict[str, float]]:
    stats = _load_training_stats().get(prop_key)
    if not stats or not isinstance(stats, dict):
        return None
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return None
    return {k: float(v) for k, v in stats.items() if v is not None}


_TRAINING_STATS_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def _load_training_stats() -> Dict[str, Dict[str, float]]:
    global _TRAINING_STATS_CACHE
    if _TRAINING_STATS_CACHE is not None:
        return _TRAINING_STATS_CACHE

    stats = dict(default_training_stats)
    stats_path = os.path.join("data", "training_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                # Prefer file stats when present
                stats.update(loaded)
        except Exception as exc:
            print(f"Warning: failed to load training stats from {stats_path}: {exc}")

    _TRAINING_STATS_CACHE = stats
    return stats


def _print_stats(stage6_rows: List[Dict[str, object]]) -> None:
    print("\nStats Summary (Stage-6 outputs vs training set)")

    targets = [
        ("bandgap (eV)", ["mlip_bandgap", "bandgap"], "mlip_bandgap", "dft_band_gap"),
        ("eps_0 (static)", ["surrogate_eps_0", "eps_0"], "surrogate_eps_0", "dft_eps_0"),
    ]

    for label, keys, training_key, dft_key in targets:
        values = _collect_values(stage6_rows, keys)
        summary = _summarize(values)
        train = _training_stats(training_key)
        dft_train = _training_stats(dft_key)

        print(f"\n{label}")
        print(
            "  n={n}  min={min}  p25={p25}  median={median}  mean={mean}  p75={p75}  max={max}  std={std}".format(
                n=summary["n"],
                min=_format_float(summary["min"]),
                p25=_format_float(summary["p25"]),
                median=_format_float(summary["median"]),
                mean=_format_float(summary["mean"]),
                p75=_format_float(summary["p75"]),
                max=_format_float(summary["max"]),
                std=_format_float(summary["std"]),
            )
        )

        if train is None and dft_train is None:
            print("  training: n/a (no training stats available)")
            continue

        train_mean = train.get("mean") if train else None
        train_std = train.get("std") if train else None
        if (
            train_mean is not None
            and train_std is not None
            and train_std > 0
            and summary["n"] > 0
            and summary["mean"] == summary["mean"]
        ):
            z_mean = (summary["mean"] - train_mean) / train_std
        else:
            z_mean = float("nan")

        if train_mean is not None and train_std is not None and summary["n"] > 0:
            within_1s = float(np.mean(np.abs(values - train_mean) <= train_std)) * 100.0
            within_2s = float(np.mean(np.abs(values - train_mean) <= 2 * train_std)) * 100.0
        else:
            within_1s = float("nan")
            within_2s = float("nan")

        if train:
            print(
                "  training (model): mean={mean}  std={std}  Δmean={delta}  z_mean={z}".format(
                    mean=_format_float(train_mean),
                    std=_format_float(train_std),
                    delta=_format_float(summary["mean"] - train_mean),
                    z=_format_float(z_mean),
                )
            )
            if all(k in train for k in ("median", "p25", "p75")):
                print(
                    "    median={median}  p25={p25}  p75={p75}".format(
                        median=_format_float(train["median"]),
                        p25=_format_float(train["p25"]),
                        p75=_format_float(train["p75"]),
                    )
                )
            print(
                "  coverage (model): within 1σ={w1}%  within 2σ={w2}%".format(
                    w1=_format_float(within_1s, width=6).strip(),
                    w2=_format_float(within_2s, width=6).strip(),
                )
            )

        if dft_train:
            print(
                "  training (dft):  mean={mean}  std={std}".format(
                    mean=_format_float(dft_train["mean"]),
                    std=_format_float(dft_train["std"]),
                )
            )
            if all(k in dft_train for k in ("median", "p25", "p75")):
                print(
                    "    median={median}  p25={p25}  p75={p75}".format(
                        median=_format_float(dft_train["median"]),
                        p25=_format_float(dft_train["p25"]),
                        p75=_format_float(dft_train["p75"]),
                    )
                )


if __name__ == "__main__":
    main()

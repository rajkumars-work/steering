"""Helpers for querying the RECAST synthetic-materials corpus.

Loads `eval/recast_corpus/index.parquet` and provides a few convenience
functions for filtering by model, dataset, or screen run, and for joining
the per-row data with the checkpoint registry (hyperparams, architecture,
notes).

Typical use:

    from chem.corpus_query import load_corpus, with_registry, by_model, by_dataset

    df = load_corpus()
    df_k7 = by_model(df, "ckpt_autolabel_v1_binrho_k7")
    df_full = with_registry(df)
    print(df_full.groupby("dataset_version")["overall_pass"].mean())
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PARQUET = ROOT / "eval/recast_corpus/index.parquet"
REGISTRY_JSON = ROOT / "data/checkpoint_registry.json"


def load_corpus(path: str | Path = CORPUS_PARQUET) -> pd.DataFrame:
    """Load the parquet index. Caller can filter as needed."""
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    if not REGISTRY_JSON.exists():
        return pd.DataFrame()
    rows = json.loads(REGISTRY_JSON.read_text())
    return pd.DataFrame(rows)


def with_registry(df: pd.DataFrame, *, suffix: str = "_reg") -> pd.DataFrame:
    """Left-join corpus rows with checkpoint-registry metadata.

    Adds columns: training_dataset_csv, hyperparams, architecture, source,
    inferred, notes — suffixed `<col>_reg` to avoid collisions.
    """
    reg = load_registry()
    if reg.empty:
        return df
    keep = ["model_version", "training_dataset_version_id",
            "training_dataset_csv", "hyperparams", "architecture",
            "source", "inferred", "notes", "kind"]
    keep = [c for c in keep if c in reg.columns]
    reg_sub = reg[keep].rename(columns={c: f"{c}{suffix}" if c != "model_version" else c
                                         for c in keep})
    return df.merge(reg_sub, on="model_version", how="left")


def by_model(df: pd.DataFrame, model_version: str | Iterable[str]) -> pd.DataFrame:
    if isinstance(model_version, str):
        return df[df.model_version == model_version]
    return df[df.model_version.isin(list(model_version))]


def by_dataset(df: pd.DataFrame, dataset_version: str | Iterable[str]) -> pd.DataFrame:
    if isinstance(dataset_version, str):
        return df[df.dataset_version == dataset_version]
    return df[df.dataset_version.isin(list(dataset_version))]


def by_screen(df: pd.DataFrame, screen_id: str | Iterable[str]) -> pd.DataFrame:
    if isinstance(screen_id, str):
        return df[df.screen_id == screen_id]
    return df[df.screen_id.isin(list(screen_id))]


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-model summary: count, pass rates, dataset version."""
    cols = {
        "n_structures": ("model_version", "size"),
        "dataset_version": ("dataset_version", "first"),
    }
    if "tier0_overall_pass" in df.columns:
        cols["frac_tier0_pass"] = ("tier0_overall_pass", "mean")
    if "overall_pass" in df.columns:
        cols["frac_overall_pass"] = ("overall_pass", "mean")
    if "is_sun" in df.columns:
        cols["frac_is_sun"] = ("is_sun", "mean")
    return (df.groupby("model_version", dropna=False)
              .agg(**cols)
              .sort_values("n_structures", ascending=False))


def compare_datasets(df: pd.DataFrame, datasets: Iterable[str], *,
                     metric: str = "tier0_overall_pass") -> pd.DataFrame:
    """Compare a metric across listed dataset_version cohorts.

    Default metric is tier0_overall_pass. Returns a small DataFrame with
    n_structures, mean(metric), and median atoms.
    """
    sub = by_dataset(df, datasets)
    out = (sub.groupby("dataset_version", dropna=False)
              .agg(n_structures=("model_version", "size"),
                   metric_mean=(metric, "mean"),
                   median_natoms=("n_atoms_actual", "median")))
    out = out.rename(columns={"metric_mean": f"mean_{metric}"})
    return out


__all__ = [
    "load_corpus", "load_registry", "with_registry",
    "by_model", "by_dataset", "by_screen",
    "summary_by_model", "compare_datasets",
    "CORPUS_PARQUET", "REGISTRY_JSON",
]

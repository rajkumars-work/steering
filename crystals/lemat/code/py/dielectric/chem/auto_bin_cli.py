#!/usr/bin/env python3
"""Dry-run driver for `chem.auto_bin`.

Loads float values for each requested property from available caches and
nequip-labeled JSONLs, runs `auto_bin`, and writes a JSON report listing
per-property edges, labels, counts, and effective support. Nothing is
mutated; the intent is to preview what the new binning would look like
before we touch the training CSV.

Sources per property (first match wins):
    band_gap              — nequip_labels/*.jsonl (nequip_band_gap)
                            ∪ data/cache/*.jsonl (band_gap where present)
    nequip_eps_0          — nequip_labels/*.jsonl (nequip_eps_0)
    stability             — data/cache/*.jsonl (stability == e_above_hull)
    density               — data/cache/*.jsonl (density)
    debye_temperature     — data/cache/*.jsonl (debye_temperature where present)

Usage:
    PYTHONPATH=/data/rkumar/code/py/dielectric \\
    python chem/auto_bin_cli.py \\
        --properties band_gap,nequip_eps_0,stability \\
        --report auto_bin_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from chem.auto_bin import BinnerConfig, auto_bin, resolve_support

# Where the caches live on disk.
LABEL_DIR = PROJECT / "eval/fair_comparison/nequip_labels"
CACHE_DIR = PROJECT / "data/cache"

# Property → how to pull values from a JSONL line
# Some properties come from the nequip labels (MLIP-predicted),
# others from the upstream caches (DFT-computed).
PROPERTY_SOURCES = {
    "band_gap": [
        (LABEL_DIR, "*.jsonl", "nequip_band_gap"),
    ],
    "nequip_eps_0": [
        (LABEL_DIR, "*.jsonl", "nequip_eps_0"),
    ],
    "stability": [
        (CACHE_DIR, "*.jsonl", "stability"),
    ],
    "density": [
        (CACHE_DIR, "*.jsonl", "density"),
    ],
    "debye_temperature": [
        (CACHE_DIR, "*.jsonl", "debye_temperature"),
    ],
}

# Short property tags for the emitted labels
PROP_TAGS = {
    "band_gap": "bg",
    "nequip_eps_0": "k",
    "stability": "hull",
    "density": "rho",
    "debye_temperature": "debye",
}


def iter_jsonl_id_value(
    dir_path: Path, glob: str, key: str,
) -> Iterable[tuple[str, float]]:
    paths = sorted(dir_path.glob(glob))
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                v = d.get(key)
                if not isinstance(v, (int, float)):
                    continue
                # IDs come in a couple of formats: "mp:mp-1234", "wbm-1-005",
                # bare material_id. Normalize to strip any "<origin>:" prefix.
                rid = d.get("id") or d.get("material_id") or ""
                if not rid:
                    continue
                if ":" in rid:
                    rid = rid.split(":", 1)[-1]
                yield rid, float(v)


def load_training_ids(csv_path: Path) -> set[str]:
    """Return the set of (prefix-stripped) IDs present in the training CSV."""
    import csv
    ids: set[str] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            if ":" in rid:
                rid = rid.split(":", 1)[-1]
            if rid:
                ids.add(rid)
    return ids


def collect_values(
    prop: str, training_ids: set[str] | None = None, limit: int | None = None,
) -> list[float]:
    sources = PROPERTY_SOURCES.get(prop)
    if sources is None:
        raise KeyError(f"Unknown property {prop!r}. "
                       f"Known: {sorted(PROPERTY_SOURCES)}")
    vals: list[float] = []
    for dir_path, glob, key in sources:
        if not dir_path.exists():
            print(f"  [{prop}] skipping {dir_path} (does not exist)")
            continue
        for rid, v in iter_jsonl_id_value(dir_path, glob, key):
            if training_ids is not None and rid not in training_ids:
                continue
            vals.append(v)
            if limit is not None and len(vals) >= limit:
                return vals
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--properties", default="band_gap,nequip_eps_0,stability",
                    help="Comma-separated property names "
                         f"(known: {', '.join(sorted(PROPERTY_SOURCES))})")
    ap.add_argument("--k-max", type=int, default=5,
                    help="Maximum bins per property (default: 5)")
    ap.add_argument("--min-support", type=int, default=None,
                    help="Override min_support. Default: derive from dropout.")
    ap.add_argument("--p-segment", type=float, default=0.2)
    ap.add_argument("--p-label", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--target-updates", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap values per property (for quick iteration)")
    ap.add_argument("--training-csv", default=None,
                    help="If set, restrict values to IDs present in this "
                         "training CSV. Recommended — bins should reflect "
                         "the data the model will actually train on.")
    ap.add_argument("--report", default="auto_bin_report.json",
                    help="Path to write the JSON report")
    args = ap.parse_args()

    training_ids: set[str] | None = None
    if args.training_csv:
        t0 = Path(args.training_csv)
        t0 = t0 if t0.is_absolute() else PROJECT / t0
        print(f"Loading training IDs from {t0}...")
        training_ids = load_training_ids(t0)
        print(f"  {len(training_ids):,} training IDs")

    report: dict = {
        "config": {
            "k_max": args.k_max,
            "min_support": args.min_support,
            "p_segment": args.p_segment,
            "p_label": args.p_label,
            "epochs": args.epochs,
            "target_updates": args.target_updates,
        },
        "properties": {},
    }

    for prop in args.properties.split(","):
        prop = prop.strip()
        if not prop:
            continue
        print(f"\n=== {prop} ===")
        cfg = BinnerConfig(
            prop_tag=PROP_TAGS.get(prop, prop),
            k_max=args.k_max,
            min_support=args.min_support,
            p_segment=args.p_segment,
            p_label=args.p_label,
            epochs=args.epochs,
            target_updates=args.target_updates,
        )
        M = resolve_support(cfg)
        print(f"  min_support M = {M:,}")

        vals = collect_values(prop, training_ids=training_ids, limit=args.limit)
        print(f"  loaded {len(vals):,} values")
        if not vals:
            print("  SKIP (no values)")
            continue

        b = auto_bin(vals, cfg)
        print(f"  K = {len(b.labels)}  tiers = {b.tiers_used}")
        print(f"  edges = {b.edges}")
        print(f"  labels/counts:")
        for lbl, cnt in zip(b.labels, b.counts):
            bar = "█" * min(60, int(60 * cnt / max(b.counts)))
            pct = 100 * cnt / sum(b.counts)
            print(f"    {lbl:<12s}  {cnt:>8,d}  {pct:5.1f}%  {bar}")
        print(f"  effective_support = {b.effective_support:,} "
              f"(threshold {b.min_support:,}) — "
              f"{'OK' if b.effective_support >= b.min_support else 'UNDER'}")

        report["properties"][prop] = asdict(b)

    out_path = PROJECT / args.report if not Path(args.report).is_absolute() else Path(args.report)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

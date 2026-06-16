#!/usr/bin/env python3
"""Cross-variant analysis of 1000-structure unconditional generation results.

Reads results from eval/uncond_1000/{v1,v2,v3,v4,v5}/ and produces:
  1. Overall comparison table
  2. Pass rate by chemical system complexity (n_elements)
  3. Pass rate by crystal system
  4. Pass rate by training-set representation (common/rare/novel)
  5. Failure mode heatmap (model x check)
  6. Novelty-quality analysis (novel structures that also pass)
  7. Diversity metrics

Usage:
    python eval/analyze_variants.py
    python eval/analyze_variants.py --outdir eval/analysis_report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

VARIANTS = {
    "v1": "V1 anon_src (comp+stoich)",
    "v2": "V2 elements_causal",
    "v3": "V3 elements_only",
    "v4": "V4 chem_causal (SG)",
    "v5": "V5 props_v2+arch",
}

DATA_FILES = {
    "v1": "data/small_d4_props_anon_src.csv",
    "v2": "data/small_d5_elements_causal_tgt.csv",
    "v3": "data/small_d5_elements_only_causal_tgt.csv",
    "v4": "data/small_d5_chem_causal_tgt.csv",
    "v5": "data/small_d4_props_anon_causal_tgt.csv",
}

RESULTS_ROOT = _PROJECT_ROOT / "eval" / "uncond_1000"

TIER0_CHECKS = [
    "element_filter_pass", "min_dist_pass", "isolated_atoms_pass",
    "smact_pass", "bvs_pass", "spacegroup_pass",
]

# SG number -> crystal system
_SG_MAP = {}
for _lo, _hi, _name in [
    (1, 2, "triclinic"), (3, 15, "monoclinic"), (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"), (143, 167, "trigonal"), (168, 194, "hexagonal"),
    (195, 230, "cubic"),
]:
    for _sg in range(_lo, _hi + 1):
        _SG_MAP[_sg] = _name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_variant(vid):
    """Load results.csv and summary.json for one variant."""
    d = RESULTS_ROOT / vid
    results_path = d / "results.csv"
    summary_path = d / "summary.json"

    if not results_path.exists():
        return None, None

    with open(results_path) as f:
        rows = list(csv.DictReader(f))

    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    return rows, summary


def load_relaxed_atoms(vid):
    """Load relaxed structures for structural analysis."""
    p = RESULTS_ROOT / vid / "relaxed.extxyz"
    if not p.exists():
        return []
    from ase.io import read as ase_read
    return ase_read(str(p), index=":")


def load_training_stats(data_csv):
    """Load formula and element set stats from training data."""
    formulas = Counter()
    element_sets = Counter()

    if not Path(data_csv).exists():
        return formulas, element_sets

    with open(data_csv) as f:
        for row in csv.DictReader(f):
            if row.get("label", "") in ("eval", "test_1k", "test_100"):
                continue
            tgt = row.get("target", "")
            if not tgt:
                continue
            # Extract elements from target
            elems = frozenset(re.findall(r'[A-Z][a-z]?', tgt.split("|")[0]))
            if len(elems) >= 2:
                element_sets[elems] += 1
            # Rough formula from first segment
            formula_seg = tgt.split("|")[0].strip()
            if formula_seg:
                formulas[formula_seg] += 1

    return formulas, element_sets


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _bool(val):
    """Parse CSV boolean."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _extract_elements(formula):
    """Extract element set from formula string."""
    if not formula:
        return frozenset()
    return frozenset(re.findall(r'[A-Z][a-z]?', formula))


def _sg_to_system(sg_str):
    """Convert spacegroup number string to crystal system."""
    try:
        sg = int(float(sg_str))
        return _SG_MAP.get(sg, "unknown")
    except (ValueError, TypeError):
        return "unknown"


def _pct(n, d):
    return f"{100*n/d:.0f}%" if d > 0 else "-"


# ---------------------------------------------------------------------------
# Analysis 1: Overall comparison
# ---------------------------------------------------------------------------

def analysis_overall(all_data):
    """Print overall comparison table."""
    print(f"\n{'='*100}")
    print("1. OVERALL COMPARISON")
    print(f"{'='*100}")

    header = f"{'Variant':30s} {'Gen':>5s} {'Tier0':>6s} {'Conv':>6s} {'Ehull':>6s} {'Pass':>6s} {'Pass%':>6s} {'Formulas':>9s}"
    print(header)
    print("-" * 80)

    for vid, label in VARIANTS.items():
        rows, summary = all_data.get(vid, (None, None))
        if rows is None:
            print(f"{label:30s} {'(no data)':>5s}")
            continue

        n_gen = sum(1 for r in rows if r.get("gen_status") == "SUCCESS" or r.get("formula"))
        n_tier0 = sum(1 for r in rows if all(_bool(r.get(k)) for k in TIER0_CHECKS))
        n_conv = sum(1 for r in rows if _bool(r.get("converged")))
        n_ehull = sum(1 for r in rows if _bool(r.get("ehull_pass")))
        n_pass = sum(1 for r in rows if _bool(r.get("overall_pass")))
        formulas = len(set(r.get("formula", "") for r in rows if r.get("formula")))

        print(f"{label:30s} {n_gen:>5} {n_tier0:>6} {n_conv:>6} {n_ehull:>6} "
              f"{n_pass:>6} {_pct(n_pass, n_gen):>6s} {formulas:>9}")


# ---------------------------------------------------------------------------
# Analysis 2: Pass rate by chemical complexity
# ---------------------------------------------------------------------------

def analysis_by_complexity(all_data):
    """Pass rate bucketed by number of elements."""
    print(f"\n{'='*100}")
    print("2. PASS RATE BY CHEMICAL COMPLEXITY (n_elements)")
    print(f"{'='*100}")

    buckets = [2, 3, 4, 5]  # 2-element, 3-element, 4-element, 5+
    bucket_labels = ["binary", "ternary", "quaternary", "5+"]

    header = f"{'Variant':30s}" + "".join(f"  {bl:>12s}" for bl in bucket_labels)
    print(header)
    print("-" * (30 + 14 * len(bucket_labels)))

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        counts = defaultdict(lambda: [0, 0])  # bucket -> [pass, total]
        for r in rows:
            formula = r.get("formula", "")
            if not formula:
                continue
            n_el = len(_extract_elements(formula))
            if n_el < 2:
                continue
            bucket = min(n_el, 5)
            counts[bucket][1] += 1
            if _bool(r.get("overall_pass")):
                counts[bucket][0] += 1

        parts = []
        for b, bl in zip(buckets, bucket_labels):
            p, t = counts[b]
            parts.append(f"  {p:>4}/{t:<4} {_pct(p, t):>4s}")
        print(f"{label:30s}" + "".join(parts))


# ---------------------------------------------------------------------------
# Analysis 3: Pass rate by crystal system
# ---------------------------------------------------------------------------

def analysis_by_crystal_system(all_data):
    """Pass rate by crystal system (from spacegroup_computed)."""
    print(f"\n{'='*100}")
    print("3. PASS RATE BY CRYSTAL SYSTEM")
    print(f"{'='*100}")

    systems = ["cubic", "hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"]

    header = f"{'Variant':30s}" + "".join(f"  {s[:8]:>10s}" for s in systems)
    print(header)
    print("-" * (30 + 12 * len(systems)))

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        counts = defaultdict(lambda: [0, 0])
        for r in rows:
            sg = r.get("spacegroup_computed", "")
            if not sg:
                continue
            cs = _sg_to_system(sg)
            counts[cs][1] += 1
            if _bool(r.get("overall_pass")):
                counts[cs][0] += 1

        parts = []
        for s in systems:
            p, t = counts[s]
            if t > 0:
                parts.append(f"  {p:>3}/{t:<3} {_pct(p, t):>4s}")
            else:
                parts.append(f"  {'':>10s}")
        print(f"{label:30s}" + "".join(parts))

    # Also print distribution of crystal systems per model
    print(f"\n  Crystal system distribution (% of generated):")
    header2 = f"{'Variant':30s}" + "".join(f"  {s[:8]:>10s}" for s in systems)
    print(header2)
    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue
        total = sum(1 for r in rows if r.get("spacegroup_computed"))
        parts = []
        for s in systems:
            n = sum(1 for r in rows if _sg_to_system(r.get("spacegroup_computed", "")) == s)
            parts.append(f"  {_pct(n, total):>10s}")
        print(f"{label:30s}" + "".join(parts))


# ---------------------------------------------------------------------------
# Analysis 4: Pass rate by training-set representation
# ---------------------------------------------------------------------------

def analysis_by_representation(all_data):
    """Pass rate by how common the chemical system is in training data."""
    print(f"\n{'='*100}")
    print("4. PASS RATE BY TRAINING-SET REPRESENTATION")
    print(f"{'='*100}")
    print("  common: element set appears 100+ times in training")
    print("  rare: 10-99 times")
    print("  novel: <10 times (including 0)")

    rep_labels = ["common", "rare", "novel"]

    # Load training stats for each variant's dataset
    train_stats = {}
    for vid in VARIANTS:
        data_csv = _PROJECT_ROOT / DATA_FILES[vid]
        _, element_sets = load_training_stats(str(data_csv))
        train_stats[vid] = element_sets

    header = f"{'Variant':30s}" + "".join(f"  {rl:>12s}" for rl in rep_labels)
    print(header)
    print("-" * (30 + 14 * len(rep_labels)))

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        el_counts = train_stats[vid]
        counts = defaultdict(lambda: [0, 0])

        for r in rows:
            formula = r.get("formula", "")
            if not formula:
                continue
            elems = _extract_elements(formula)
            if len(elems) < 2:
                continue

            train_count = el_counts.get(elems, 0)
            if train_count >= 100:
                bucket = "common"
            elif train_count >= 10:
                bucket = "rare"
            else:
                bucket = "novel"

            counts[bucket][1] += 1
            if _bool(r.get("overall_pass")):
                counts[bucket][0] += 1

        parts = []
        for rl in rep_labels:
            p, t = counts[rl]
            parts.append(f"  {p:>4}/{t:<4} {_pct(p, t):>4s}")
        print(f"{label:30s}" + "".join(parts))


# ---------------------------------------------------------------------------
# Analysis 5: Failure mode breakdown
# ---------------------------------------------------------------------------

def analysis_failure_modes(all_data):
    """Which check kills each model?"""
    print(f"\n{'='*100}")
    print("5. FAILURE MODE BREAKDOWN")
    print(f"{'='*100}")

    # Use tier0_ prefixed columns (present for all structures)
    tier0_checks = [
        "tier0_element_filter_pass", "tier0_min_dist_pass", "tier0_isolated_atoms_pass",
        "tier0_smact_pass", "tier0_bvs_pass", "tier0_spacegroup_pass",
    ]
    short_names = [c.replace("tier0_", "").replace("_pass", "") for c in tier0_checks]

    print(f"\n  Tier-0 pass rates (% of generated):")
    header = f"{'Variant':30s}" + "".join(f"  {s:>12s}" for s in short_names) + f"  {'tier0_all':>10s}"
    print(header)
    print("-" * (30 + 14 * len(tier0_checks) + 12))

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        n_gen = sum(1 for r in rows if r.get("formula"))
        parts = []
        for c in tier0_checks:
            n_pass = sum(1 for r in rows if r.get(c) == "True" or r.get(c) is True)
            parts.append(f"  {_pct(n_pass, n_gen):>12s}")
        n_tier0 = sum(1 for r in rows if r.get("tier0_pass") == "True" or r.get("tier0_pass") is True)
        parts.append(f"  {_pct(n_tier0, n_gen):>10s}")
        print(f"{label:30s}" + "".join(parts))

    # Primary tier-0 failure reason
    print(f"\n  Primary tier-0 failure reason (first check to fail):")
    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        primary = Counter()
        for r in rows:
            if not r.get("formula"):
                continue
            t0 = r.get("tier0_pass")
            if t0 == "True" or t0 is True:
                continue
            for c in tier0_checks:
                val = r.get(c)
                if val == "False" or val is False:
                    primary[c.replace("tier0_", "").replace("_pass", "")] += 1
                    break
            else:
                primary["unknown"] += 1

        n_fail = sum(primary.values())
        top = primary.most_common(5)
        top_str = ", ".join(f"{k}:{v} ({_pct(v, n_fail)})" for k, v in top)
        print(f"  {label:30s} {top_str}")

    # Ehull pass rate (among screened structures only)
    print(f"\n  Ehull pass rate (among tier-0 passers that were screened):")
    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue
        screened = [r for r in rows if r.get("ehull_pass") in ("True", "False", True, False)]
        n_screened = len(screened)
        n_ehull = sum(1 for r in screened if r.get("ehull_pass") == "True" or r.get("ehull_pass") is True)
        n_overall = sum(1 for r in rows if r.get("overall_pass") == "True" or r.get("overall_pass") is True)
        print(f"  {label:30s} ehull: {n_ehull}/{n_screened} ({_pct(n_ehull, n_screened)})  overall: {n_overall}")


# ---------------------------------------------------------------------------
# Analysis 6: Novelty vs quality
# ---------------------------------------------------------------------------

def analysis_novelty_quality(all_data):
    """Novel structures that also pass screening."""
    print(f"\n{'='*100}")
    print("6. NOVELTY vs QUALITY")
    print(f"{'='*100}")
    print("  novel: element combination not in training set")
    print("  variant: known elements, new stoichiometry")
    print("  copy: exact formula match")

    # Load training stats
    train_stats = {}
    for vid in VARIANTS:
        data_csv = _PROJECT_ROOT / DATA_FILES[vid]
        formulas, element_sets = load_training_stats(str(data_csv))
        train_stats[vid] = (formulas, element_sets)

    tiers = ["copy", "variant", "novel"]
    header = f"{'Variant':30s}" + "".join(f"  {t:>16s}" for t in tiers) + f"  {'novel+pass':>12s}"
    print(header)
    print("-" * (30 + 18 * len(tiers) + 14))

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        formulas_train, el_sets_train = train_stats[vid]
        tier_counts = defaultdict(lambda: [0, 0])  # tier -> [pass, total]

        for r in rows:
            formula = r.get("formula", "")
            if not formula:
                continue
            elems = _extract_elements(formula)

            if elems not in el_sets_train:
                tier = "novel"
            elif formula not in formulas_train:
                tier = "variant"
            else:
                tier = "copy"

            tier_counts[tier][1] += 1
            if _bool(r.get("overall_pass")):
                tier_counts[tier][0] += 1

        parts = []
        for t in tiers:
            p, tot = tier_counts[t]
            parts.append(f"  {p:>4}/{tot:<4} {_pct(p, tot):>5s}")

        # Novel structures that pass
        novel_pass = tier_counts["novel"][0] + tier_counts["variant"][0]
        novel_total = tier_counts["novel"][1] + tier_counts["variant"][1]
        parts.append(f"  {novel_pass:>4}/{novel_total:<4} {_pct(novel_pass, novel_total):>4s}")
        print(f"{label:30s}" + "".join(parts))


# ---------------------------------------------------------------------------
# Analysis 7: Diversity metrics
# ---------------------------------------------------------------------------

def analysis_diversity(all_data):
    """Diversity: unique formulas, SGs, elements, element combinations."""
    print(f"\n{'='*100}")
    print("7. DIVERSITY METRICS")
    print(f"{'='*100}")

    header = f"{'Variant':30s} {'Formulas':>9s} {'SGs':>5s} {'Elements':>9s} {'ElemCombos':>11s} {'MeanNatoms':>11s}"
    print(header)
    print("-" * 80)

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        formulas = set()
        sgs = set()
        all_elements = set()
        element_combos = set()
        natoms_list = []

        for r in rows:
            formula = r.get("formula", "")
            if formula:
                formulas.add(formula)
                elems = _extract_elements(formula)
                all_elements.update(elems)
                element_combos.add(elems)

            sg = r.get("spacegroup_computed", "")
            if sg:
                sgs.add(sg)

            # Estimate natoms from formula
            nums = re.findall(r'(\d+)', formula)
            if nums:
                natoms_list.append(sum(int(n) for n in nums))

        mean_natoms = f"{np.mean(natoms_list):.1f}" if natoms_list else "-"
        print(f"{label:30s} {len(formulas):>9} {len(sgs):>5} {len(all_elements):>9} "
              f"{len(element_combos):>11} {mean_natoms:>11s}")


# ---------------------------------------------------------------------------
# Analysis 8: Relaxation convergence
# ---------------------------------------------------------------------------

def analysis_relaxation(all_data):
    """Relaxation convergence and time statistics."""
    print(f"\n{'='*100}")
    print("8. RELAXATION STATISTICS")
    print(f"{'='*100}")

    header = f"{'Variant':30s} {'Conv%':>6s} {'Timeout%':>9s} {'MeanTime':>9s} {'MedianTime':>11s}"
    print(header)
    print("-" * 70)

    for vid, label in VARIANTS.items():
        rows, _ = all_data.get(vid, (None, None))
        if rows is None:
            continue

        n_total = sum(1 for r in rows if r.get("formula"))
        n_conv = sum(1 for r in rows if _bool(r.get("converged")))
        n_timeout = sum(1 for r in rows if _bool(r.get("relax_timeout")))

        times = []
        for r in rows:
            t = r.get("relax_time")
            if t:
                try:
                    times.append(float(t))
                except ValueError:
                    pass

        mean_t = f"{np.mean(times):.1f}s" if times else "-"
        med_t = f"{np.median(times):.1f}s" if times else "-"

        print(f"{label:30s} {_pct(n_conv, n_total):>6s} {_pct(n_timeout, n_total):>9s} "
              f"{mean_t:>9s} {med_t:>11s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Cross-variant analysis of 1000-structure results")
    p.add_argument("--outdir", default=None, help="Save text report and JSON summary")
    p.add_argument("--variants", default=None, help="Comma-separated variant IDs (default: all)")
    args = p.parse_args()

    variants_to_run = args.variants.split(",") if args.variants else list(VARIANTS.keys())

    # Load all data
    all_data = {}
    available = []
    for vid in variants_to_run:
        rows, summary = load_variant(vid)
        if rows is not None:
            all_data[vid] = (rows, summary)
            available.append(vid)
            print(f"Loaded {vid}: {len(rows)} rows")
        else:
            print(f"Skipped {vid}: no results.csv")

    if not all_data:
        print("No variant data found. Check eval/uncond_1000/{v1..v5}/results.csv")
        sys.exit(1)

    print(f"\nAnalyzing {len(available)} variants: {', '.join(available)}")

    # Run all analyses
    analysis_overall(all_data)
    analysis_by_complexity(all_data)
    analysis_by_crystal_system(all_data)
    analysis_by_representation(all_data)
    analysis_failure_modes(all_data)
    analysis_novelty_quality(all_data)
    analysis_diversity(all_data)
    analysis_relaxation(all_data)

    # Save summary JSON
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Build summary dict
        summary = {}
        for vid in available:
            rows, _ = all_data[vid]
            n_gen = sum(1 for r in rows if r.get("formula"))
            n_pass = sum(1 for r in rows if _bool(r.get("overall_pass")))
            summary[vid] = {
                "label": VARIANTS[vid],
                "n_generated": n_gen,
                "n_overall_pass": n_pass,
                "pass_rate": round(n_pass / max(n_gen, 1), 4),
            }
        (outdir / "variant_comparison.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {outdir}/variant_comparison.json")


if __name__ == "__main__":
    main()

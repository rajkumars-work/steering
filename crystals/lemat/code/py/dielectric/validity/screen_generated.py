"""Screen generated crystal structures for physical validity.

Runs the checks in structural.py on every structure in an extxyz file and
prints a per-check pass-rate summary.  By default only the cheap Tier-0
checks (min_dist, SMACT) are run.  Pass --full to also run the MLIP-based
checks (UQ, e-hull, phonons; much slower).

Usage
-----
# Tier-0 only (fast, no GPU needed):
    python screen_generated.py generated.extxyz

# Full pipeline (requires MACE + PET loaded on GPU):
    python screen_generated.py generated.extxyz --full

# Limit to first 200 structures:
    python screen_generated.py generated.extxyz --max 200

# Save detailed per-structure CSV:
    python screen_generated.py generated.extxyz --csv results.csv

# Save passing structures to extxyz:
    python screen_generated.py generated.extxyz --output passing.extxyz

# Run all checks on every structure (no early termination):
    python screen_generated.py generated.extxyz --full --no-funnel
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys

import numpy as np
from ase.io import read as ase_read, write as ase_write

from .structural import (
    check_element_filter, check_min_dist, check_isolated_atoms,
    check_smact, check_bvs, check_spacegroup,
    compute_uniqueness, compute_novelty,
    compute_uniqueness_mask, compute_novelty_mask,
)

DEFAULT_STRUCTURE_TIMEOUT = 120  # seconds per structure


class _StructureTimeout(Exception):
    """Raised when a single structure evaluation exceeds the time limit."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass_rate(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return "n/a"
    n_pass = sum(bool(v) for v in vals)
    return f"{n_pass}/{len(vals)} ({100 * n_pass / len(vals):.1f} %)"


def _print_summary(rows: list[dict], full: bool) -> None:
    tier0_keys = ["element_filter_pass", "min_dist_pass", "isolated_atoms_pass",
                  "smact_pass", "bvs_pass", "spacegroup_pass"]
    mlip_keys  = ["uq_pass", "ehull_pass", "phonon_pass",
                  "cross_pass", "md_pass", "overall_pass"]

    keys = tier0_keys + (mlip_keys if full else [])
    labels = {
        "element_filter_pass":  "No radioactive / blocked elements",
        "min_dist_pass":        "Min interatomic distance ≥ 0.5 Å",
        "isolated_atoms_pass":  "No isolated atoms (cutoff 4 Å)",
        "smact_pass":           "SMACT charge balance / electronegativity",
        "bvs_pass":             "Bond Valence Sum self-consistent",
        "spacegroup_pass":      "Spacegroup matches target (pymatgen)",
        "uq_pass":              "MLIP uncertainty ≤ 0.05 eV/atom",
        "ehull_pass":           "E above hull ≤ 0.1 eV/atom",
        "phonon_pass":          "No imaginary phonon modes",
        "cross_pass":           "PET–MACE energy agreement (soft)",
        "md_pass":              "NVT-MD stability (soft)",
        "overall_pass":         "Overall (all hard rejects passed)",
    }

    col = max(len(v) for v in labels.values()) + 2
    print()
    print(f"{'Check':<{col}} Pass rate")
    print("-" * (col + 22))
    for k in keys:
        if any(k in r for r in rows):
            print(f"{labels.get(k, k):<{col}} {_pass_rate(rows, k)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    timeout = getattr(args, "timeout", DEFAULT_STRUCTURE_TIMEOUT)
    print(f"Reading structures from {args.extxyz} …")
    atoms_list = ase_read(args.extxyz, index=":")
    if args.max:
        atoms_list = atoms_list[: args.max]
    print(f"  {len(atoms_list)} structures loaded")

    funnel = getattr(args, "funnel", True)
    if args.full:
        from .structural import load_mace, load_pet, run_checks
        print("Loading MLIP calculators …")
        if not funnel:
            print("  (funnel disabled — all checks run on every structure)")
        pet  = load_pet(device=args.device)
        mace = load_mace(device=args.device)
        ehull_cache: dict = {}

    rows: list[dict] = []
    for i, atoms in enumerate(atoms_list):
        formula = atoms.get_chemical_formula()
        print(f"  [{i + 1}/{len(atoms_list)}] {formula}", end="\r")

        if args.full:
            def _timeout_handler(signum, frame):
                raise _StructureTimeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                result = run_checks(atoms, pet, mace, ehull_cache,
                                    run_md=args.md, run_cross=args.cross,
                                    funnel=funnel)
            except _StructureTimeout:
                print(f"\n  WARNING: {formula} timed out after "
                      f"{timeout}s — skipping")
                result = {"timed_out": True}
            except Exception as ex:
                print(f"\n  WARNING: {formula} failed: {ex!r} — skipping")
                result = {"error": str(ex)}
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            def _timeout_handler(signum, frame):
                raise _StructureTimeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                result = {}
                result.update(check_element_filter(atoms))
                result.update(check_min_dist(atoms))
                result.update(check_isolated_atoms(atoms))
                result.update(check_smact(atoms))
                result.update(check_bvs(atoms))
                result.update(check_spacegroup(atoms))
            except _StructureTimeout:
                print(f"\n  WARNING: {formula} timed out after "
                      f"{timeout}s — skipping")
                result = {"timed_out": True}
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        rows.append({"index": i, "formula": formula, **result})

    print()  # clear \r line

    _print_summary(rows, full=args.full)

    # Composite: fraction passing ALL checks
    if args.full:
        all_pass = sum(1 for r in rows if r.get("overall_pass"))
    else:
        tier0_keys = ["element_filter_pass", "min_dist_pass", "isolated_atoms_pass",
                      "smact_pass", "bvs_pass", "spacegroup_pass"]
        all_pass = sum(1 for r in rows if all(r.get(k) for k in tier0_keys))
    print(f"All-Tier0-pass (composite)             "
          f"  {all_pass}/{len(rows)} ({100 * all_pass / len(rows):.1f} %)")
    print()

    # Set-level metrics
    print("Computing uniqueness …")
    uniq = compute_uniqueness(atoms_list)
    print(f"  Uniqueness : {uniq['n_unique']}/{uniq['n_total']} "
          f"({100 * uniq['uniqueness']:.1f} %)")

    ref_atoms = None
    if args.reference:
        print(f"Computing novelty against {args.reference} …")
        ref_atoms = ase_read(args.reference, index=":")
        nov = compute_novelty(atoms_list, ref_atoms)
        print(f"  Novelty    : {nov['n_novel']}/{nov['n_total']} "
              f"({100 * nov['novelty']:.1f} %)")
    print()

    # --sun: Stability × Uniqueness × Novelty summary
    if getattr(args, "sun", False):
        n = len(atoms_list)
        # Stability: per-structure e-hull ≤ 0.1 eV/atom (requires --full)
        stable_mask = np.array([bool(r.get("ehull_pass")) for r in rows])

        # Uniqueness & Novelty: per-structure boolean masks
        print("Computing per-structure uniqueness mask …")
        unique_mask = compute_uniqueness_mask(atoms_list)

        if args.reference:
            print("Computing per-structure novelty mask …")
            if ref_atoms is None:
                ref_atoms = ase_read(args.reference, index=":")
            novel_mask = compute_novelty_mask(atoms_list, ref_atoms)
        else:
            # Without a reference set, novelty is undefined — assume all novel
            novel_mask = np.ones(n, dtype=bool)

        sun_mask = stable_mask & unique_mask & novel_mask

        s_frac = stable_mask.sum() / n if n else 0.0
        u_frac = unique_mask.sum() / n if n else 0.0
        n_frac = novel_mask.sum() / n if n else 0.0
        sun_frac = sun_mask.sum() / n if n else 0.0

        col = 42
        print(f"\n{'SUN Score Summary':=^{col + 22}}")
        print(f"{'Metric':<{col}} {'Count':>8}  {'Rate':>8}")
        print("-" * (col + 22))
        print(f"{'Stability  (e-hull ≤ 0.1 eV/atom)':<{col}} "
              f"{int(stable_mask.sum()):>5}/{n}  {100 * s_frac:>6.1f} %")
        print(f"{'Uniqueness (StructureMatcher)':<{col}} "
              f"{int(unique_mask.sum()):>5}/{n}  {100 * u_frac:>6.1f} %")
        if args.reference:
            print(f"{'Novelty    (not in reference set)':<{col}} "
                  f"{int(novel_mask.sum()):>5}/{n}  {100 * n_frac:>6.1f} %")
        else:
            print(f"{'Novelty    (no --reference given, assuming 100%)':<{col}} "
                  f"{int(novel_mask.sum()):>5}/{n}  {100 * n_frac:>6.1f} %")
        print("-" * (col + 22))
        print(f"{'S ∩ U ∩ N  (stable + unique + novel)':<{col}} "
              f"{int(sun_mask.sum()):>5}/{n}  {100 * sun_frac:>6.1f} %")
        print()

        # Append SUN columns to rows for CSV export
        for i, r in enumerate(rows):
            r["sun_stable"] = bool(stable_mask[i])
            r["sun_unique"] = bool(unique_mask[i])
            r["sun_novel"] = bool(novel_mask[i])
            r["sun_pass"] = bool(sun_mask[i])

    if args.csv:
        # Collect ALL unique keys across rows (first row may be incomplete)
        seen = set()
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Per-structure results saved to {args.csv}")

    if args.output:
        if args.full:
            passing = [atoms_list[r["index"]] for r in rows
                       if r.get("overall_pass")]
        else:
            passing = [atoms_list[r["index"]] for r in rows
                       if all(r.get(k) for k in tier0_keys)]
        ase_write(args.output, passing, format="extxyz")
        print(f"Saved {len(passing)} passing structures to {args.output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen generated structures for physical validity.")
    parser.add_argument("extxyz", help="Path to input .extxyz file")
    parser.add_argument("--full", action="store_true",
                        help="Run full MLIP-based checks (UQ, e-hull, phonons)")
    parser.add_argument("--md", action="store_true",
                        help="Include NVT-MD soft-flag check (slow; implies --full)")
    parser.add_argument("--cross", action="store_true", default=True,
                        help="Include cross-MLIP soft-flag check (implies --full)")
    parser.add_argument("--max", type=int, default=None,
                        help="Only process the first N structures")
    parser.add_argument("--device", default="cuda",
                        help="Device for MLIP calculators (default: cuda)")
    parser.add_argument("--reference", default=None,
                        help="Reference extxyz (e.g. training set) for novelty computation")
    parser.add_argument("--sun", action="store_true",
                        help="Compute SUN score (Stability, Uniqueness, Novelty); "
                             "implies --full.  Use --reference for novelty.")
    parser.add_argument("--csv", default=None,
                        help="Save per-structure results to this CSV path")
    parser.add_argument("--output", default=None,
                        help="Save passing structures to this extxyz path")
    parser.add_argument("--timeout", type=int, default=DEFAULT_STRUCTURE_TIMEOUT,
                        help=f"Per-structure timeout in seconds (default: {DEFAULT_STRUCTURE_TIMEOUT})")
    parser.add_argument("--funnel", action="store_true", default=True,
                        help="Stop early on hard rejects (default, faster)")
    parser.add_argument("--no-funnel", dest="funnel", action="store_false",
                        help="Run all checks on every structure (slower, wider sample)")
    return parser


def main_cli() -> None:
    """Entry point for the ``di-check`` command-line tool."""
    args = _build_parser().parse_args()
    if args.md or args.sun:
        args.full = True
    main(args)


if __name__ == "__main__":
    main_cli()

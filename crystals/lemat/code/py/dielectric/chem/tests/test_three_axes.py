"""Smoke + latency test for chem.{stability, novelty, validity}.

Runs all three axes on 10 representative compositions and prints per-call
timing. Does not assert anything beyond "compute completes without exception" —
the goal is sanity + latency.

Run:
    cd /data/rkumar/code/py
    PYTHONPATH=dielectric python dielectric/chem/tests/test_three_axes.py
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Any

from ase import Atoms
from ase.build import bulk


def make_test_set() -> list[tuple[str, Atoms]]:
    """10 compositions covering ionic, oxide, metal, alloy, semiconductor."""
    cases: list[tuple[str, Atoms]] = []
    cases.append(("NaCl_rocksalt", bulk("NaCl", crystalstructure="rocksalt", a=5.64)))
    cases.append(("MgO_rocksalt",  bulk("MgO",  crystalstructure="rocksalt", a=4.21)))
    cases.append(("CaF2_fluorite", bulk("CaF2", crystalstructure="fluorite", a=5.46)))
    cases.append(("ZnS_zincblende", bulk("ZnS", crystalstructure="zincblende", a=5.41)))
    cases.append(("CsCl_b2",       bulk("CsCl", crystalstructure="cesiumchloride", a=4.12)))
    cases.append(("Si_diamond",    bulk("Si",   crystalstructure="diamond", a=5.43)))
    cases.append(("GaAs_zb",       bulk("GaAs", crystalstructure="zincblende", a=5.65)))
    cases.append(("Cu_fcc",        bulk("Cu",   crystalstructure="fcc", a=3.61)))
    cases.append(("Fe_bcc",        bulk("Fe",   crystalstructure="bcc", a=2.87)))
    cases.append(("Al_fcc",        bulk("Al",   crystalstructure="fcc", a=4.05)))
    return cases


def fmt(s: float) -> str:
    return f"{s*1000:7.0f} ms" if s < 10 else f"{s:7.2f} s "


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}", flush=True)


def run_validity(cases: list[tuple[str, Atoms]]) -> None:
    section("AXIS 1: validity (SMACT + BVS + atomic-distance)")
    from chem.validity import check_validity

    print(f"{'Case':22}  {'pass':>4}  {'min_d':>7}  {'bvs':>6}  {'smact':>5}  {'time':>10}")
    print("-" * 72)
    total = 0.0
    for name, atoms in cases:
        t0 = time.perf_counter()
        try:
            r = check_validity(atoms, timeout=30)
            dt = time.perf_counter() - t0
            print(f"{name:22}  {str(r.get('validity_pass')):>4}  "
                  f"{r.get('min_dist_A', 0) or 0:>7.2f}  "
                  f"{(r.get('bvs_max_deviation') if r.get('bvs_max_deviation') is not None else float('nan')):>6.2f}  "
                  f"{str(r.get('smact_pass')):>5}  {fmt(dt)}",
                  flush=True)
            total += dt
        except Exception as e:
            print(f"{name:22}  ERROR: {e}", flush=True)
            traceback.print_exc()
    print("-" * 72)
    print(f"{'total':22}  {fmt(total)}")
    print(f"{'mean / case':22}  {fmt(total / len(cases))}")


def run_novelty(cases: list[tuple[str, Atoms]]) -> None:
    section("AXIS 2: lemat-novelty (StructureMatcher vs LeMat-Bulk)")
    from chem.novelty import check_lemat_novelty, get_lematbulk_checker

    # Warm-load the checker once and time it separately
    t0 = time.perf_counter()
    get_lematbulk_checker()
    print(f"checker load: {fmt(time.perf_counter() - t0)}\n")

    print(f"{'Case':22}  {'novel':>5}  {'time':>10}")
    print("-" * 72)
    total = 0.0
    for name, atoms in cases:
        t0 = time.perf_counter()
        try:
            novel = check_lemat_novelty(atoms, timeout=30)
            dt = time.perf_counter() - t0
            print(f"{name:22}  {str(novel):>5}  {fmt(dt)}", flush=True)
            total += dt
        except Exception as e:
            print(f"{name:22}  ERROR: {e}", flush=True)
            traceback.print_exc()
    print("-" * 72)
    print(f"{'total':22}        {fmt(total)}")
    print(f"{'mean / case':22}        {fmt(total / len(cases))}")


def run_stability(cases: list[tuple[str, Atoms]]) -> None:
    section("AXIS 3: stability (MACE e_above_hull)")
    from chem.stability import compute_e_above_hull, load_stability_calc

    t0 = time.perf_counter()
    calc = load_stability_calc(device="cuda")
    print(f"MACE load: {fmt(time.perf_counter() - t0)}\n")

    mem_cache: dict = {}
    print(f"{'Case':22}  {'pass':>4}  {'e_hull(eV/at)':>14}  {'time':>10}")
    print("-" * 72)
    total = 0.0
    for name, atoms in cases:
        t0 = time.perf_counter()
        try:
            r = compute_e_above_hull(atoms, calc=calc, mem_cache=mem_cache, timeout=30)
            dt = time.perf_counter() - t0
            e = r.get("e_above_hull")
            e_str = f"{e:.4f}" if isinstance(e, float) else str(e)
            err = r.get("ehull_error", "")
            timed = " (timeout)" if r.get("ehull_timed_out") else ""
            print(f"{name:22}  {str(r.get('ehull_pass')):>4}  {e_str:>14}  {fmt(dt)}{timed} {err}",
                  flush=True)
            total += dt
        except Exception as e:
            print(f"{name:22}  ERROR: {e}", flush=True)
            traceback.print_exc()
    print("-" * 72)
    print(f"{'total':22}                    {fmt(total)}")
    print(f"{'mean / case':22}                    {fmt(total / len(cases))}")


def main() -> int:
    cases = make_test_set()
    print(f"Test set: {len(cases)} compositions")
    for name, atoms in cases:
        print(f"  {name:22}  {atoms.get_chemical_formula():12}  ({len(atoms)} atoms)")

    run_validity(cases)
    run_novelty(cases)
    run_stability(cases)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

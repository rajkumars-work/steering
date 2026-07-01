#!/usr/bin/env python3
"""Post-process V4 conditional generation: Egip + surrogates + synthesizability + plots.

Run AFTER cond_compare.py has finished and produced passing.extxyz.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/cond_v4_postprocess.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

OUTDIR = _PROJECT_ROOT / "eval" / "cond_v4_full"
MODEL_DIR = _PROJECT_ROOT / "chem" / "data"


def step1_egip(atoms_list):
    """Compute MLIP band gap and eps_0 via Egip/MLIP."""
    print("\n=== Step 1: Egip/MLIP Band Gap + eps_0 ===")

    # 1a. Band gap via Dielectrics
    try:
        from chem.props.dielectric import Dielectrics
        with Dielectrics(shutdown_ray=False) as di:
            results = di.compute(atoms_list, relax=False, compute_phonons=False)
        n_ok = 0
        for atoms, res in zip(atoms_list, results):
            val = res.get("mlip_bandgap")
            if val is not None:
                atoms.info["egip_bandgap"] = round(float(val), 3)
                n_ok += 1
        print(f"  Egip band gap: {n_ok}/{len(atoms_list)} OK")
    except Exception as e:
        print(f"  Egip band gap FAILED: {e}")
        for a in atoms_list:
            a.info.setdefault("egip_bandgap", float("nan"))

    # 1b. eps_0 via MLIP
    try:
        from chem.props.dielectric import compute_eps_0
        n_ok = 0
        for atoms in atoms_list:
            try:
                eps = compute_eps_0(atoms)
                if eps is not None and eps > 0:  # drop negative values
                    atoms.info["egip_eps_0"] = round(float(eps), 2)
                    n_ok += 1
                else:
                    atoms.info["egip_eps_0"] = float("nan")
            except Exception:
                atoms.info["egip_eps_0"] = float("nan")
        print(f"  Egip eps_0: {n_ok}/{len(atoms_list)} OK (negative values dropped)")
    except Exception as e:
        print(f"  Egip eps_0 FAILED: {e}")
        for a in atoms_list:
            a.info.setdefault("egip_eps_0", float("nan"))


def step2_surrogates(atoms_list):
    """Compute surrogate band gap and eps_0 from composition features."""
    print("\n=== Step 2: Composition Surrogates ===")
    from chem.surrogates.eps0 import compute_composition_features

    eps_model = xgb.Booster()
    eps_model.load_model(str(MODEL_DIR / "xgb_composition_dft_eps_0.json"))
    bg_model = xgb.Booster()
    bg_model.load_model(str(MODEL_DIR / "xgb_composition_dft_band_gap.json"))

    n_ok = 0
    for atoms in atoms_list:
        try:
            features = compute_composition_features(atoms)
            dm = xgb.DMatrix(features.reshape(1, -1))
            atoms.info["surrogate_band_gap"] = round(max(0.0, float(bg_model.predict(dm)[0])), 3)
            atoms.info["surrogate_eps_0"] = round(max(1.0, float(eps_model.predict(dm)[0])), 2)
            n_ok += 1
        except Exception:
            atoms.info.setdefault("surrogate_band_gap", float("nan"))
            atoms.info.setdefault("surrogate_eps_0", float("nan"))
    print(f"  Surrogates: {n_ok}/{len(atoms_list)} OK")


def step3_synthesizability(atoms_list):
    """Compute synthesizability score on passing structures."""
    print("\n=== Step 3: Synthesizability ===")
    try:
        from scripts.synthesizability_score import SynthesizabilityScorer
        scorer = SynthesizabilityScorer()

        # We need screen results for the scorer — build minimal dict from atoms.info
        n_ok = 0
        for atoms in atoms_list:
            try:
                screen_results = {
                    "e_above_hull": atoms.info.get("e_above_hull"),
                    "smact_pass": atoms.info.get("smact_pass", True),
                    "bvs_pass": atoms.info.get("bvs_pass", True),
                    "bvs_max_deviation": atoms.info.get("bvs_max_deviation"),
                }
                report = scorer.score(atoms, screen_results=screen_results)
                atoms.info["synth_score"] = round(report.composite_score, 1)
                atoms.info["synth_verdict"] = report.verdict
                n_ok += 1
            except Exception as e:
                atoms.info["synth_score"] = float("nan")
                atoms.info["synth_error"] = str(e)
        print(f"  Synthesizability: {n_ok}/{len(atoms_list)} OK")
    except Exception as e:
        print(f"  Synthesizability scorer FAILED to load: {e}")
        for a in atoms_list:
            a.info.setdefault("synth_score", float("nan"))


def step4_rank_and_report(atoms_list):
    """Rank by composite score and report top 10."""
    print("\n=== Step 4: Ranking ===")

    # Target: band_gap ~ 4 eV, eps_0 < 3
    # Composite score: closeness to target properties + synthesizability bonus
    # Lower is better (distance metric)

    ranked = []
    for atoms in atoms_list:
        formula = atoms.get_chemical_formula(mode="hill")

        # Use Egip values if available, else surrogate
        egip_bg = atoms.info.get("egip_bandgap", float("nan"))
        egip_eps = atoms.info.get("egip_eps_0", float("nan"))
        surr_bg = atoms.info.get("surrogate_band_gap", float("nan"))
        surr_eps = atoms.info.get("surrogate_eps_0", float("nan"))
        synth = atoms.info.get("synth_score", float("nan"))
        ehull = atoms.info.get("e_above_hull", float("nan"))

        # Property distance: how far from target (bg=4, eps=2)?
        bg_val = egip_bg if not math.isnan(egip_bg) else surr_bg
        eps_val = egip_eps if not math.isnan(egip_eps) else surr_eps
        if math.isnan(bg_val) or math.isnan(eps_val):
            prop_dist = 999
        else:
            # Normalized L2 distance: bg target=4, scale=4; eps target=2, scale=3
            prop_dist = math.sqrt(((bg_val - 4.0) / 4.0) ** 2 + ((eps_val - 2.0) / 3.0) ** 2)

        # Synthesizability bonus: higher synth_score = more synthesizable
        synth_bonus = 0
        if not math.isnan(synth):
            synth_bonus = -synth / 200.0  # scale to ~-0.5 for score=100

        # Stability bonus: lower ehull = more stable
        stab_bonus = 0
        if not math.isnan(ehull):
            stab_bonus = ehull  # penalize by ehull directly

        composite = prop_dist + stab_bonus + synth_bonus

        ranked.append({
            "formula": formula,
            "egip_bg": egip_bg,
            "egip_eps": egip_eps,
            "surr_bg": surr_bg,
            "surr_eps": surr_eps,
            "ehull": ehull,
            "synth": synth,
            "composite": composite,
            "atoms": atoms,
        })

    ranked.sort(key=lambda x: x["composite"])

    # Report
    print(f"\n{'='*90}")
    print(f"TOP 10 CANDIDATES (wide-gap + low-k, ranked by composite score)")
    print(f"{'='*90}")
    print(f"{'#':>3s} {'Formula':22s} {'Egip BG':>8s} {'Egip eps':>8s} {'Surr BG':>8s} {'Surr eps':>8s} "
          f"{'Ehull':>7s} {'Synth':>6s} {'Score':>7s}")
    print("-" * 100)

    for i, r in enumerate(ranked[:10]):
        egip = f"{r['egip_bg']:.2f}" if not math.isnan(r['egip_bg']) else "N/A"
        eeps = f"{r['egip_eps']:.1f}" if not math.isnan(r['egip_eps']) else "N/A"
        sbg = f"{r['surr_bg']:.2f}" if not math.isnan(r['surr_bg']) else "N/A"
        seps = f"{r['surr_eps']:.1f}" if not math.isnan(r['surr_eps']) else "N/A"
        eh = f"{r['ehull']:.3f}" if not math.isnan(r['ehull']) else "N/A"
        syn = f"{r['synth']:.0f}" if not math.isnan(r['synth']) else "N/A"
        print(f"{i+1:3d} {r['formula']:22s} {egip:>8s} {eeps:>8s} {sbg:>8s} {seps:>8s} "
              f"{eh:>7s} {syn:>6s} {r['composite']:>7.3f}")

    # Save full ranking
    csv_path = OUTDIR / "ranked_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "formula", "egip_bandgap", "egip_eps_0", "surrogate_bandgap",
            "surrogate_eps_0", "e_above_hull", "synth_score", "composite_score",
        ])
        w.writeheader()
        for i, r in enumerate(ranked):
            w.writerow({
                "rank": i + 1,
                "formula": r["formula"],
                "egip_bandgap": r["egip_bg"],
                "egip_eps_0": r["egip_eps"],
                "surrogate_bandgap": r["surr_bg"],
                "surrogate_eps_0": r["surr_eps"],
                "e_above_hull": r["ehull"],
                "synth_score": r["synth"],
                "composite_score": round(r["composite"], 4),
            })
    print(f"\nSaved ranking to {csv_path}")

    return ranked


def step5_plots(atoms_list):
    """Plot band gap and eps_0 distributions — separate panels for Egip vs surrogate."""
    print("\n=== Step 5: Plots ===")

    egip_bg = [a.info.get("egip_bandgap", float("nan")) for a in atoms_list]
    egip_eps = [a.info.get("egip_eps_0", float("nan")) for a in atoms_list]
    surr_bg = [a.info.get("surrogate_band_gap", float("nan")) for a in atoms_list]
    surr_eps = [a.info.get("surrogate_eps_0", float("nan")) for a in atoms_list]

    # Clean NaNs (keep paired for scatter)
    egip_pairs = [(bg, eps) for bg, eps in zip(egip_bg, egip_eps)
                  if not math.isnan(bg) and not math.isnan(eps)]
    surr_pairs = [(bg, eps) for bg, eps in zip(surr_bg, surr_eps)
                  if not math.isnan(bg) and not math.isnan(eps)]

    egip_bg_clean = [v for v in egip_bg if not math.isnan(v)]
    egip_eps_clean = [v for v in egip_eps if not math.isnan(v)]
    surr_bg_clean = [v for v in surr_bg if not math.isnan(v)]
    surr_eps_clean = [v for v in surr_eps if not math.isnan(v)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Surrogate
    # 1a. Scatter: Surrogate BG vs eps_0
    ax = axes[0, 0]
    if surr_pairs:
        bgs, epss = zip(*surr_pairs)
        ax.scatter(bgs, epss, alpha=0.6, s=50, c="steelblue", edgecolors="k", linewidths=0.3)
    ax.axvline(3.0, color="red", ls="--", alpha=0.5, label="wide-gap (3 eV)")
    ax.axhline(3.0, color="orange", ls="--", alpha=0.5, label="low-k (eps=3)")
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("eps_0")
    ax.set_title(f"Surrogate: BG vs eps_0 (n={len(surr_pairs)})")
    ax.legend(fontsize=8)

    # 1b. Histogram: Surrogate BG
    ax = axes[0, 1]
    bins_bg = np.linspace(0, 8, 17)
    ax.hist(surr_bg_clean, bins=bins_bg, alpha=0.7, color="steelblue")
    ax.axvline(3.0, color="red", ls="--", alpha=0.7, label="target: wide-gap")
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Surrogate Band Gap (n={len(surr_bg_clean)})")
    ax.legend(fontsize=8)

    # 1c. Histogram: Surrogate eps_0
    ax = axes[0, 2]
    if surr_eps_clean:
        eps_max = min(20, max(surr_eps_clean) + 1)
        ax.hist(surr_eps_clean, bins=np.linspace(0, eps_max, 16), alpha=0.7, color="steelblue")
    ax.axvline(3.0, color="orange", ls="--", alpha=0.7, label="target: low-k")
    ax.set_xlabel("eps_0")
    ax.set_ylabel("Count")
    ax.set_title(f"Surrogate eps_0 (n={len(surr_eps_clean)})")
    ax.legend(fontsize=8)

    # Row 2: Egip/MLIP
    # 2a. Scatter: Egip BG vs eps_0
    ax = axes[1, 0]
    if egip_pairs:
        bgs, epss = zip(*egip_pairs)
        ax.scatter(bgs, epss, alpha=0.6, s=50, c="coral", edgecolors="k", linewidths=0.3)
    ax.axvline(3.0, color="red", ls="--", alpha=0.5, label="wide-gap (3 eV)")
    ax.axhline(3.0, color="orange", ls="--", alpha=0.5, label="low-k (eps=3)")
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("eps_0")
    ax.set_title(f"Egip/MLIP: BG vs eps_0 (n={len(egip_pairs)})")
    ax.legend(fontsize=8)

    # 2b. Histogram: Egip BG
    ax = axes[1, 1]
    ax.hist(egip_bg_clean, bins=bins_bg, alpha=0.7, color="coral")
    ax.axvline(3.0, color="red", ls="--", alpha=0.7, label="target: wide-gap")
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Egip Band Gap (n={len(egip_bg_clean)})")
    ax.legend(fontsize=8)

    # 2c. Histogram: Egip eps_0
    ax = axes[1, 2]
    if egip_eps_clean:
        eps_max = min(20, max(egip_eps_clean) + 1)
        ax.hist(egip_eps_clean, bins=np.linspace(0, eps_max, 16), alpha=0.7, color="coral")
    ax.axvline(3.0, color="orange", ls="--", alpha=0.7, label="target: low-k")
    ax.set_xlabel("eps_0")
    ax.set_ylabel("Count")
    ax.set_title(f"Egip eps_0 (n={len(egip_eps_clean)})")
    ax.legend(fontsize=8)

    plt.suptitle("V4 Conditional Generation: Wide-Gap + Low-K Target\n"
                 "Top row: Surrogate predictions  |  Bottom row: Egip/MLIP predictions\n"
                 "(passing structures only, negative eps_0 dropped)", fontsize=12)
    plt.tight_layout()

    plot_path = OUTDIR / "property_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plots to {plot_path}")


def main():
    from ase.io import read as ase_read

    passing_path = OUTDIR / "passing.extxyz"
    if not passing_path.exists():
        print(f"Error: {passing_path} not found. Run cond_compare.py first.")
        sys.exit(1)

    atoms_list = ase_read(str(passing_path), index=":")
    print(f"Loaded {len(atoms_list)} passing structures from {passing_path}")

    # Load screen results to attach ehull etc. to atoms
    results_csv = OUTDIR / "results.csv"
    if results_csv.exists():
        with open(results_csv) as f:
            all_rows = list(csv.DictReader(f))
        # Match by formula (index may not be present)
        screen_by_formula = {}
        for r in all_rows:
            if r.get("overall_pass") == "True":
                screen_by_formula[r.get("formula", "")] = r
        for atoms in atoms_list:
            formula = atoms.get_chemical_formula()
            r = screen_by_formula.get(formula)
            if r:
                for k in ("e_above_hull", "bvs_pass", "smact_pass", "bvs_max_deviation"):
                    if k in r and r[k]:
                        try:
                            atoms.info[k] = float(r[k]) if "." in str(r[k]) else r[k]
                        except (ValueError, TypeError):
                            atoms.info[k] = r[k]

    step1_egip(atoms_list)
    step2_surrogates(atoms_list)
    step3_synthesizability(atoms_list)
    ranked = step4_rank_and_report(atoms_list)
    step5_plots(atoms_list)

    # Save enriched extxyz
    from ase.io import write as ase_write
    ase_write(str(OUTDIR / "passing_enriched.extxyz"), atoms_list, format="extxyz")
    print(f"\nSaved enriched structures to {OUTDIR / 'passing_enriched.extxyz'}")


if __name__ == "__main__":
    main()

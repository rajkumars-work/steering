#!/usr/bin/env python3
"""V4 conditional generation: 250 structures → novelty filter → relax → BVS → ehull → properties → top 10.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/run_v4_cond_novel_250.py
"""
import sys, os, time, csv, re, copy, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ed"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read as ase_read, write as ase_write
from ase.optimize import LBFGS

OUTDIR = Path("eval/cond_v4_novel_250")
DATA = "data/small_d5_chem_causal_tgt.csv"
CKPT_DIR = Path("/data/rkumar/code/py/ed/checkpoints/sweep_novelty/v4_chem_causal")
N = 250
SEED = 42


def step1_generate():
    """Generate 250 structures from wide-gap + low-k sources."""
    print("=== Step 1: Generate 250 ===", flush=True)
    from dielectric_data.reader import get_dataset_info, parse_target, robust_fuzzy_parse
    from ed_data import load_sp
    from ed_model import EdGPT
    from ed_train import generate

    tag_groups = [{"wide-gap", "very-wide-gap"}, {"low-k", "very-low-k"}]
    sources = []
    with open(DATA) as f:
        for row in csv.DictReader(f):
            if row.get("label", "") not in ("eval", "test_1k", "test_100"):
                continue
            tags = set(row["source"].split("|")[-1].strip().split())
            if all(tags & g for g in tag_groups):
                sources.append((row["source"], row.get("origin", "mp")))

    import random
    rng = random.Random(SEED)
    selected = rng.choices(sources, k=N)

    ds_info = get_dataset_info(DATA)
    version_id = ds_info.get("version_id", "d6")

    ckpt = torch.load(CKPT_DIR / "ed_ckpt_best_probe.pt", map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    if not any("q_norm" in k for k in state_dict) and getattr(config, "qk_norm", False):
        config.qk_norm = False
    model = EdGPT(config).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    sp = load_sp(str(CKPT_DIR / "model_sp.model"))

    generated = []
    for i, (src, origin) in enumerate(selected):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] generating...", flush=True)
        tgt = generate(model, sp, src, "cuda", max_length=512, top_k=10)
        atoms = parse_target(tgt, version_id, origin)
        if atoms is None:
            atoms = robust_fuzzy_parse(tgt)
        if atoms is not None and len(atoms) > 0:
            atoms.info["source"] = src
            atoms.info["target"] = tgt
            generated.append(atoms)

    print(f"Generated: {len(generated)}/{N}", flush=True)
    return generated


def step2_novelty_filter(generated):
    """Filter for novel structures (not in training set)."""
    print("\n=== Step 2: Novelty filter ===", flush=True)
    train_formulas = set()
    with open(DATA) as f:
        for row in csv.DictReader(f):
            if row.get("label", "") in ("eval", "test_1k", "test_100"):
                continue
            tgt = row.get("target", "")
            for seg in tgt.split("|")[:2]:
                seg = seg.strip()
                if seg and not seg.startswith(("SG ", "A")):
                    train_formulas.add(seg)

    novel = []
    for atoms in generated:
        formula = atoms.get_chemical_formula()
        if formula not in train_formulas:
            novel.append(atoms)

    print(f"Novel: {len(novel)}/{len(generated)} ({100*len(novel)/max(len(generated),1):.0f}%)", flush=True)
    return novel


def step3_relax(novel):
    """Relax with MACE."""
    print(f"\n=== Step 3: Relax {len(novel)} structures ===", flush=True)
    from chem.props.relax import get_calculator
    calc = get_calculator("mace", None, "cuda")
    relaxed = []
    n_conv = 0
    for i, atoms in enumerate(novel):
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(novel)}] relaxing...", flush=True)
        r = copy.deepcopy(atoms)
        r.calc = calc
        try:
            opt = LBFGS(r, logfile=None)
            converged = opt.run(fmax=0.05, steps=200)
            r.info.update(atoms.info)
            r.info["converged"] = converged
            if converged:
                n_conv += 1
        except Exception:
            r.info["converged"] = False
        r.calc = None
        relaxed.append(r)
    print(f"Converged: {n_conv}/{len(novel)}", flush=True)
    return relaxed


def step4_bvs_filter(relaxed):
    """Filter by BVS."""
    print(f"\n=== Step 4: BVS filter ===", flush=True)
    from validity.structural import check_bvs
    passed = []
    for atoms in relaxed:
        bvs = check_bvs(atoms)
        atoms.info["bvs_pass"] = bvs["bvs_pass"]
        atoms.info["bvs_max_deviation"] = bvs.get("bvs_max_deviation")
        if bvs["bvs_pass"]:
            passed.append(atoms)
    print(f"BVS pass: {len(passed)}/{len(relaxed)}", flush=True)
    return passed


def step5_ehull_filter(bvs_passed):
    """Filter by ehull on a single GPU (sequential)."""
    print(f"\n=== Step 5: Ehull ({len(bvs_passed)} structures, single GPU) ===", flush=True)
    from validity.structural import load_pet, check_ehull

    t0 = time.time()
    pet = load_pet(device="cuda")
    cache = {}
    passed = []
    all_results = []
    for i, atoms in enumerate(bvs_passed):
        try:
            r = check_ehull(atoms, pet, cache=cache)
            atoms.info["e_above_hull"] = r.get("e_above_hull")
            atoms.info["ehull_pass"] = r["ehull_pass"]
            all_results.append(atoms)
            if r["ehull_pass"]:
                passed.append(atoms)
        except Exception:
            atoms.info["ehull_pass"] = False
            all_results.append(atoms)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(bvs_passed)}] ehull checking...", flush=True)

    print(f"Ehull pass: {len(passed)}/{len(bvs_passed)} ({time.time()-t0:.0f}s)", flush=True)
    return passed, all_results


def step6_properties(passing):
    """Compute band gap and eps_0 (both Egip and surrogate)."""
    print(f"\n=== Step 6: Properties ({len(passing)} structures) ===", flush=True)
    import xgboost as xgb

    # Surrogates
    MODEL_DIR = Path("chem/data")
    eps_model = xgb.Booster()
    eps_model.load_model(str(MODEL_DIR / "xgb_composition_dft_eps_0.json"))
    bg_model = xgb.Booster()
    bg_model.load_model(str(MODEL_DIR / "xgb_composition_dft_band_gap.json"))
    from chem.surrogates.eps0 import compute_composition_features

    for atoms in passing:
        try:
            features = compute_composition_features(atoms)
            dm = xgb.DMatrix(features.reshape(1, -1))
            atoms.info["surrogate_band_gap"] = round(max(0.0, float(bg_model.predict(dm)[0])), 3)
            atoms.info["surrogate_eps_0"] = round(max(1.0, float(eps_model.predict(dm)[0])), 2)
        except Exception:
            pass
    print("  Surrogates done", flush=True)

    # Egip band gap
    try:
        from chem.props.dielectric import Dielectrics
        with Dielectrics(shutdown_ray=False) as di:
            results = di.compute(passing, relax=False, compute_phonons=False)
        for atoms, res in zip(passing, results):
            val = res.get("mlip_bandgap")
            if val is not None:
                atoms.info["egip_bandgap"] = round(float(val), 3)
        print("  Egip band gap done", flush=True)
    except Exception as e:
        print(f"  Egip band gap failed: {e}", flush=True)

    # Egip eps_0
    try:
        from chem.props.dielectric import compute_eps_0
        for atoms in passing:
            try:
                eps = compute_eps_0(atoms)
                if eps is not None and eps > 0:
                    atoms.info["egip_eps_0"] = round(float(eps), 2)
            except Exception:
                pass
        print("  Egip eps_0 done", flush=True)
    except Exception as e:
        print(f"  Egip eps_0 failed: {e}", flush=True)


def step7_plot(passing):
    """Plot band gap and eps_0 distributions."""
    print(f"\n=== Step 7: Plots ===", flush=True)
    egip_pairs = [(a.info.get("egip_bandgap"), a.info.get("egip_eps_0"))
                  for a in passing if a.info.get("egip_bandgap") and a.info.get("egip_eps_0")]
    surr_pairs = [(a.info.get("surrogate_band_gap"), a.info.get("surrogate_eps_0"))
                  for a in passing if a.info.get("surrogate_band_gap") and a.info.get("surrogate_eps_0")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    bins_bg = np.linspace(0, 10, 21)

    for row, pairs, label, color in [(0, surr_pairs, "Surrogate", "steelblue"),
                                      (1, egip_pairs, "Egip/MLIP", "coral")]:
        if not pairs:
            continue
        bgs, epss = zip(*pairs)
        bgs, epss = list(bgs), list(epss)

        ax = axes[row, 0]
        ax.scatter(bgs, epss, alpha=0.6, s=50, c=color, edgecolors="k", linewidths=0.3)
        ax.axvline(3.0, color="red", ls="--", alpha=0.5)
        ax.axhline(3.0, color="orange", ls="--", alpha=0.5)
        ax.set_xlabel("Band Gap (eV)")
        ax.set_ylabel("eps_0")
        ax.set_title(f"{label}: BG vs eps_0 (n={len(pairs)})")

        ax = axes[row, 1]
        ax.hist(bgs, bins=bins_bg, alpha=0.7, color=color)
        ax.axvline(3.0, color="red", ls="--", alpha=0.7)
        ax.set_xlabel("Band Gap (eV)")
        ax.set_title(f"{label} Band Gap")

        ax = axes[row, 2]
        eps_clean = [e for e in epss if e > 0]
        if eps_clean:
            ax.hist(eps_clean, bins=np.linspace(0, min(20, max(eps_clean)+1), 16), alpha=0.7, color=color)
        ax.axvline(3.0, color="orange", ls="--", alpha=0.7)
        ax.set_xlabel("eps_0")
        ax.set_title(f"{label} eps_0")

    plt.suptitle(f"V4 Conditional: Wide-Gap + Low-K (n={len(passing)} novel, relaxed, BVS+ehull pass)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTDIR / "property_plots.png", dpi=150)
    print(f"  Saved {OUTDIR / 'property_plots.png'}", flush=True)


def step8_top10(passing):
    """Rank and print top 10."""
    print(f"\n=== Step 8: Top 10 ===", flush=True)

    ranked = []
    seen = set()
    for atoms in passing:
        formula = atoms.get_chemical_formula(mode="hill")
        if formula in seen:
            continue
        seen.add(formula)

        egip_bg = atoms.info.get("egip_bandgap", float("nan"))
        egip_eps = atoms.info.get("egip_eps_0", float("nan"))
        surr_bg = atoms.info.get("surrogate_band_gap", float("nan"))
        surr_eps = atoms.info.get("surrogate_eps_0", float("nan"))
        ehull = atoms.info.get("e_above_hull", float("nan"))

        bg = egip_bg if not math.isnan(egip_bg) else surr_bg
        eps = egip_eps if not math.isnan(egip_eps) else surr_eps
        if math.isnan(bg) or math.isnan(eps):
            score = 999
        else:
            score = math.sqrt(((bg - 4.0)/4.0)**2 + ((eps - 2.0)/3.0)**2)
        if not math.isnan(ehull):
            score += ehull

        ranked.append({"formula": formula, "egip_bg": egip_bg, "egip_eps": egip_eps,
                        "surr_bg": surr_bg, "surr_eps": surr_eps, "ehull": ehull, "score": score})

    ranked.sort(key=lambda x: x["score"])

    print(f"\n{'#':>3s} {'Formula':22s} {'Egip BG':>8s} {'Egip eps':>8s} {'Surr BG':>8s} {'Surr eps':>8s} {'Ehull':>7s} {'Score':>7s}", flush=True)
    print("-" * 90, flush=True)
    for i, r in enumerate(ranked[:10]):
        def fmt(v): return f"{v:.2f}" if not math.isnan(v) else "N/A"
        def fmt1(v): return f"{v:.1f}" if not math.isnan(v) else "N/A"
        def fmt3(v): return f"{v:.3f}" if not math.isnan(v) else "N/A"
        print(f"{i+1:3d} {r['formula']:22s} {fmt(r['egip_bg']):>8s} {fmt1(r['egip_eps']):>8s} "
              f"{fmt(r['surr_bg']):>8s} {fmt1(r['surr_eps']):>8s} {fmt3(r['ehull']):>7s} {r['score']:>7.3f}", flush=True)

    # Save CSV
    with open(OUTDIR / "ranked_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","formula","egip_bg","egip_eps","surr_bg","surr_eps","ehull","score"])
        w.writeheader()
        for i, r in enumerate(ranked):
            w.writerow({"rank": i+1, **{k: v for k, v in r.items() if k != "score"}, "score": round(r["score"], 4)})
    print(f"\nSaved {OUTDIR / 'ranked_candidates.csv'}", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    generated = step1_generate()
    novel = step2_novelty_filter(generated)
    relaxed = step3_relax(novel)
    bvs_passed = step4_bvs_filter(relaxed)
    ehull_passed, all_ehull = step5_ehull_filter(bvs_passed)

    if not ehull_passed:
        print("\nNo structures passed all filters!")
        return

    # Save passing structures
    ase_write(str(OUTDIR / "passing.extxyz"), ehull_passed, format="extxyz")
    print(f"\nSaved {len(ehull_passed)} passing structures", flush=True)

    step6_properties(ehull_passed)
    step7_plot(ehull_passed)
    step8_top10(ehull_passed)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"PIPELINE SUMMARY", flush=True)
    print(f"  Generated: {len(generated)}/{N}", flush=True)
    print(f"  Novel: {len(novel)}", flush=True)
    print(f"  Relaxed: {len(relaxed)}", flush=True)
    print(f"  BVS pass: {len(bvs_passed)}", flush=True)
    print(f"  Ehull pass: {len(ehull_passed)}", flush=True)
    print(f"  Total time: {time.time()-t_total:.0f}s", flush=True)

    summary = {"n_generated": len(generated), "n_novel": len(novel), "n_relaxed": len(relaxed),
               "n_bvs_pass": len(bvs_passed), "n_ehull_pass": len(ehull_passed)}
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

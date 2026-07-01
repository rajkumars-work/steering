#!/usr/bin/env python3
"""1000-structure unconditional generation + analysis for all 5 model variants.

For each variant:
1. Generate 1000 structures from eval sources
2. Relax with MACE
3. Screen: tier-0 + ehull (no funnel)
4. Analyze: element frequency, crystal system, novelty tiers, pass rates

Usage: python -u eval/run_1000_unconditional.py [--variant V1] [--gpu 0]
       Or run all 5 sequentially: python -u eval/run_1000_unconditional.py --all
"""
import sys, os, time, csv, re, copy, json, math, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ed"))

import numpy as np
import torch
from ase.io import read as ase_read, write as ase_write
from ase.optimize import LBFGS

VARIANTS = {
    "V1": {
        "ckpt": "sweep_novelty/v1_anon_src",
        "data": "small_d4_props_anon_src.csv",
        "label": "V1 anon_src (comp+stoich)",
    },
    "V2": {
        "ckpt": "sweep_novelty/v2_elements_causal",
        "data": "small_d5_elements_causal_tgt.csv",
        "label": "V2 elements_causal",
    },
    "V3": {
        "ckpt": "sweep_novelty/v3_elements_only",
        "data": "small_d5_elements_only_causal_tgt.csv",
        "label": "V3 elements_only",
    },
    "V4": {
        "ckpt": "sweep_novelty/v4_chem_causal",
        "data": "small_d5_chem_causal_tgt.csv",
        "label": "V4 chem_causal (SG)",
    },
    "V5": {
        "ckpt": "sweep_novelty/v5_props_v2",
        "data": "small_d4_props_anon_causal_tgt.csv",
        "label": "V5 props_v2+arch",
    },
}

CKPT_ROOT = Path("/data/rkumar/code/py/ed/checkpoints")
DATA_ROOT = Path("/data/rkumar/code/py/dielectric/data")
OUTROOT = Path("eval/uncond_1000")
N = 1000
SEED = 42


def load_model(ckpt_dir, device):
    from ed_data import load_sp
    from ed_model import EdGPT
    from expert_xattn_model import ExpertXAttnEdGPT

    ckpt_path = ckpt_dir / "ed_ckpt_best_probe.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    if not any("q_norm" in k for k in state_dict) and getattr(config, "qk_norm", False):
        config.qk_norm = False
    is_xattn = any("attn_x.kv_projs" in k for k in state_dict)
    if is_xattn:
        model = ExpertXAttnEdGPT(config).to(device)
    else:
        model = EdGPT(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    sp = load_sp(str(ckpt_dir / "model_sp.model"))
    return model, sp


def run_variant(variant_id, gpu_id=0):
    v = VARIANTS[variant_id]
    ckpt_dir = CKPT_ROOT / v["ckpt"]
    data_path = DATA_ROOT / v["data"]
    outdir = OUTROOT / variant_id.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}", flush=True)
    print(f"{v['label']} — 1000 unconditional structures", flush=True)
    print(f"Checkpoint: {ckpt_dir.name}", flush=True)
    print(f"Data: {data_path.name}", flush=True)
    print(f"GPU: {gpu_id}", flush=True)
    print(f"{'='*70}", flush=True)

    t_total = time.time()

    # --- 1. Load sources ---
    from dielectric_data.reader import get_dataset_info, parse_target, robust_fuzzy_parse
    ds_info = get_dataset_info(str(data_path))
    version_id = ds_info.get("version_id", "d6")

    sources = []
    with open(data_path) as f:
        for row in csv.DictReader(f):
            if row.get("label", "") in ("eval", "test_1k", "test_100"):
                sources.append((row["source"], row.get("origin", "mp")))

    import random
    rng = random.Random(SEED)
    selected = rng.choices(sources, k=N)
    print(f"Selected {N} from {len(sources)} eval sources", flush=True)

    # --- 2. Generate ---
    print(f"\n=== Generate {N} ===", flush=True)
    model, sp = load_model(ckpt_dir, device)
    from ed_train import generate

    generated = []
    for i, (src, origin) in enumerate(selected):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}]", flush=True)
        try:
            tgt = generate(model, sp, src, device, max_length=512, top_k=10)
            atoms = parse_target(tgt, version_id, origin)
            if atoms is None:
                atoms = robust_fuzzy_parse(tgt)
            if atoms is not None and len(atoms) > 0:
                atoms.info["source"] = src
                atoms.info["target"] = tgt
                generated.append(atoms)
        except Exception:
            pass

    del model  # free GPU memory
    torch.cuda.empty_cache()
    print(f"Generated: {len(generated)}/{N}", flush=True)
    ase_write(str(outdir / "generated.extxyz"), generated, format="extxyz")

    # --- 3. Relax ---
    RELAX_TIMEOUT = 60  # seconds per structure
    print(f"\n=== Relax {len(generated)} (timeout={RELAX_TIMEOUT}s) ===", flush=True)
    from chem.props.relax import get_calculator
    calc = get_calculator("mace", None, device)
    relaxed = []
    n_conv = 0
    n_timeout = 0
    for i, atoms in enumerate(generated):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(generated)}] conv={n_conv} timeout={n_timeout}", flush=True)
        r = copy.deepcopy(atoms)
        r.calc = calc
        try:
            import signal
            def _timeout_handler(signum, frame):
                raise TimeoutError()
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(RELAX_TIMEOUT)
            try:
                opt = LBFGS(r, logfile=None)
                converged = opt.run(fmax=0.05, steps=200)
                r.info.update(atoms.info)
                r.info["converged"] = converged
                if converged:
                    n_conv += 1
            except TimeoutError:
                r.info.update(atoms.info)
                r.info["converged"] = False
                r.info["relax_timeout"] = True
                n_timeout += 1
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            r.info["converged"] = False
        r.calc = None
        relaxed.append(r)
    del calc
    torch.cuda.empty_cache()
    print(f"Converged: {n_conv}/{len(generated)} (timeout: {n_timeout})", flush=True)
    ase_write(str(outdir / "relaxed.extxyz"), relaxed, format="extxyz")

    # --- 4. Screen: tier-0 ---
    print(f"\n=== Tier-0 screen ===", flush=True)
    from validity.checkers import AtomicDistanceChecker, ChemistryChecker, ScreeningPipeline
    from validity.screening_version import CHECK_THRESHOLDS

    tier0_pipeline = ScreeningPipeline([
        AtomicDistanceChecker(CHECK_THRESHOLDS),
        ChemistryChecker(CHECK_THRESHOLDS),
    ])
    checks = ['element_filter_pass', 'min_dist_pass', 'isolated_atoms_pass',
              'smact_pass', 'bvs_pass', 'spacegroup_pass']
    results = []
    for atoms in relaxed:
        r = tier0_pipeline.screen(atoms, funnel=False)
        r["formula"] = atoms.get_chemical_formula()
        results.append(r)

    tier0_pass_idx = [i for i, r in enumerate(results) if all(r.get(k) is True for k in checks)]
    print(f"Tier-0: {len(tier0_pass_idx)}/{len(relaxed)}", flush=True)

    # --- 5. Ehull on tier-0 passers ---
    print(f"\n=== Ehull ({len(tier0_pass_idx)} structures) ===", flush=True)
    from validity.structural import load_pet, check_ehull
    pet = load_pet(device=device)
    cache = {}
    t_ehull = time.time()
    for j, idx in enumerate(tier0_pass_idx):
        if (j + 1) % 25 == 0:
            print(f"  [{j+1}/{len(tier0_pass_idx)}] ({time.time()-t_ehull:.0f}s)", flush=True)
        try:
            r = check_ehull(relaxed[idx], pet, cache=cache)
            results[idx].update(r)
        except Exception as e:
            results[idx]["ehull_pass"] = False
            results[idx]["ehull_error"] = str(e)
    del pet
    torch.cuda.empty_cache()
    print(f"Ehull done: {time.time()-t_ehull:.0f}s", flush=True)

    # Compute overall
    for r in results:
        t0_ok = all(r.get(k) is True for k in checks)
        ehull_ok = r.get("ehull_pass", None)
        r["overall_pass"] = t0_ok and (ehull_ok is True)

    # Save results
    fieldnames = sorted(set().union(*(r.keys() for r in results)))
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    passing = [a for r, a in zip(results, relaxed) if r.get("overall_pass") is True]
    if passing:
        ase_write(str(outdir / "passing.extxyz"), passing, format="extxyz")

    # --- 6. Analysis ---
    print(f"\n=== Analysis ===", flush=True)
    n = len(results)
    n_gen = len(generated)
    n_tier0 = len(tier0_pass_idx)
    n_ehull = sum(1 for r in results if r.get("ehull_pass") is True)
    n_overall = len(passing)

    # Novelty
    train_formulas = set()
    train_element_sets = set()
    with open(data_path) as f:
        for row in csv.DictReader(f):
            if row.get("label", "") in ("eval", "test_1k", "test_100"):
                continue
            tgt = row.get("target", "")
            for seg in tgt.split("|")[:2]:
                seg = seg.strip()
                if seg and not seg.startswith(("SG ", "A")):
                    train_formulas.add(seg)
                    elems = frozenset(re.findall(r'[A-Z][a-z]?', seg))
                    if len(elems) >= 2:
                        train_element_sets.add(elems)

    tiers = Counter()
    elem_freq = Counter()
    crystal_sys = Counter()
    n_elements_dist = Counter()
    formulas_seen = set()
    for r, atoms in zip(results, relaxed):
        if r.get("formula"):
            formula = r["formula"]
            formulas_seen.add(formula)
            elems = frozenset(re.findall(r'[A-Z][a-z]?', formula))
            for el in elems:
                elem_freq[el] += 1
            n_elements_dist[len(elems)] += 1

            if elems not in train_element_sets:
                tiers["novel"] += 1
            elif formula not in train_formulas:
                tiers["variant"] += 1
            else:
                tiers["copy"] += 1

        sg = r.get("spacegroup_computed")
        if sg:
            crystal_sys[str(sg)] += 1

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY: {v['label']}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Generated:    {n_gen}/{N}", flush=True)
    print(f"  Tier-0:       {n_tier0}/{n_gen} ({100*n_tier0/max(n_gen,1):.0f}%)", flush=True)
    print(f"  Ehull:        {n_ehull}/{n_gen} ({100*n_ehull/max(n_gen,1):.0f}%)", flush=True)
    print(f"  Overall:      {n_overall}/{n_gen} ({100*n_overall/max(n_gen,1):.0f}%)", flush=True)
    print(f"  Converged:    {n_conv}/{n_gen} ({100*n_conv/max(n_gen,1):.0f}%)", flush=True)
    print(f"  Unique formulas: {len(formulas_seen)}", flush=True)
    print(f"  Novelty: copy={tiers['copy']} variant={tiers['variant']} novel={tiers['novel']}", flush=True)

    for k in checks:
        p = sum(1 for r in results if r.get(k) is True)
        print(f"    {k:25s}: {p}/{n_gen} ({100*p/max(n_gen,1):.0f}%)", flush=True)

    print(f"\n  Top 15 elements:", flush=True)
    for el, cnt in elem_freq.most_common(15):
        print(f"    {el:3s}: {cnt}", flush=True)

    print(f"\n  Elements per structure:", flush=True)
    for ne in sorted(n_elements_dist.keys()):
        print(f"    {ne}-element: {n_elements_dist[ne]}", flush=True)

    print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)

    # Save summary
    summary = {
        "variant": variant_id,
        "label": v["label"],
        "n_requested": N,
        "n_generated": n_gen,
        "n_converged": n_conv,
        "n_tier0": n_tier0,
        "n_ehull": n_ehull,
        "n_overall": n_overall,
        "n_unique_formulas": len(formulas_seen),
        "novelty": dict(tiers),
        "per_check": {k: sum(1 for r in results if r.get(k) is True) for k in checks},
        "top_elements": dict(elem_freq.most_common(20)),
        "elements_per_structure": dict(n_elements_dist),
        "total_time_s": round(time.time() - t_total),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {outdir}/", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS.keys()), help="Run one variant")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--all", action="store_true", help="Run all 5 sequentially")
    args = p.parse_args()

    OUTROOT.mkdir(parents=True, exist_ok=True)

    if args.all:
        summaries = []
        for vid in VARIANTS:
            s = run_variant(vid, gpu_id=args.gpu)
            summaries.append(s)

        # Print comparison
        print(f"\n\n{'='*90}", flush=True)
        print("COMPARISON: ALL 5 VARIANTS", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'Variant':30s} {'Gen':>5s} {'Tier0':>6s} {'Ehull':>6s} {'Overall':>8s} {'Conv%':>6s} {'Formulas':>9s} {'Copy':>5s} {'Var':>5s} {'Novel':>6s}", flush=True)
        print("-" * 95, flush=True)
        for s in summaries:
            ng = s["n_generated"]
            nov = s["novelty"]
            print(f"{s['label']:30s} {ng:>5} {s['n_tier0']:>5} {s['n_ehull']:>5} {s['n_overall']:>7} "
                  f"{100*s['n_converged']/max(ng,1):>5.0f}% {s['n_unique_formulas']:>9} "
                  f"{nov.get('copy',0):>5} {nov.get('variant',0):>5} {nov.get('novel',0):>6}", flush=True)

        (OUTROOT / "comparison.json").write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved comparison to {OUTROOT / 'comparison.json'}", flush=True)

    elif args.variant:
        run_variant(args.variant, gpu_id=args.gpu)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

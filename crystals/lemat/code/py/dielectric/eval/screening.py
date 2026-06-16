"""Composable screening primitives — the single source of truth for all eval scripts.

Every ad-hoc eval script should import from here instead of reimplementing
generation, relaxation, or screening loops. Each primitive:
  - Has well-defined timeouts
  - Returns structured results
  - Handles errors gracefully (never crashes the batch)
  - Logs progress consistently

Usage:
    from eval.screening import (
        load_model, load_sources, generate_batch, relax_batch,
        screen_batch, classify_novelty_batch, summarize,
        DEFAULTS,
    )

    model, sp = load_model(ckpt_dir, device)
    sources = load_sources(data_csv, n=100, split="eval")
    generated = generate_batch(model, sp, sources, device)
    relaxed = relax_batch(generated, device)
    results = screen_batch(relaxed, device, full=True)
    summarize(results)
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ED_ROOT = _PROJECT_ROOT.parent / "ed"

for p in (str(_PROJECT_ROOT), str(_ED_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Defaults — single source of truth
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Generation
    "max_length": 512,
    "top_k": 10,
    "gen_timeout": 20,       # seconds per structure

    # Relaxation
    "fmax": 0.05,
    "relax_max_steps": 200,
    "relax_timeout": 60,     # seconds per structure
    "relax_model": None,     # MACE default (mpa-0)

    # Screening
    "full": True,            # include MLIP checks
    "funnel": False,         # run all checks even on failure
    "screen_timeout": 120,   # legacy, per-checker timeouts used instead
}


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

ED_CHECKPOINTS = Path("/data/rkumar/code/py/ed/checkpoints")


def load_model(ckpt_dir, device="cuda"):
    """Load ED model and tokenizer from checkpoint directory.

    Accepts:
      - Absolute path: "/data/rkumar/code/py/ed/checkpoints/sweep_novelty/v4_chem_causal"
      - Relative to ED checkpoints: "sweep_novelty/v4_chem_causal"
      - Relative to cwd (fallback)

    Handles: EdGPT, ExpertXAttnEdGPT, qk_norm mismatch, torch.compile prefixes.
    Returns (model, sp) ready for generation.
    """
    import torch
    from ed_data import load_sp
    from ed_model import EdGPT
    from expert_xattn_model import ExpertXAttnEdGPT
    from d_model import DecoderOnlyGPT

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        # Try relative to ED checkpoints root
        alt = ED_CHECKPOINTS / ckpt_dir
        if alt.exists():
            ckpt_dir = alt
        else:
            raise FileNotFoundError(
                f"Checkpoint dir not found: {ckpt_dir}\n"
                f"Also tried: {alt}"
            )

    # Find best checkpoint
    for name in ("ed_ckpt_best_probe.pt", "ed_ckpt_ema_final.pt", "ed_ckpt_final.pt"):
        p = ckpt_dir / name
        if p.exists():
            ckpt_path = p
            break
    else:
        # Fall back to latest .pt
        pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not pts:
            raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
        ckpt_path = pts[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    # Fix qk_norm mismatch (v1 checkpoints)
    if not any("q_norm" in k for k in state_dict) and getattr(config, "qk_norm", False):
        config.qk_norm = False

    # Detect architecture from state_dict structure:
    # - ExpertXAttnEdGPT: has `attn_x.kv_projs` (multiple expert kv projections)
    # - DecoderOnlyGPT: no encoder layers (`transformer.e.`) and no cross-attn
    # - EdGPT: standard encoder-decoder (default)
    is_xattn = any("attn_x.kv_projs" in k for k in state_dict)
    is_decoder_only = (
        not is_xattn
        and not any(k.startswith("transformer.e.") for k in state_dict)
        and not any("attn_x" in k for k in state_dict)
    )
    if is_xattn:
        model = ExpertXAttnEdGPT(config).to(device)
    elif is_decoder_only:
        model = DecoderOnlyGPT(config).to(device)
    else:
        model = EdGPT(config).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    sp = load_sp(str(ckpt_dir / "model_sp.model"))
    print(f"Loaded {ckpt_path.name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, sp


# ---------------------------------------------------------------------------
# 2. Source loading
# ---------------------------------------------------------------------------

def load_sources(csv_path, n=None, seed=42, split="test"):
    """Load (source, origin) pairs from a dataset CSV.

    Args:
        split: "test" (test_1k + test_100), "eval" (eval + test), or "all"
    Returns: list of (source_str, origin_str) tuples
    """
    sources = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_col = "label" if "label" in reader.fieldnames else None
        for row in reader:
            if label_col and split != "all":
                lbl = row.get(label_col, "")
                if split == "test" and lbl not in ("test_1k", "test_100"):
                    continue
                if split == "eval" and lbl not in ("eval", "test_1k", "test_100"):
                    continue
            sources.append((row["source"], row.get("origin", "mp")))

    if n is not None and n < len(sources):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(sources), size=n, replace=False)
        idx.sort()
        sources = [sources[i] for i in idx]

    return sources


# ---------------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    atoms: object           # ASE Atoms or None
    target: str             # raw target string
    source: str
    origin: str
    status: str             # SUCCESS, FAIL_LATTICE, FAIL_ATOMS, TIMEOUT, ERROR
    num_segs: int
    parse_reason: str = ""


class _GenTimeout(Exception):
    pass


def generate_one(model, sp, source, origin, version_id, device,
                 max_length=None, top_k=None, timeout=None):
    """Generate one structure. Returns GenResult. Never raises."""
    max_length = max_length or DEFAULTS["max_length"]
    top_k = top_k if top_k is not None else DEFAULTS["top_k"]
    timeout = timeout or DEFAULTS["gen_timeout"]

    from dielectric_data.reader import parse_target
    from ed_translate import translate_batch

    def _handler(signum, frame):
        raise _GenTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        translations = translate_batch(model, sp, [source], device, max_length, top_k)
        tgt = translations[0]
        num_segs = len(tgt.split("|"))
        atoms = parse_target(tgt, version_id, origin)
        if atoms is not None:
            atoms.info["source"] = source
            atoms.info["target"] = tgt
            return GenResult(atoms, tgt, source, origin, "SUCCESS", num_segs)

        parts = [p.strip() for p in tgt.split("|")]
        has_lattice = any(len(p.split()) == 6 for p in parts)
        status = "FAIL_LATTICE" if not has_lattice else "FAIL_ATOMS"
        reason = "no 6-float lattice" if not has_lattice else "no valid atom tokens"
        return GenResult(None, tgt, source, origin, status, num_segs, reason)

    except _GenTimeout:
        return GenResult(None, "", source, origin, "TIMEOUT", 0, f">{timeout}s")
    except Exception as e:
        return GenResult(None, "", source, origin, "ERROR", 0, str(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_batch(model, sp, sources, device, version_id=None,
                   max_length=None, top_k=None, timeout=None,
                   log_interval=100):
    """Generate structures for a batch of (source, origin) pairs.

    Returns list of GenResult. Auto-detects version_id from data if not given.
    """
    results = []
    n = len(sources)
    t0 = time.time()
    n_ok = 0

    for i, (src, origin) in enumerate(sources):
        r = generate_one(model, sp, src, origin, version_id or "d6", device,
                         max_length, top_k, timeout)
        results.append(r)
        if r.status == "SUCCESS":
            n_ok += 1

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  gen [{i+1}/{n}] ok={n_ok} ({elapsed:.0f}s)", flush=True)

    print(f"Generated: {n_ok}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# 4. Relaxation
# ---------------------------------------------------------------------------

@dataclass
class RelaxResult:
    atoms: object           # ASE Atoms (relaxed or original on failure)
    converged: bool
    steps: int
    energy: float | None
    time_s: float
    timed_out: bool = False


def relax_one(atoms, calc, fmax=None, max_steps=None, timeout=None):
    """Relax one structure. Returns RelaxResult. Never raises."""
    fmax = fmax or DEFAULTS["fmax"]
    max_steps = max_steps or DEFAULTS["relax_max_steps"]
    timeout = timeout or DEFAULTS["relax_timeout"]

    from chem.props.relax import relax_structure

    t0 = time.time()
    try:
        r_atoms, converged, steps, energy = relax_structure(
            atoms, calc, fmax=fmax, max_steps=max_steps, timeout=timeout)

        # Clean calculator arrays
        r_atoms.calc = None
        for key in list(r_atoms.arrays.keys()):
            if key not in ("numbers", "positions"):
                del r_atoms.arrays[key]

        # Preserve info from original
        for k, v in atoms.info.items():
            if k not in r_atoms.info:
                r_atoms.info[k] = v

        return RelaxResult(r_atoms, converged, steps, energy, time.time() - t0)
    except Exception:
        return RelaxResult(atoms, False, 0, None, time.time() - t0)


def relax_batch(gen_results, device, fmax=None, max_steps=None, timeout=None,
                relax_model=None, log_interval=100):
    """Relax all successfully generated structures.

    Returns list of RelaxResult (same length as gen_results; None for failed gens).
    """
    import torch
    from chem.props.relax import get_calculator

    calc = get_calculator("mace", relax_model or DEFAULTS["relax_model"], device)

    results = []
    n_total = sum(1 for g in gen_results if g.status == "SUCCESS")
    n_conv = 0
    n_timeout = 0
    idx = 0
    t0 = time.time()

    for g in gen_results:
        if g.status != "SUCCESS":
            results.append(None)
            continue

        r = relax_one(g.atoms, calc, fmax, max_steps, timeout)
        results.append(r)
        if r.converged:
            n_conv += 1
        if r.timed_out:
            n_timeout += 1
        idx += 1

        if idx % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  relax [{idx}/{n_total}] conv={n_conv} timeout={n_timeout} ({elapsed:.0f}s)",
                  flush=True)

    del calc
    torch.cuda.empty_cache()
    print(f"Relaxed: {n_conv}/{n_total} converged, {n_timeout} timed out ({time.time()-t0:.0f}s)",
          flush=True)
    return results


# ---------------------------------------------------------------------------
# 5. Screening
# ---------------------------------------------------------------------------

@dataclass
class ScreenResult:
    checks: dict            # all check results (pass/fail + values)
    overall_pass: bool
    time_s: float


def screen_one(atoms, device="cuda", full=None, funnel=None,
               pet=None, mace=None, ehull_cache=None, synth_scorer=None,
               run_cross=False):
    """Screen one structure. Returns ScreenResult. Never raises."""
    full = full if full is not None else DEFAULTS["full"]
    funnel = funnel if funnel is not None else DEFAULTS["funnel"]

    from validity.checkers import (
        AtomicDistanceChecker, ChemistryChecker, StabilityChecker,
        SynthesizabilityChecker, ScreeningPipeline,
    )
    from validity.screening_version import CHECK_THRESHOLDS

    t0 = time.time()
    try:
        checkers = [
            AtomicDistanceChecker(CHECK_THRESHOLDS),
            ChemistryChecker(CHECK_THRESHOLDS),
        ]
        if full and pet is not None:
            checkers.append(StabilityChecker(
                CHECK_THRESHOLDS, pet=pet, mace=mace, ehull_cache=ehull_cache,
                run_cross=run_cross))
        if full and synth_scorer is not None:
            checkers.append(SynthesizabilityChecker(
                CHECK_THRESHOLDS, scorer=synth_scorer))

        pipeline = ScreeningPipeline(checkers)
        checks = pipeline.screen(atoms, funnel=funnel)
        overall = checks.get("overall_pass", False)
        return ScreenResult(checks, overall, time.time() - t0)
    except Exception as e:
        return ScreenResult({"error": str(e), "overall_pass": False}, False, time.time() - t0)


def screen_batch(gen_results, relax_results, device="cuda", full=None, funnel=None,
                 log_interval=50, run_cross=False):
    """Screen all relaxed structures.

    Loads MLIP models once, screens all structures, cleans up.
    Returns list of ScreenResult (same length as gen_results; None for failed gens).
    """
    import torch
    full = full if full is not None else DEFAULTS["full"]

    pet, mace, ehull_cache, synth_scorer = None, None, None, None
    if full:
        from validity.structural import load_stability_calc, load_mace
        pet = load_stability_calc(device=device)
        mace = load_mace(device=device)
        ehull_cache = {}
        try:
            from scripts.synthesizability_score import SynthesizabilityScorer
            synth_scorer = SynthesizabilityScorer()
        except Exception:
            pass

    results = []
    n_total = sum(1 for r in relax_results if r is not None)
    n_pass = 0
    idx = 0
    t0 = time.time()

    for g, r in zip(gen_results, relax_results):
        if r is None:
            results.append(None)
            continue

        sr = screen_one(r.atoms, device, full, funnel, pet, mace, ehull_cache, synth_scorer,
                        run_cross=run_cross)
        results.append(sr)
        if sr.overall_pass:
            n_pass += 1
        idx += 1

        if idx % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  screen [{idx}/{n_total}] pass={n_pass} ({elapsed:.0f}s)", flush=True)

    if pet is not None:
        del pet, mace
        torch.cuda.empty_cache()

    print(f"Screened: {n_pass}/{n_total} pass ({time.time()-t0:.0f}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# 6. Novelty classification
# ---------------------------------------------------------------------------

def build_novelty_index(csv_path):
    """Build novelty reference index from training data. Uses disk cache."""
    from validity.checkers import NoveltyIndex
    return NoveltyIndex.load_or_build(csv_path)


def classify_novelty(atoms, novelty_index):
    """Classify one structure: copy / polymorph / variant / novel."""
    if novelty_index is None:
        return "unknown"
    return novelty_index.classify(atoms)


def classify_novelty_batch(gen_results, relax_results, novelty_index):
    """Classify novelty for all generated structures. Returns list of tier strings."""
    tiers = []
    for g, r in zip(gen_results, relax_results):
        if r is None or g.status != "SUCCESS":
            tiers.append(None)
            continue
        tiers.append(classify_novelty(r.atoms, novelty_index))
    return tiers


# ---------------------------------------------------------------------------
# 7. Results aggregation and output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Tier-0 only (no GPU needed)
# ---------------------------------------------------------------------------

def tier0_one(atoms):
    """Run tier-0 checks only (element filter, min dist, isolated atoms, SMACT, BVS, SG).
    No MLIP, no GPU. Returns ScreenResult."""
    from validity.checkers import (
        AtomicDistanceChecker, ChemistryChecker, ScreeningPipeline,
    )
    from validity.screening_version import CHECK_THRESHOLDS, TIER0_CHECKS

    t0 = time.time()
    try:
        pipeline = ScreeningPipeline([
            AtomicDistanceChecker(CHECK_THRESHOLDS),
            ChemistryChecker(CHECK_THRESHOLDS),
        ])
        checks = pipeline.screen(atoms, funnel=False)
        tier0_pass = all(checks.get(k) is True for k in TIER0_CHECKS)
        checks["tier0_pass"] = tier0_pass
        return ScreenResult(checks, tier0_pass, time.time() - t0)
    except Exception as e:
        return ScreenResult({"error": str(e), "tier0_pass": False}, False, time.time() - t0)


# ---------------------------------------------------------------------------
# 8. Standard pipeline — the canonical ordering
# ---------------------------------------------------------------------------
#
#   1. Generate
#   2. Novelty filter (optional — skip copies of training data)
#   3. Tier-0 screen (fast, no GPU — reject garbage before expensive steps)
#   4. Relax (MACE, expensive)
#   5. Ehull + MLIP checks (most expensive per-structure)
#
# This ordering minimizes wasted compute by filtering early.
# Every eval script should use run_pipeline() instead of ad-hoc loops.

@dataclass
class PipelineResult:
    """Full result for one structure through the pipeline."""
    gen: GenResult
    novelty_tier: str | None = None
    tier0: ScreenResult | None = None
    relax: RelaxResult | None = None
    screen: ScreenResult | None = None
    dropped_at: str | None = None      # "gen" | "novelty" | "tier0" | None

    @property
    def overall_pass(self) -> bool:
        if self.screen is not None:
            return self.screen.overall_pass
        return False


def run_pipeline(
    model, sp, sources, device, data_csv,
    version_id="d6",
    novelty_filter=True,
    novelty_keep=("variant", "novel", "polymorph"),
    full=True,
    log_interval=100,
    label="",
):
    """Run the standard screening pipeline on a batch of sources.

    Steps:
        1. Generate all structures
        2. (Optional) Novelty filter — drop copies
        3. Tier-0 screen on unrelaxed — drop garbage
        4. Relax survivors
        5. Full screen (tier-0 + ehull + MLIP) on relaxed

    Returns list of PipelineResult.
    """
    import torch
    n = len(sources)
    t0 = time.time()
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}", flush=True)

    # --- Step 1: Generate ---
    print(f"\n=== Step 1: Generate {n} ===", flush=True)
    gen_results = generate_batch(model, sp, sources, device, version_id,
                                 log_interval=log_interval)
    n_gen = sum(1 for g in gen_results if g.status == "SUCCESS")

    # Free model GPU memory — we won't need it again
    del model
    torch.cuda.empty_cache()

    # --- Step 2: Novelty filter ---
    novelty_index = None
    novelty_tiers = [None] * n
    if novelty_filter:
        print(f"\n=== Step 2: Novelty filter ===", flush=True)
        novelty_index = build_novelty_index(data_csv)
        for i, g in enumerate(gen_results):
            if g.atoms is not None:
                novelty_tiers[i] = classify_novelty(g.atoms, novelty_index)
        tier_counts = Counter(t for t in novelty_tiers if t is not None)
        print(f"  Tiers: {dict(tier_counts)}", flush=True)
    else:
        print(f"\n=== Step 2: Novelty filter (skipped) ===", flush=True)

    # --- Step 3: Tier-0 on unrelaxed ---
    print(f"\n=== Step 3: Tier-0 screen (unrelaxed) ===", flush=True)
    tier0_results = [None] * n
    n_tier0 = 0
    for i, g in enumerate(gen_results):
        if g.atoms is None:
            continue
        # Drop if novelty filter is on and it's a copy
        if novelty_filter and novelty_tiers[i] not in novelty_keep:
            continue
        t0r = tier0_one(g.atoms)
        tier0_results[i] = t0r
        if t0r.overall_pass:
            n_tier0 += 1
    print(f"  Tier-0 pass: {n_tier0}/{n_gen}", flush=True)

    # --- Step 4: Relax tier-0 passers ---
    relax_indices = [i for i in range(n)
                     if tier0_results[i] is not None and tier0_results[i].overall_pass]
    print(f"\n=== Step 4: Relax {len(relax_indices)} structures ===", flush=True)

    from chem.props.relax import get_calculator
    calc = get_calculator("mace", DEFAULTS["relax_model"], device)
    relax_results = [None] * n
    n_conv = 0
    for j, i in enumerate(relax_indices):
        r = relax_one(gen_results[i].atoms, calc)
        relax_results[i] = r
        if r.converged:
            n_conv += 1
        if (j + 1) % log_interval == 0:
            print(f"  relax [{j+1}/{len(relax_indices)}] conv={n_conv}", flush=True)
    del calc
    torch.cuda.empty_cache()
    print(f"  Converged: {n_conv}/{len(relax_indices)}", flush=True)

    # --- Step 5: Full screen (ehull + MLIP) on relaxed ---
    screen_indices = [i for i in relax_indices if relax_results[i] is not None]
    print(f"\n=== Step 5: Full screen {len(screen_indices)} structures ===", flush=True)

    pet, mace, ehull_cache, synth_scorer = None, None, None, None
    if full:
        from validity.structural import load_stability_calc, load_mace
        pet = load_stability_calc(device=device)
        mace = load_mace(device=device)
        ehull_cache = {}
        try:
            from scripts.synthesizability_score import SynthesizabilityScorer
            synth_scorer = SynthesizabilityScorer()
        except Exception:
            pass

    screen_results = [None] * n
    n_pass = 0
    for j, i in enumerate(screen_indices):
        sr = screen_one(relax_results[i].atoms, device, full=full, funnel=False,
                        pet=pet, mace=mace, ehull_cache=ehull_cache,
                        synth_scorer=synth_scorer)
        screen_results[i] = sr
        if sr.overall_pass:
            n_pass += 1
        if (j + 1) % 25 == 0:
            print(f"  screen [{j+1}/{len(screen_indices)}] pass={n_pass}", flush=True)

    if pet is not None:
        del pet, mace
        torch.cuda.empty_cache()
    print(f"  Overall pass: {n_pass}/{len(screen_indices)}", flush=True)

    # --- Assemble PipelineResults ---
    pipeline_results = []
    for i in range(n):
        g = gen_results[i]
        pr = PipelineResult(gen=g, novelty_tier=novelty_tiers[i])

        if g.atoms is None:
            pr.dropped_at = "gen"
        elif novelty_filter and novelty_tiers[i] not in novelty_keep:
            pr.dropped_at = "novelty"
        elif tier0_results[i] is not None and not tier0_results[i].overall_pass:
            pr.tier0 = tier0_results[i]
            pr.dropped_at = "tier0"
        else:
            pr.tier0 = tier0_results[i]
            pr.relax = relax_results[i]
            pr.screen = screen_results[i]

        pipeline_results.append(pr)

    elapsed = time.time() - t0
    print(f"\nPipeline complete: {n_pass}/{n_gen} pass ({elapsed:.0f}s total)", flush=True)

    return pipeline_results


def pipeline_summary(results, label=""):
    """Print summary from PipelineResult list. Returns summary dict."""
    n = len(results)
    n_gen = sum(1 for r in results if r.gen.status == "SUCCESS")
    n_novel = sum(1 for r in results if r.dropped_at != "novelty" and r.gen.status == "SUCCESS")
    n_tier0 = sum(1 for r in results if r.tier0 is not None and r.tier0.overall_pass)
    n_relax = sum(1 for r in results if r.relax is not None and r.relax.converged)
    n_pass = sum(1 for r in results if r.overall_pass)

    tier_counts = Counter(r.novelty_tier for r in results if r.novelty_tier)
    dropped = Counter(r.dropped_at for r in results if r.dropped_at)

    print(f"\n{'='*60}")
    if label:
        print(f"PIPELINE SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"  1. Generated:    {n_gen}/{n}")
    print(f"  2. After novelty: {n_novel} (dropped {dropped.get('novelty', 0)} copies)")
    print(f"  3. Tier-0 pass:  {n_tier0} (dropped {dropped.get('tier0', 0)})")
    print(f"  4. Relax conv:   {n_relax}/{n_tier0}")
    print(f"  5. Overall pass: {n_pass}")

    if tier_counts:
        print(f"\n  Novelty: {dict(tier_counts)}")

    return {
        "n": n, "generated": n_gen, "after_novelty": n_novel,
        "tier0_pass": n_tier0, "relaxed": n_relax, "overall_pass": n_pass,
        "novelty_tiers": dict(tier_counts), "dropped_at": dict(dropped),
    }


def save_pipeline_results(outdir, results, label=""):
    """Save PipelineResult list to outdir."""
    from ase.io import write as ase_write

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen_atoms = [r.gen.atoms for r in results if r.gen.atoms is not None]
    relax_atoms = [r.relax.atoms for r in results if r.relax is not None]
    passing_atoms = [r.relax.atoms for r in results if r.overall_pass]

    if gen_atoms:
        ase_write(str(outdir / "generated.extxyz"), gen_atoms, format="extxyz")
    if relax_atoms:
        ase_write(str(outdir / "relaxed.extxyz"), relax_atoms, format="extxyz")
    if passing_atoms:
        ase_write(str(outdir / "passing.extxyz"), passing_atoms, format="extxyz")

    # CSV
    rows = []
    for i, r in enumerate(results):
        row = {
            "index": i,
            "source": r.gen.source,
            "target": r.gen.target,
            "gen_status": r.gen.status,
            "novelty_tier": r.novelty_tier,
            "dropped_at": r.dropped_at,
        }
        if r.tier0 is not None:
            row.update({f"tier0_{k}": v for k, v in r.tier0.checks.items()
                        if isinstance(v, bool)})
        if r.relax is not None:
            row["formula"] = r.relax.atoms.get_chemical_formula()
            row["converged"] = r.relax.converged
            row["relax_time"] = round(r.relax.time_s, 2)
        if r.screen is not None:
            row.update(r.screen.checks)
            row["overall_pass"] = r.screen.overall_pass
        rows.append(row)

    if rows:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with open(outdir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    summary = pipeline_summary(results, label=label)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {outdir}/")
    return summary


def save_results(outdir, gen_results, relax_results, screen_results,
                 novelty_tiers=None):
    """Save structured results to outdir: results.csv, generated/relaxed/passing.extxyz."""
    from ase.io import write as ase_write

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect atoms lists
    generated_atoms = [g.atoms for g in gen_results if g.atoms is not None]
    relaxed_atoms = [r.atoms for r in relax_results if r is not None]
    passing_atoms = [r.atoms for r, s in zip(relax_results, screen_results)
                     if r is not None and s is not None and s.overall_pass]

    if generated_atoms:
        ase_write(str(outdir / "generated.extxyz"), generated_atoms, format="extxyz")
    if relaxed_atoms:
        ase_write(str(outdir / "relaxed.extxyz"), relaxed_atoms, format="extxyz")
    if passing_atoms:
        ase_write(str(outdir / "passing.extxyz"), passing_atoms, format="extxyz")

    # CSV
    rows = []
    for i, (g, r, s) in enumerate(zip(gen_results, relax_results, screen_results)):
        row = {
            "index": i,
            "source": g.source,
            "target": g.target,
            "gen_status": g.status,
            "num_segs": g.num_segs,
            "parse_reason": g.parse_reason,
        }
        if r is not None:
            row["formula"] = r.atoms.get_chemical_formula() if r.atoms is not None else ""
            row["converged"] = r.converged
            row["relax_steps"] = r.steps
            row["relax_energy"] = r.energy
            row["relax_time"] = round(r.time_s, 2)
        if s is not None:
            row.update(s.checks)
            row["overall_pass"] = s.overall_pass
        if novelty_tiers is not None and novelty_tiers[i] is not None:
            row["novelty_tier"] = novelty_tiers[i]
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(outdir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"Saved to {outdir}/ ({len(generated_atoms)} gen, {len(relaxed_atoms)} relax, "
          f"{len(passing_atoms)} pass)")
    return outdir


def summarize(gen_results, relax_results, screen_results, novelty_tiers=None, label=""):
    """Print a compact summary table. Returns summary dict."""
    n = len(gen_results)
    n_gen = sum(1 for g in gen_results if g.status == "SUCCESS")
    n_conv = sum(1 for r in relax_results if r is not None and r.converged)
    n_pass = sum(1 for s in screen_results if s is not None and s.overall_pass)

    # Per-check pass rates
    from validity.screening_version import TIER0_CHECKS, MLIP_CHECKS
    check_counts = Counter()
    for s in screen_results:
        if s is None:
            continue
        for k, v in s.checks.items():
            if v is True:
                check_counts[k] += 1

    # Novelty
    tier_counts = Counter(t for t in (novelty_tiers or []) if t is not None)

    print(f"\n{'='*60}")
    if label:
        print(f"SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"  Generated:  {n_gen}/{n} ({100*n_gen/max(n,1):.0f}%)")
    print(f"  Converged:  {n_conv}/{n_gen} ({100*n_conv/max(n_gen,1):.0f}%)")
    print(f"  Overall:    {n_pass}/{n_gen} ({100*n_pass/max(n_gen,1):.0f}%)")

    all_checks = list(TIER0_CHECKS) + list(MLIP_CHECKS)
    for k in all_checks:
        c = check_counts.get(k, 0)
        print(f"    {k:25s}: {c}/{n_gen} ({100*c/max(n_gen,1):.0f}%)")

    if tier_counts:
        print(f"\n  Novelty tiers:")
        for tier in ("copy", "polymorph", "variant", "novel"):
            c = tier_counts.get(tier, 0)
            print(f"    {tier:12s}: {c}")

    summary = {
        "n_requested": n,
        "n_generated": n_gen,
        "n_converged": n_conv,
        "n_overall_pass": n_pass,
        "per_check": dict(check_counts),
        "novelty_tiers": dict(tier_counts),
    }
    return summary

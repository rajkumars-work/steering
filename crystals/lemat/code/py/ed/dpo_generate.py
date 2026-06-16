"""Generate preference pairs for DPO training.

For each source prompt, generate a completion from the model, score it against
the ground-truth target, and keep pairs where the generation is clearly bad.
The ground truth becomes "chosen" and the model output becomes "rejected".

Processing is done in batches.  After each batch the scored pairs are appended
to the output CSV and a small .progress file is updated, so a crash only loses
the current batch.  Re-running the same command automatically resumes.

Supports multi-GPU parallelism via --shard / --num_shards.

Usage:
    # Single GPU
    python dpo_generate.py --data ../dielectric/data/small.csv \
        --checkpoint checkpoints/ed_ckpt_final.pt --output dpo_pairs.csv

    # Resume after crash (just re-run the same command)
    python dpo_generate.py --data ../dielectric/data/small.csv \
        --checkpoint checkpoints/ed_ckpt_final.pt --output dpo_pairs.csv

    # 4-GPU parallel
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python dpo_generate.py \
        --data small.csv --checkpoint ckpt.pt \
        --output dpo_shard_$i.csv --shard $i --num_shards 4 &
    done
    wait
    head -1 dpo_shard_0.csv > dpo_pairs.csv
    tail -n+2 -q dpo_shard_*.csv >> dpo_pairs.csv
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from ase.neighborlist import neighbor_list as ase_neighbor_list
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from configs import get_device
from ed_data import load_sp
from ed_model import EdGPT
from ed_translate import parse_target, translate_batch


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

def parse_source(source_str):
    """Extract expected natoms, element set, and density from a source string.

    Source format (pipe-separated):
        crystal_system space_group | anonymous_formula elements | natoms | density [| props]
    e.g.: "monoclinic P2_1/c | A2B2C2D Ag N O S | 28 | 4.6"

    Returns dict with keys: natoms, elements (set), density (float),
    anonymous_formula (str), or None values if parsing fails.
    """
    parts = [p.strip() for p in source_str.split("|")]
    info = {"natoms": None, "elements": None, "density": None, "anonymous_formula": None}

    if len(parts) < 4:
        return info

    formula_tokens = parts[1].split()
    if formula_tokens:
        info["anonymous_formula"] = formula_tokens[0]
        if len(formula_tokens) > 1:
            info["elements"] = set(formula_tokens[1:])

    try:
        info["natoms"] = int(parts[2].strip())
    except (ValueError, IndexError):
        pass

    try:
        info["density"] = float(parts[3].strip())
    except (ValueError, IndexError):
        pass

    return info


def anonymous_formula(atoms):
    """Compute reduced anonymous formula from an Atoms object.

    e.g. Ag8N8O8S4 -> counts [8,8,8,4] -> GCD=4 -> [2,2,2,1] -> A2B2C2D
    """
    from collections import Counter
    from math import gcd
    from functools import reduce
    counts = Counter(atoms.get_chemical_symbols())
    sorted_counts = sorted(counts.values(), reverse=True)
    g = reduce(gcd, sorted_counts) if sorted_counts else 1
    sorted_counts = [c // g for c in sorted_counts]
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for i, c in enumerate(sorted_counts):
        if i >= len(labels):
            break
        parts.append(f"{labels[i]}{c}" if c > 1 else labels[i])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pair(source_str, target_str, pred_str, pred_atoms, source_info,
               mace_calc=None, energy_threshold=-2.0):
    """Score a generation against the ground truth.

    Returns (is_bad: bool, reasons: list[str]).
    A generation is "bad" (= should become a rejected sample) if any check fails.

    If *mace_calc* is provided, structures with MACE energy per atom above
    *energy_threshold* eV are flagged as high-energy (cheap proxy for E above
    hull — most stable inorganic structures sit below -3 eV/atom).
    """
    reasons = []

    # 1. Parse failure
    if pred_atoms is None:
        return True, ["parse_failure"]

    # 2. Element mismatch
    if source_info["elements"] is not None:
        pred_elements = set(pred_atoms.get_chemical_symbols())
        if pred_elements != source_info["elements"]:
            reasons.append("element_mismatch")

    # 3. Atom count mismatch
    if source_info["natoms"] is not None:
        if len(pred_atoms) != source_info["natoms"]:
            reasons.append("atom_count_mismatch")

    # 4. Stoichiometry (anonymous formula) mismatch
    if source_info["anonymous_formula"] is not None:
        pred_anon = anonymous_formula(pred_atoms)
        if pred_anon != source_info["anonymous_formula"]:
            reasons.append("stoichiometry_mismatch")

    # 5. Density error > 20%
    if source_info["density"] is not None and source_info["density"] > 0:
        try:
            pred_vol = pred_atoms.get_volume()
            pred_mass = sum(pred_atoms.get_masses())
            pred_density = pred_mass / pred_vol * 1.6605  # g/cm^3
            rel_err = abs(pred_density - source_info["density"]) / source_info["density"]
            if rel_err > 0.20:
                reasons.append("density_error")
        except Exception:
            reasons.append("density_error")

    # 6. Min interatomic distance < 0.5 Å
    try:
        dists = ase_neighbor_list("d", pred_atoms, cutoff=0.5, self_interaction=False)
        if len(dists) > 0:
            reasons.append("min_dist")
    except Exception:
        pass

    # 7. Isolated atoms (no neighbour within 4 Å)
    try:
        i_list = ase_neighbor_list("i", pred_atoms, cutoff=4.0, self_interaction=False)
        connected = set(i_list)
        if len(connected) < len(pred_atoms):
            reasons.append("isolated_atoms")
    except Exception:
        pass

    # 8. Bond Valence Sum check
    try:
        structure = AseAtomsAdaptor.get_structure(pred_atoms)
        bva = BVAnalyzer()
        valences = bva.get_valences(structure)
        max_dev = max(abs(v - round(v)) for v in valences)
        if max_dev > 0.5:
            reasons.append("bvs_fail")
    except Exception:
        reasons.append("bvs_fail")

    # 9. MACE energy per atom (cheap E-above-hull proxy)
    if mace_calc is not None:
        try:
            import copy as _copy
            tmp = _copy.deepcopy(pred_atoms)
            tmp.calc = mace_calc
            e_per_atom = tmp.get_potential_energy() / len(tmp)
            if e_per_atom > energy_threshold:
                reasons.append("high_energy")
        except Exception:
            pass  # don't reject if calc fails

    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _progress_path(output_path):
    return output_path + ".progress"


def _read_progress(output_path):
    """Return number of input rows already processed, or 0."""
    pp = _progress_path(output_path)
    if not os.path.exists(pp):
        return 0
    try:
        with open(pp, "r") as f:
            return int(json.load(f)["rows_processed"])
    except Exception:
        return 0


def _write_progress(output_path, rows_processed, pairs_written, reason_counts):
    pp = _progress_path(output_path)
    with open(pp, "w") as f:
        json.dump({
            "rows_processed": rows_processed,
            "pairs_written": pairs_written,
            "reason_counts": reason_counts,
        }, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS = ["source", "chosen", "rejected", "score"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate DPO preference pairs")
    p.add_argument("--data", required=True, help="Input CSV with source/target columns")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--sp_model", default="checkpoints/model_sp.model",
                   help="SentencePiece model path")
    p.add_argument("--output", default="dpo_pairs.csv", help="Output CSV path")
    p.add_argument("--n", type=int, default=None, help="Max rows to process (default: all)")
    p.add_argument("--max_length", type=int, default=512, help="Generation max length")
    p.add_argument("--top_k", type=int, default=10, help="Sampling top-k")
    p.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--shard", type=int, default=None,
                   help="Shard index for multi-GPU parallelism (0-indexed)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards")
    p.add_argument("--no_resume", action="store_true",
                   help="Start fresh, ignoring any existing checkpoint")
    p.add_argument("--mace_model", default=None,
                   help="Path to MACE checkpoint for energy-based rejection "
                        "(e.g. /data/assets/checkpoints/mace/mace-mp-0b3-medium.model)")
    p.add_argument("--energy_threshold", type=float, default=-2.0,
                   help="MACE energy per atom threshold (eV); structures above this are rejected")
    return p.parse_args()


def load_model(ckpt_path, device):
    """Load model from checkpoint, stripping _orig_mod. prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = EdGPT(config).to(device)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(get_device())

    # ------------------------------------------------------------------
    # Load input data
    # ------------------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.n is not None:
        rows = rows[:args.n]

    if args.shard is not None and args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: processing {len(rows)} rows")

    sources = [r["source"] for r in rows]
    targets = [r["target"] for r in rows]
    total = len(sources)

    print(f"Loaded {total} rows from {args.data}")

    # ------------------------------------------------------------------
    # Resume handling
    # ------------------------------------------------------------------
    if args.no_resume:
        already_done = 0
        for f in [args.output, _progress_path(args.output)]:
            if os.path.exists(f):
                os.remove(f)
    else:
        already_done = _read_progress(args.output)

    if already_done >= total:
        print(f"Already complete ({already_done}/{total} rows processed). "
              f"Use --no_resume to regenerate.")
        return

    if already_done > 0:
        print(f"Resuming from row {already_done}/{total} "
              f"({already_done / total * 100:.1f}% done)")

    # ------------------------------------------------------------------
    # Open output CSV (create with header if new, else append)
    # ------------------------------------------------------------------
    if already_done == 0:
        out_f = open(args.output, "w", encoding="utf-8", newline="")
        out_writer = csv.DictWriter(out_f, fieldnames=_OUTPUT_FIELDS)
        out_writer.writeheader()
        out_f.flush()
    else:
        out_f = open(args.output, "a", encoding="utf-8", newline="")
        out_writer = csv.DictWriter(out_f, fieldnames=_OUTPUT_FIELDS)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, config = load_model(args.checkpoint, device)
    sp = load_sp(args.sp_model)

    # Optional MACE calculator for energy-based rejection
    mace_calc = None
    if args.mace_model is not None:
        from mace.calculators import MACECalculator
        mace_calc = MACECalculator(
            model_paths=[args.mace_model], device=str(device), default_dtype="float64"
        )
        print(f"MACE energy proxy enabled (threshold {args.energy_threshold} eV/atom)")

    # ------------------------------------------------------------------
    # Process in batches: generate → score → write → update progress
    # ------------------------------------------------------------------
    rows_processed = already_done
    pairs_written = 0
    reason_counts = {}
    t0 = time.time()

    remaining_sources = sources[already_done:]
    remaining_targets = targets[already_done:]

    for i in range(0, len(remaining_sources), args.batch_size):
        batch_src = remaining_sources[i:i + args.batch_size]
        batch_tgt = remaining_targets[i:i + args.batch_size]

        # Generate
        preds = translate_batch(model, sp, batch_src, device,
                                max_length=args.max_length, top_k=args.top_k)

        # Score and write each pair
        batch_pairs = 0
        for src, tgt, pred in zip(batch_src, batch_tgt, preds):
            pred_atoms = parse_target(pred)
            source_info = parse_source(src)
            is_bad, reasons = score_pair(src, tgt, pred, pred_atoms, source_info,
                                        mace_calc=mace_calc,
                                        energy_threshold=args.energy_threshold)

            if is_bad:
                score_str = "+".join(reasons) if reasons else "unknown"
                out_writer.writerow({
                    "source": src,
                    "chosen": tgt,
                    "rejected": pred,
                    "score": score_str,
                })
                batch_pairs += 1
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1

        # Flush output and update progress after each batch
        out_f.flush()
        rows_processed += len(batch_src)
        pairs_written += batch_pairs
        _write_progress(args.output, rows_processed, pairs_written, reason_counts)

        # Progress display
        elapsed = time.time() - t0
        done_since_resume = rows_processed - already_done
        rate = done_since_resume / elapsed if elapsed > 0 else 0
        remaining = total - rows_processed
        eta = remaining / rate if rate > 0 else 0
        batch_num = i // args.batch_size + 1
        if batch_num % 20 == 0 or rows_processed == total:
            print(f"  {rows_processed}/{total}  "
                  f"{pairs_written} pairs  "
                  f"({rate:.0f} rows/s, ETA {eta / 60:.0f}m)")

    out_f.close()

    # Free GPU
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Clean up progress file
    pp = _progress_path(args.output)
    if os.path.exists(pp):
        os.remove(pp)

    # Final summary
    rejection_rate = pairs_written / total * 100 if total > 0 else 0
    print(f"\nResults: {pairs_written}/{total} pairs "
          f"({rejection_rate:.1f}% rejection rate)")
    print(f"Written to {args.output}")
    print("\nFailure breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count / total * 100:.1f}%)")


if __name__ == "__main__":
    main()

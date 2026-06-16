"""Generate DPO preference pairs using best-of-N sampling with continuous scoring.

Instead of pairing ground truth vs single generation (v1), generates K samples
per prompt, scores each with a continuous quality metric, and pairs the best
against the worst.  Only creates pairs when the score gap exceeds a threshold.

Scoring components (all 0-1, higher = better):
  - smact:    1 if SMACT passes, 0 otherwise
  - bvs:     max(0, 1 - max_bvs_deviation)  (continuous)
  - density: max(0, 1 - |density_error|/0.3) (continuous)
  - min_dist: 1 if min distance >= 0.5 Å, 0 otherwise
  - parse:    1 if parseable, 0 otherwise

Usage:
    # Single GPU
    python dpo_generate_v2.py --data ../dielectric/data/small_scratchpad_ox.csv \
        --checkpoint checkpoints_scratchpad_ox/ed_ckpt_final.pt \
        --sp_model checkpoints_scratchpad_ox/model_sp.model \
        --output dpo_v2_pairs.csv --K 8

    # 2-GPU parallel
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python dpo_generate_v2.py \
        --data small.csv --checkpoint ckpt.pt \
        --output dpo_v2_shard_$i.csv --shard $i --num_shards 2 --K 8 &
    done
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
# Source parsing (same as v1)
# ---------------------------------------------------------------------------

def parse_source(source_str):
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


# ---------------------------------------------------------------------------
# Continuous scoring
# ---------------------------------------------------------------------------

def score_structure(pred_str, pred_atoms, source_info):
    """Score a generated structure on a continuous 0-1 scale.

    Returns (total_score: float, component_scores: dict).
    Higher = better.  total_score is the weighted sum of components.
    """
    scores = {}

    # Parse check
    if pred_atoms is None:
        return 0.0, {"parse": 0.0, "smact": 0.0, "bvs": 0.0, "density": 0.0,
                      "min_dist": 0.0, "elements": 0.0}

    scores["parse"] = 1.0

    # Element match
    if source_info["elements"] is not None:
        pred_elements = set(pred_atoms.get_chemical_symbols())
        scores["elements"] = 1.0 if pred_elements == source_info["elements"] else 0.0
    else:
        scores["elements"] = 1.0

    # Min interatomic distance
    try:
        dists = ase_neighbor_list("d", pred_atoms, cutoff=0.5, self_interaction=False)
        scores["min_dist"] = 0.0 if len(dists) > 0 else 1.0
    except Exception:
        scores["min_dist"] = 0.5  # uncertain

    # SMACT
    try:
        from smact.screening import smact_filter
        from smact import Element as SmactElement
        symbols = pred_atoms.get_chemical_symbols()
        from collections import Counter
        comp = Counter(symbols)
        elements_list = sorted(comp.keys())
        stoichs = [comp[e] for e in elements_list]
        smact_elems = [SmactElement(e) for e in elements_list]

        # Electronegativity ordering
        ens = [e.pauling_eneg for e in smact_elems]
        if None in ens:
            scores["smact"] = 0.5  # can't check
        else:
            en_ok = all(ens[i] <= ens[i+1] for i in range(len(ens)-1)) or len(ens) <= 1

            # Charge balance
            ox_combos = [e.oxidation_states for e in smact_elems]
            charge_ok = False
            import itertools
            for combo in itertools.product(*ox_combos):
                total = sum(ox * n for ox, n in zip(combo, stoichs))
                if total == 0:
                    charge_ok = True
                    break

            if en_ok and charge_ok:
                scores["smact"] = 1.0
            elif charge_ok:
                scores["smact"] = 0.5  # charge balanced but wrong EN order
            else:
                scores["smact"] = 0.0
    except Exception:
        scores["smact"] = 0.5

    # BVS — continuous score based on max deviation
    try:
        structure = AseAtomsAdaptor.get_structure(pred_atoms)
        bva = BVAnalyzer()
        valences = bva.get_valences(structure)
        max_dev = max(abs(v - round(v)) for v in valences)
        # Score: 1.0 at dev=0, 0.0 at dev>=1.0, linear in between
        scores["bvs"] = max(0.0, 1.0 - max_dev)
    except Exception:
        scores["bvs"] = 0.0  # BVS failure = worst score

    # Density accuracy — continuous
    if source_info["density"] is not None and source_info["density"] > 0:
        try:
            pred_vol = pred_atoms.get_volume()
            pred_mass = sum(pred_atoms.get_masses())
            pred_density = pred_mass / pred_vol * 1.6605
            rel_err = abs(pred_density - source_info["density"]) / source_info["density"]
            scores["density"] = max(0.0, 1.0 - rel_err / 0.3)
        except Exception:
            scores["density"] = 0.0
    else:
        scores["density"] = 0.5

    # Weighted total — BVS and SMACT weighted highest since those are our targets
    weights = {
        "parse": 1.0,
        "elements": 2.0,
        "min_dist": 1.0,
        "smact": 3.0,
        "bvs": 3.0,
        "density": 1.0,
    }
    total = sum(weights[k] * scores[k] for k in weights)
    max_total = sum(weights.values())
    normalized = total / max_total

    return normalized, scores


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _progress_path(output_path):
    return output_path + ".progress"


def _read_progress(output_path):
    pp = _progress_path(output_path)
    if not os.path.exists(pp):
        return 0
    try:
        with open(pp, "r") as f:
            return int(json.load(f)["rows_processed"])
    except Exception:
        return 0


def _write_progress(output_path, rows_processed, pairs_written, stats):
    pp = _progress_path(output_path)
    with open(pp, "w") as f:
        json.dump({
            "rows_processed": rows_processed,
            "pairs_written": pairs_written,
            "stats": stats,
        }, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS = ["source", "chosen", "rejected", "score"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate DPO v2 preference pairs (best-of-N)")
    p.add_argument("--data", required=True, help="Input CSV with source/target columns")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--sp_model", default="checkpoints/model_sp.model",
                   help="SentencePiece model path")
    p.add_argument("--output", default="dpo_v2_pairs.csv", help="Output CSV path")
    p.add_argument("--n", type=int, default=None, help="Max prompts to process")
    p.add_argument("--K", type=int, default=8, help="Number of samples per prompt")
    p.add_argument("--min_gap", type=float, default=0.15,
                   help="Minimum score gap between best and worst to create a pair")
    p.add_argument("--max_length", type=int, default=512, help="Generation max length")
    p.add_argument("--top_k", type=int, default=10, help="Sampling top-k")
    p.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--no_resume", action="store_true")
    return p.parse_args()


def load_model(ckpt_path, device):
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

    # Load data
    with open(args.data, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.n is not None:
        rows = rows[:args.n]

    if args.shard is not None and args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: processing {len(rows)} prompts")

    sources = [r["source"] for r in rows]
    total = len(sources)
    print(f"Loaded {total} prompts from {args.data}")
    print(f"K={args.K} samples/prompt, min_gap={args.min_gap}")

    # Resume
    if args.no_resume:
        already_done = 0
        for f_path in [args.output, _progress_path(args.output)]:
            if os.path.exists(f_path):
                os.remove(f_path)
    else:
        already_done = _read_progress(args.output)

    if already_done >= total:
        print(f"Already complete ({already_done}/{total}).")
        return

    if already_done > 0:
        print(f"Resuming from prompt {already_done}/{total}")

    # Output CSV
    if already_done == 0:
        out_f = open(args.output, "w", encoding="utf-8", newline="")
        out_writer = csv.DictWriter(out_f, fieldnames=_OUTPUT_FIELDS)
        out_writer.writeheader()
        out_f.flush()
    else:
        out_f = open(args.output, "a", encoding="utf-8", newline="")
        out_writer = csv.DictWriter(out_f, fieldnames=_OUTPUT_FIELDS)

    # Load model
    model, config = load_model(args.checkpoint, device)
    sp = load_sp(args.sp_model)

    # Process prompts
    remaining = sources[already_done:]
    rows_processed = already_done
    pairs_written = 0
    score_gaps = []
    t0 = time.time()

    # Process in batches of prompts; for each prompt, generate K samples
    prompt_batch_size = max(1, args.batch_size // args.K)

    for i in range(0, len(remaining), prompt_batch_size):
        prompt_batch = remaining[i:i + prompt_batch_size]

        for src in prompt_batch:
            source_info = parse_source(src)

            # Generate K samples by repeating the source K times
            src_repeated = [src] * args.K
            # Generate in sub-batches if K is large
            all_preds = []
            for k_start in range(0, args.K, args.batch_size):
                k_batch = src_repeated[k_start:k_start + args.batch_size]
                preds = translate_batch(model, sp, k_batch, device,
                                        max_length=args.max_length,
                                        top_k=args.top_k)
                all_preds.extend(preds)

            # Score all K samples
            scored = []
            for pred_str in all_preds:
                pred_atoms = parse_target(pred_str)
                total_score, components = score_structure(pred_str, pred_atoms, source_info)
                scored.append((total_score, pred_str, components))

            # Sort by score
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_str, best_comp = scored[0]
            worst_score, worst_str, worst_comp = scored[-1]
            gap = best_score - worst_score

            if gap >= args.min_gap:
                score_detail = (f"gap={gap:.3f} "
                                f"best={best_score:.3f}({_comp_str(best_comp)}) "
                                f"worst={worst_score:.3f}({_comp_str(worst_comp)})")
                out_writer.writerow({
                    "source": src,
                    "chosen": best_str,
                    "rejected": worst_str,
                    "score": score_detail,
                })
                pairs_written += 1
                score_gaps.append(gap)

            rows_processed += 1

        out_f.flush()
        _write_progress(args.output, rows_processed, pairs_written,
                        {"mean_gap": float(np.mean(score_gaps)) if score_gaps else 0})

        # Progress
        elapsed = time.time() - t0
        done_since_resume = rows_processed - already_done
        rate = done_since_resume / elapsed if elapsed > 0 else 0
        eta_remaining = (total - rows_processed) / rate if rate > 0 else 0
        if done_since_resume % 50 < prompt_batch_size or rows_processed == total:
            print(f"  {rows_processed}/{total}  "
                  f"{pairs_written} pairs  "
                  f"(mean_gap={np.mean(score_gaps):.3f} "
                  f"| {rate:.1f} prompts/s, ETA {eta_remaining/60:.0f}m)")

    out_f.close()

    # Cleanup
    pp = _progress_path(args.output)
    if os.path.exists(pp):
        os.remove(pp)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Summary
    print(f"\nResults: {pairs_written}/{total} pairs "
          f"({pairs_written/total*100:.1f}% of prompts yielded pairs)")
    if score_gaps:
        print(f"Score gap: mean={np.mean(score_gaps):.3f}, "
              f"median={np.median(score_gaps):.3f}, "
              f"max={max(score_gaps):.3f}")


def _comp_str(components):
    """Compact string of component scores."""
    return ",".join(f"{k[0]}={v:.1f}" for k, v in components.items())


if __name__ == "__main__":
    main()

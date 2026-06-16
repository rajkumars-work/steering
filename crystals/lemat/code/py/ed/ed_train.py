"""Training script for encoder-decoder transformer on any CSV of (source, target) pairs.

Supports single-GPU and multi-GPU (via torchrun) training with gradient accumulation,
EMA weight averaging, cosine LR schedule, and periodic evaluation/checkpointing.

Continuing training:
    Resume is automatic.  When --checkpoint points to a directory with .pt files,
    the most recent checkpoint is loaded and training continues from the saved step.
    No --resume flag is needed — just make sure the step budget (--max_steps, or
    --max_steps 0 --epochs N) exceeds the checkpoint's step.  Optimizer state is
    restored automatically (pass --fresh_optimizer to reset it).

Usage:
    # Single GPU
    python ed_train.py --data data.csv --checkpoint checkpoints/run1

    # Multi-GPU
    torchrun --nproc_per_node=4 ed_train.py --data data.csv --checkpoint checkpoints/run1

    # Continue training (same checkpoint dir, higher step budget)
    torchrun --nproc_per_node=4 ed_train.py --data data.csv --checkpoint checkpoints/run1 \\
        --max_steps 8000

See TRAINING.md for full parameter reference and examples.
"""

import argparse
import os
import signal
import sys
from contextlib import nullcontext
from pathlib import Path

# Ignore SIGHUP so training survives terminal disconnect.
# For full protection (torchrun, process groups), use launch.py.
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import copy

from configs import Config, config_ed, get_device
from ed_data import build_tokenizer, get_loader, load_sp
from ed_model import EdGPT, get_lr
from expert_xattn_model import ExpertXAttnEdGPT
from d_model import DecoderOnlyGPT
from run_registry import RunRecord

# Module-level ref so we can save on interrupt from __main__
_active_run_rec: RunRecord | None = None


def _ema_key(name):
    # Normalize torch.compile's "_orig_mod." prefix so EMA shadow keys match
    # model.named_parameters() whether compiled or not, fresh or resumed. Without this a
    # compiled RESUME raises KeyError('_orig_mod....'): the restored shadow has unprefixed
    # keys but the compiled model yields prefixed ones.
    return name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name


class EMA:
    """Exponential Moving Average of model parameters (compile-prefix agnostic)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {_ema_key(name): p.data.clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[_ema_key(name)].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {_ema_key(k): v for k, v in state_dict.items()}

    def apply_to(self, model):
        """Temporarily swap model weights with EMA weights. Returns backup."""
        backup = {}
        for name, p in model.named_parameters():
            backup[name] = p.data.clone()
            p.data.copy_(self.shadow[_ema_key(name)])
        return backup

    def restore(self, model, backup):
        """Restore original weights from backup."""
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])


## PlateauTracker removed — it was too aggressive and killed training runs
## by halving LR repeatedly before the model had time to learn.
## TODO: revisit with a better design (e.g. longer patience, min LR floor,
## or only activate after warmup completes).
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Train encoder-decoder transformer")
    p.add_argument("--data", required=True, help="Path to CSV with 'source' and 'target' columns")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs (the STOP point)")
    p.add_argument("--lr_schedule_epochs", type=int, default=0,
                   help="If >0, anchor the cosine LR schedule to this many epochs, DECOUPLED "
                        "from --epochs (the stop point). Lets you train fewer epochs while "
                        "keeping the LR on a longer, less-decayed curve (e.g. --epochs 75 "
                        "--lr_schedule_epochs 150 stops at 75 with LR still ~3.4e-5 instead of "
                        "flooring at 5e-6). Default 0 = use --epochs (unchanged behavior).")
    p.add_argument("--skip_resume_replay", action="store_true",
                   help="On resume, do NOT replay/consume resume_step*grad_accum batches to "
                        "restore the exact data-iterator position. The replay is GPU-idle, "
                        "tokenize-only work that scales with the resume step (~30min+ here) — "
                        "the silent phase that looks like a stall. Skipping it makes resume "
                        "near-instant; cost is only that data ORDER from the resume point "
                        "differs (irrelevant for shuffled warm-restart training).")
    p.add_argument("--max_steps", type=int, default=4000,
                   help="Optional cap on optimizer steps (default: 4000; set 0 to derive from epochs)")
    p.add_argument(
        "--checkpoint",
        default="./checkpoints",
        help="Checkpoint directory or checkpoint file. If a directory is given, "
             "the latest checkpoint in that directory is used if present.",
    )
    p.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size for SP tokenizer")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size per device (default: 64 for stability)")
    p.add_argument("--max_seq_len", type=int, default=512,
                   help="Maximum sequence length (default: 512 — RECAST-Lite default; "
                        "autobin token p99=465 so ~0.5%% truncation. Set 1024 for legacy training.)")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Maximum learning rate (default: 5e-5 — RECAST-Lite default. "
                        "Old default was 1e-3; that's only safe for short max_steps. Use 1e-4 for "
                        "ed-control reproductions.)")
    p.add_argument("--edim", type=int, default=None, help="Embedding dimension (default: from config_ed)")
    p.add_argument("--layers", type=int, default=None, help="Number of layers (default: from config_ed)")
    p.add_argument("--enc_layers", type=int, default=None,
                   help="Encoder layer count (ed only). Overrides --layers for encoder. "
                        "Use with --dec_layers for asymmetric ed (e.g. --enc_layers 6 --dec_layers 18).")
    p.add_argument("--dec_layers", type=int, default=None,
                   help="Decoder layer count (ed only). Overrides --layers for decoder.")
    p.add_argument("--heads", type=int, default=None, help="Number of attention heads (default: from config_ed)")
    p.add_argument("--log_interval", type=int, default=50, help="Log loss every N steps")
    p.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    p.add_argument("--max_keep_checkpoints", type=int, default=0,
                   help="If >0, keep only the N most recent step checkpoints "
                        "(ed_ckpt_NNNNNN.pt); best_probe/final/ema are never pruned. "
                        "0 = keep all (default, preserves legacy behavior).")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--device", default="cuda", help="GPU device to use")
    p.add_argument("--prefix", default="ed", help="Checkpoint filename prefix (default: ed)")
    p.add_argument("--prop_dropout", type=float, default=0.2,
                   help="Probability of dropping each source segment (default: 0.2)")
    p.add_argument("--label_dropout", type=float, default=0.0,
                   help="Inner per-label dropout within kept props segment. "
                        "0 (default) = current behavior. 0.2 = each label drops "
                        "independently with p=0.2 when the segment survives the "
                        "outer segment-level drop.")
    p.add_argument(
        "--fresh_optimizer",
        action="store_true",
        help="Discard saved optimizer state on resume and start with a fresh optimizer.",
    )
    p.add_argument("--model_type", default="ed", choices=["ed", "expert_xattn", "d"],
                   help="Model architecture: 'ed' (encoder-decoder, default), "
                        "'expert_xattn' (expert cross-attention), or "
                        "'d' (decoder-only with concatenated source+target, prefix-LM)")
    p.add_argument("--warmstart", default=None,
                   help="Path to pretrained EdGPT checkpoint for warm-starting expert_xattn model")
    p.add_argument("--n_expert_xattn_layers", type=int, default=4,
                   help="Number of decoder layers (from end) with expert cross-attention (default: 4)")
    p.add_argument("--eval_interval", type=int, default=500,
                   help="Compute eval loss every N steps (default: 500, 0=disabled). Uses 'split' column in CSV.")
    p.add_argument("--eval_batches", type=int, default=None,
                   help="Number of batches to average for eval loss (default: train batch size)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes per rank (default: 4)")
    p.add_argument("--grad_clip_norm", type=float, default=0.5,
                   help="Gradient clipping max norm (default: 0.5 for stability)")
    p.add_argument("--detect_nan", action="store_true", default=True,
                   help="Detect and log NaN/Inf in loss and gradients (default: True)")
    p.add_argument("--skip_norm", type=float, default=0,
                   help="Skip optimizer step if grad norm exceeds this value (0=disabled)")
    # Plateau LR reduction removed — see PlateauTracker comment above
    p.add_argument("--plateau_patience", type=int, default=0,
                   help="(disabled) Was: halve LR after N evals without improvement")
    p.add_argument("--plateau_factor", type=float, default=0.5,
                   help="(disabled) Was: factor to multiply LR by on plateau")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay rate for weight averaging (0=disabled, default: 0.999)")
    p.add_argument("--warmup_frac", type=float, default=0.15,
                   help="Fraction of max_steps for LR warmup (default: 0.15 — RECAST-Lite default. "
                        "Old default was 0.05, which is fine for short runs but undertrains warmup at 150 epochs.)")
    p.add_argument("--total_batch_size", type=int, default=524_288,
                   help="Target total batch size in tokens for gradient accumulation (default: 524288)")
    # v2 architecture: stabilization features for thin-source formats
    p.add_argument("--qk_norm", action="store_true", default=True,
                   help="L2-normalize Q and K in cross-attention (default: on)")
    p.add_argument("--no_qk_norm", action="store_true",
                   help="Disable QK-norm in cross-attention (for backward compatibility)")
    p.add_argument("--gated_cross_attn", action="store_true",
                   help="Learnable gate on cross-attention output, initialized to 0 (Flamingo-style)")
    p.add_argument("--encoder_lr_scale", type=float, default=1.0,
                   help="Scale encoder LR relative to decoder (e.g. 0.2 = encoder gets 0.2x the LR)")
    p.add_argument("--probe_interval", type=int, default=1000,
                   help="Run structural validity probe every N steps (default: 1000; 0=disabled). "
                        "Generates structures from eval sources and checks tier-0 validity (SMACT, "
                        "BVS, min-distance, spacegroup) without relaxation. ~10s overhead per probe. "
                        "Replaced cond_probe as the default since 2026-05-06 (cond_probe is dielectric-specific).")
    p.add_argument("--probe_n", type=int, default=15,
                   help="Number of structures to generate per probe (default: 15)")
    p.add_argument("--probe_version_id", default="d15_binrho_k7",
                   help="Version-spec id used to parse probe-generated targets. Default "
                        "'d15_binrho_k7' matches the production d15 target schema "
                        "(anon|formula|SG|scratchpad|lattice|atoms). get_dataset_info() "
                        "auto-detection is unreliable on custom CSVs (falls back to 'd6', "
                        "which mis-parses d15 generations -> gen_ok=0).")
    p.add_argument("--best_probe_metric", default="tier0",
                   choices=["tier0", "semi"],
                   help="Which probe metric to use for saving best_probe checkpoint. "
                        "Use 'semi' for topological fine-tuning (saves when semi-probe SG "
                        "correct count improves). Default: 'tier0' (structural validity).")
    p.add_argument("--cond_probe_interval", type=int, default=0,
                   help="Run dielectric-specific cond probe every N steps (0=disabled, default). "
                        "DEPRECATED for general use — too domain-specific (it bins generations against "
                        "bandgap and eps_0 targets via nequip MLIPs). Use --probe_interval (structural "
                        "validity probe) for the general quick-check signal. cond_probe is still useful "
                        "for dielectric-conditioned models where you want explicit bg/eps compliance "
                        "tracked during training. ~30-60s per probe.")
    return p.parse_args()


def resolve_checkpoint_paths(checkpoint_arg: str) -> tuple[Path, Path | None]:
    """Resolve checkpoint directory and optional resume checkpoint from one path."""
    checkpoint_path = Path(checkpoint_arg)

    if checkpoint_path.exists() and checkpoint_path.is_file():
        return checkpoint_path.parent, checkpoint_path

    if checkpoint_path.exists() and checkpoint_path.is_dir():
        candidates = sorted(
            (p for p in checkpoint_path.glob("*.pt") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        latest = candidates[-1] if candidates else None
        return checkpoint_path, latest

    if checkpoint_path.suffix == ".pt":
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return checkpoint_path, None


def generate(model, sp, src_text, device, max_length=64, top_k=10):
    """Generate a translation for a single source string.

    Dispatches on model type: encoder-decoder (EdGPT, ExpertXAttnEdGPT)
    uses cached encoder output and a separate decoder loop; decoder-only
    (DecoderOnlyGPT) prepends source as a causal prefix.
    """
    # Detect decoder-only model by absence of `encode` method.
    if not hasattr(model, "encode"):
        return _generate_d(model, sp, src_text, device, max_length=max_length, top_k=top_k)

    model.eval()
    src_ids = sp.encode(src_text, out_type=int)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.ones(1, len(src_ids), dtype=torch.bool, device=device)

    pad_id = sp.pad_id()
    eos_id = sp.eos_id()
    bos_id = sp.bos_id()

    tokens = torch.full((1, max_length), pad_id, dtype=torch.long, device=device)
    tokens[0, 0] = bos_id
    cur_len = 1

    with torch.no_grad():
        encoder_out = model.encode(src_tensor, src_mask).contiguous()

        while cur_len < max_length:
            tgt_mask = torch.zeros(1, max_length, dtype=torch.bool, device=device)
            tgt_mask[0, :cur_len] = True

            logits, _ = model(tokens, targets=None, tgt_mask=tgt_mask, encoder_out=encoder_out)
            logits = logits[0, cur_len - 1, :]
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            next_idx = torch.multinomial(top_probs, 1)
            next_token = top_indices[next_idx].item()

            tokens[0, cur_len] = next_token
            cur_len += 1

            if next_token == eos_id:
                break

    result_ids = tokens[0, 1:cur_len].tolist()
    result_ids = [t for t in result_ids if t != eos_id]
    model.train()
    return sp.decode(result_ids)


def _generate_d(model, sp, src_text, device, max_length=64, top_k=10):
    """Decoder-only generation: source as causal prefix, append target tokens.

    Mirrors the encoder-decoder `generate` API. The full prefix
    `[src; BOS; tgt_so_far]` is re-fed at each step (no KV cache yet —
    inference cost is O(n²) in target length but fine at the lengths we
    use in probes).
    """
    model.eval()
    src_ids = sp.encode(src_text, out_type=int)
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # Start with [src_ids, BOS]. Decoder is causal — tokens before BOS form
    # the conditioning prefix.
    prefix = src_ids + [bos_id]
    src_len = len(src_ids)
    tokens = list(prefix)

    # Cap so prefix + generated <= model context.
    tdim = getattr(model, "config", None).tdim if hasattr(model, "config") else 1024
    max_tokens = min(tdim, src_len + max_length)

    with torch.no_grad():
        while len(tokens) < max_tokens:
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            mask = torch.ones(x.shape, dtype=torch.bool, device=device)
            logits, _ = model(x, targets=None, key_padding_mask=mask)
            # next-token prediction is at the last position
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            next_idx = torch.multinomial(top_probs, 1)
            next_token = top_indices[next_idx].item()
            tokens.append(next_token)
            if next_token == eos_id:
                break

    # Strip the source prefix (everything before BOS) and the BOS token,
    # plus any EOS at the end.
    target_ids = tokens[src_len + 1:]
    if target_ids and target_ids[-1] == eos_id:
        target_ids = target_ids[:-1]
    model.train()
    return sp.decode(target_ids)


@torch.no_grad()
def eval_loss(model, eval_loader, device, n_batches, use_xattn=False, use_decoder_only=False):
    """Compute average loss over n_batches from eval_loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    eval_iter = iter(eval_loader)
    device_type = device.type
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    for _ in range(n_batches):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break
        if use_decoder_only:
            input_ids, target_ids, padding_mask, _src_lens = batch
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True).bool()
            with autocast_ctx:
                _, loss = model(input_ids, targets=target_ids, key_padding_mask=padding_mask)
        else:
            if use_xattn:
                src, dec_in, labels, src_mask, tgt_mask, wp_class = batch
                wp_class = wp_class.to(device).long()
            else:
                src, dec_in, labels, src_mask, tgt_mask = batch
                wp_class = None
            src = src.to(device, non_blocking=True)
            dec_in = dec_in.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True).bool()
            tgt_mask = tgt_mask.to(device, non_blocking=True).bool()
            with autocast_ctx:
                _, loss = model(
                    dec_in, src, targets=labels, src_mask=src_mask, tgt_mask=tgt_mask,
                    **({"wp_class": wp_class} if wp_class is not None else {})
                )
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def semimetal_probe(model, sp, device, version_id, max_length=512, top_k=10):
    """Generate known topological semimetal compositions and check spacegroups.

    Tests whether the model generates the correct crystal structure (not just
    correct composition) for known topological materials. The target SGs are
    the experimentally confirmed topological phases.

    Returns dict with n_correct_sg, n_gen_ok, per-material results.
    """
    try:
        from dielectric_data.reader import parse_target, robust_fuzzy_parse
    except ImportError:
        return None

    import spglib

    # Known topological materials with target spacegroups
    TARGETS = [
        # (source_prompt, formula, target_sg, description)
        ("Nb As | 4 | 8.5 | metal stable tetragonal topological-semimetal",
         "NbAs", 109, "Weyl I4_1md"),
        ("Nb P | 4 | 7.5 | metal stable tetragonal topological-semimetal",
         "NbP", 109, "Weyl I4_1md"),
        ("Ta As | 4 | 11.0 | metal stable tetragonal topological-semimetal",
         "TaAs", 109, "Weyl I4_1md"),
        ("Ta P | 4 | 10.0 | metal stable tetragonal topological-semimetal",
         "TaP", 109, "Weyl I4_1md"),
        ("Mo P | 2 | 7.2 | metal stable hexagonal topological-semimetal",
         "MoP", 187, "triple-point P-6m2"),
        ("Co Si | 8 | 6.6 | metal stable cubic topological-semimetal",
         "CoSi", 198, "chiral P2_13"),
        ("Rh Si | 8 | 8.0 | metal stable cubic topological-semimetal",
         "RhSi", 198, "chiral P2_13"),
        ("Zr Si S | 6 | 4.5 | metal stable tetragonal topological-semimetal",
         "ZrSiS", 129, "nodal-line P4/nmm"),
    ]

    model.eval()
    results = {"total": len(TARGETS), "gen_ok": 0, "correct_sg": 0, "details": []}

    for src, target_formula, target_sg, desc in TARGETS:
        detail = {"formula": target_formula, "target_sg": target_sg, "desc": desc}
        try:
            tgt_text = generate(model, sp, src, device, max_length=max_length, top_k=top_k)
            atoms = parse_target(tgt_text, version_id, "mp")
            if atoms is None:
                atoms = robust_fuzzy_parse(tgt_text)
            if atoms is None or len(atoms) == 0:
                detail["status"] = "parse_fail"
                results["details"].append(detail)
                continue

            results["gen_ok"] += 1

            # Get spacegroup
            sg_data = spglib.get_symmetry_dataset(
                (atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=0.1)
            gen_sg = sg_data["number"] if sg_data else 0
            detail["gen_sg"] = gen_sg
            detail["gen_formula"] = atoms.get_chemical_formula(mode="reduce")
            detail["status"] = "correct_sg" if gen_sg == target_sg else f"wrong_sg_{gen_sg}"

            if gen_sg == target_sg:
                results["correct_sg"] += 1

        except Exception as e:
            detail["status"] = f"error: {str(e)[:50]}"

        results["details"].append(detail)

    return results


def structural_probe(model, sp, eval_sources, device, version_id, n=10,
                     max_length=512, top_k=10, training_formulas=None):
    """Generate structures from eval sources and run tier-0 checks (no relaxation).

    The default in-training quick-check probe (since 2026-05-06). Domain-agnostic:
    uses only structural validity (SMACT charge balance, BVS bond-valence-sum,
    min interatomic distance, spacegroup detection) plus composition novelty.
    No MLIPs, no GPU memory beyond the training model itself.

    Tier-0 pass = SMACT ∧ BVS ∧ min_dist ∧ plausibility (matches the offline
    screening pipeline's tier-0 definition).

    Returns dict with pass rates for BVS, min_dist, spacegroup, SMACT, tier-0,
    plus composition novelty and formula diversity.
    Takes ~1-3s per structure on GPU. Use as a training-time quality signal.

    For dielectric-specific bg/eps_0 bin matching, see `cond_probe.py` (legacy,
    domain-specific; loads nequip MLIPs).
    """
    import random as _rng
    _probe_rng = _rng.Random(42)

    # Lazy imports — only needed when probe is active
    try:
        from dielectric_data.reader import parse_target, robust_fuzzy_parse
        from validity.structural import (
            check_min_dist, check_bvs, check_spacegroup, check_smact,
            check_plausibility, check_element_filter,
        )
    except ImportError:
        return None

    sources = _probe_rng.sample(eval_sources, min(n, len(eval_sources)))
    results = {"gen_ok": 0, "bvs": 0, "min_dist": 0, "sg": 0, "smact": 0,
               "plausibility": 0, "pair_dist": 0, "tier0": 0,
               "comp_novel": 0, "proto_novel": 0, "n_formulas": 0, "n_protos": 0, "total": len(sources)}
    seen_formulas = set()
    seen_protos = set()

    def _structure_fingerprint(atoms):
        """Compute (reduced_formula, spg_number, wyckoff_string) fingerprint.
        Symmetry-aware, immune to small lattice/position perturbations. ~4ms."""
        try:
            import spglib
            formula = atoms.get_chemical_formula(mode="reduce")
            sg_data = spglib.get_symmetry_dataset(
                (atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=0.1)
            spg_num = sg_data["number"] if sg_data else 0
            # Build Wyckoff string: sort by (element, letter), count multiplicities
            wl = atoms.info.get("wyckoff_letters", [])
            if wl and len(wl) == len(atoms):
                from collections import Counter
                syms = atoms.get_chemical_symbols()
                wp_counts = Counter(f"{s}_{w}" for s, w in zip(syms, wl))
                wp_str = ";".join(f"{k}:{v}" for k, v in sorted(wp_counts.items()))
            else:
                wp_str = ""
            return (formula, spg_num, wp_str)
        except Exception:
            return None

    model.eval()
    for src_text in sources:
        # Determine origin from source content (heuristic: if it has crystal_sys, it's mp)
        origin = "mp"
        try:
            tgt_text = generate(model, sp, src_text, device, max_length=max_length, top_k=top_k)
            atoms = parse_target(tgt_text, version_id, origin)
            if atoms is None:
                atoms = robust_fuzzy_parse(tgt_text)
            if atoms is None or len(atoms) == 0:
                continue
            results["gen_ok"] += 1

            # Composition novelty and diversity
            try:
                formula = atoms.get_chemical_formula(mode="reduce")
                seen_formulas.add(formula)
                if training_formulas is not None and formula not in training_formulas:
                    results["comp_novel"] += 1
            except Exception:
                pass

            # Structural prototype fingerprint (symmetry-aware)
            fp = _structure_fingerprint(atoms)
            if fp is not None:
                seen_protos.add(fp)

            # Run tier-0 checks
            md = check_min_dist(atoms)
            bv = check_bvs(atoms)
            sg = check_spacegroup(atoms)
            sm = check_smact(atoms)
            pl = check_plausibility(atoms)

            md_ok = md.get("min_dist_pass", False)
            bv_ok = bv.get("bvs_pass", False)
            sg_ok = sg.get("spacegroup_pass", False)
            sm_ok = sm.get("smact_pass", False)
            pl_ok = pl.get("plausibility_pass", False)

            # Element-pair-aware distance check (GenBench style)
            import numpy as np
            from ase.data import covalent_radii as _cr, atomic_numbers as _an
            pair_ok = True
            try:
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                syms = atoms.get_chemical_symbols()
                for i in range(len(atoms)):
                    for j in range(i+1, len(atoms)):
                        thresh = (0.7 + _cr[_an[syms[i]]] + _cr[_an[syms[j]]]) * 0.5
                        if dists[i,j] < thresh:
                            pair_ok = False
                            break
                    if not pair_ok:
                        break
            except Exception:
                pair_ok = False

            if md_ok: results["min_dist"] += 1
            if bv_ok: results["bvs"] += 1
            if sg_ok: results["sg"] += 1
            if sm_ok: results["smact"] += 1
            if pl_ok: results["plausibility"] += 1
            if pair_ok: results["pair_dist"] += 1
            if md_ok and bv_ok and sg_ok and sm_ok:
                results["tier0"] += 1
        except Exception:
            continue

    results["n_formulas"] = len(seen_formulas)
    results["n_protos"] = len(seen_protos)
    del results["proto_novel"]  # needs training set comparison — done at screen time
    model.train()
    return results


def train(args):
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_rank = int(os.environ.get("RANK", 0))
    master_process = ddp_rank == 0
    checkpoint_dir, resume_path = resolve_checkpoint_paths(args.checkpoint)

    # Init DDP early so we can use barriers during setup
    if ddp_world_size > 1:
        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # Auto-train SentencePiece if needed — save in checkpoint dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Emit a checkpoint metadata sidecar so the corpus registry can pick up
    # explicit (non-inferred) provenance for this run. Master-only so we
    # don't write from every DDP rank.
    if master_process:
        try:
            import sys as _sys
            _sys.path.insert(0, "/data/rkumar/code/py/dielectric")
            from chem.corpus_provenance import write_checkpoint_metadata
            data_csv = Path(args.data).name if args.data else None
            hyperparams = {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "max_seq_len": args.max_seq_len,
                "prop_dropout": args.prop_dropout,
                "label_dropout": args.label_dropout,
                "vocab_size": args.vocab_size,
                "ddp_world_size": ddp_world_size,
            }
            write_checkpoint_metadata(
                str(checkpoint_dir),
                training_dataset=Path(args.data).stem if args.data else None,
                training_dataset_csv=data_csv,
                hyperparams=hyperparams,
            )
        except Exception as _e:
            print(f"  [warn] metadata.json emit failed: {_e}", flush=True)

    sp_prefix = str(checkpoint_dir / "model_sp")
    sp_model_path = str(checkpoint_dir / "model_sp.model")
    if not Path(sp_model_path).exists():
        if master_process:
            print(f"Training SentencePiece tokenizer (vocab_size={args.vocab_size})...")
            build_tokenizer(args.data, sp_prefix, args.vocab_size)
        if ddp_world_size > 1:
            dist.barrier()

    sp = load_sp(sp_model_path)
    actual_vocab_size = sp.get_piece_size()

    # Build config
    c = Config(
        name="ed",
        edim=args.edim or config_ed.edim,
        layers=args.layers or config_ed.layers,
        heads=args.heads or config_ed.heads,
        tdim=args.max_seq_len,
        dropout=0.1,
        vocab_size=actual_vocab_size,
        B=args.batch_size,
        lr=args.lr,
        qk_norm=args.qk_norm and not args.no_qk_norm,
        gated_cross_attn=args.gated_cross_attn,
        enc_layers=args.enc_layers if args.enc_layers is not None else config_ed.enc_layers,
        dec_layers=args.dec_layers if args.dec_layers is not None else config_ed.dec_layers,
    )

    use_xattn = args.model_type == "expert_xattn"
    use_decoder_only = args.model_type == "d"

    def _build_model():
        if use_xattn:
            return ExpertXAttnEdGPT(c, n_expert_xattn_layers=args.n_expert_xattn_layers)
        if use_decoder_only:
            return DecoderOnlyGPT(c)
        return EdGPT(c)

    if ddp_world_size > 1:
        m = _build_model().to(device)
        m.train()
        m = DDP(m, device_ids=[local_rank])
        if args.compile:
            m = torch.compile(m)
        raw_m = m.module
    else:
        device = torch.device(get_device(args.device))
        m = _build_model().to(device)
        m.train()
        if args.compile:
            m = torch.compile(m)
        # raw_m must be the UNDERLYING module: torch.compile wraps m as an OptimizedModule
        # whose keys are "_orig_mod."-prefixed. Checkpoint load / EMA / save use raw_m with
        # UNprefixed keys, so raw_m=m (the wrapper) silently loads zero keys -> random init
        # (loss ~9.8). m._orig_mod shares the same params the compiled forward uses.
        raw_m = m._orig_mod if args.compile else m

    device_type = device.type

    if args.encoder_lr_scale != 1.0 and use_decoder_only:
        if master_process:
            print(f"WARNING: --encoder_lr_scale={args.encoder_lr_scale} is a no-op for "
                  f"--model_type d (no encoder). Using uniform LR.")
    if args.encoder_lr_scale != 1.0 and not use_decoder_only:
        # Separate param groups for encoder vs decoder with different LR
        encoder_names = {"transformer.e.", "transformer.ete.", "transformer.epe.", "transformer.ln_e_f."}
        enc_decay, enc_nodecay, dec_decay, dec_nodecay = [], [], [], []
        for name, p in raw_m.named_parameters():
            if not p.requires_grad:
                continue
            is_enc = any(name.startswith(prefix) for prefix in encoder_names)
            is_decay = p.dim() >= 2
            if is_enc:
                (enc_decay if is_decay else enc_nodecay).append(p)
            else:
                (dec_decay if is_decay else dec_nodecay).append(p)
        fused_available = "fused" in __import__("inspect").signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW([
            {"params": enc_decay, "weight_decay": 0.1, "lr_scale": args.encoder_lr_scale},
            {"params": enc_nodecay, "weight_decay": 0.0, "lr_scale": args.encoder_lr_scale},
            {"params": dec_decay, "weight_decay": 0.1, "lr_scale": 1.0},
            {"params": dec_nodecay, "weight_decay": 0.0, "lr_scale": 1.0},
        ], lr=c.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        if master_process:
            print(f"Encoder LR scale: {args.encoder_lr_scale} ({len(enc_decay)+len(enc_nodecay)} params)")
    if (args.encoder_lr_scale == 1.0) or use_decoder_only:
        optimizer = raw_m.configure_optimizers(
            weight_decay=0.1, learning_rate=c.lr, device_type=device_type
        )

    # Warm-start expert_xattn from pretrained EdGPT (before resume)
    if args.warmstart and use_xattn:
        from load_pretrained_xattn import load_pretrained_into_xattn
        pretrained = EdGPT(c).to(device)
        ws_ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
        ws_state = {k.removeprefix("_orig_mod."): v for k, v in ws_ckpt["model"].items()}
        pretrained.load_state_dict(ws_state)
        load_pretrained_into_xattn(pretrained, raw_m)
        del pretrained, ws_ckpt, ws_state
        if master_process:
            print(f"Warm-started expert_xattn from {args.warmstart}")

    # Resume from checkpoint if requested
    resume_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        # Handle v1→v2 checkpoint loading: if checkpoint lacks QK-norm weights
        # but model expects them, load with strict=False and reinitialize missing keys
        try:
            raw_m.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key" in str(e) and "q_norm" in str(e):
                if master_process:
                    print("Loading v1 checkpoint into v2 model (QK-norm weights will be initialized fresh)")
                raw_m.load_state_dict(state_dict, strict=False)
            else:
                raise
        if not args.fresh_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        resume_step = ckpt.get("step", 0)
        _saved_ema_state = ckpt.get("ema")  # Save for later EMA init (may be None)
        if master_process:
            opt_state_msg = "fresh optimizer (--fresh_optimizer)" if args.fresh_optimizer else "restored optimizer state"
            print(f"Resumed from {resume_path} at step {resume_step} ({opt_state_msg})")

    # Count steps from dataset size
    train_split = "train" if args.eval_interval > 0 else None
    loader = get_loader(
        args.data,
        sp_model_path,
        args.batch_size,
        args.max_seq_len,
        prop_dropout=args.prop_dropout,
        label_dropout=args.label_dropout,
        emit_wp_class=use_xattn,
        split=train_split,
        num_workers=args.num_workers,
        pin_memory=device_type == "cuda",
        distributed_rank=ddp_rank if ddp_world_size > 1 else None,
        distributed_world_size=ddp_world_size if ddp_world_size > 1 else None,
        model_type=args.model_type,
    )
    batches_per_epoch = len(loader)
    train_sampler = loader.sampler if isinstance(loader.sampler, DistributedSampler) else None

    eval_batches = args.eval_batches if args.eval_batches is not None else args.batch_size

    # Eval loader (if eval_interval is set)
    eval_loader = None
    if args.eval_interval > 0:
        eval_loader = get_loader(
            args.data,
            sp_model_path,
            args.batch_size,
            args.max_seq_len,
            prop_dropout=0.0,
            emit_wp_class=use_xattn,
            split="eval",
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device_type == "cuda",
            model_type=args.model_type,
        )
        if master_process:
            print(f"Eval: {len(eval_loader.dataset):,} rows, {eval_batches} batches every {args.eval_interval} steps")

    total_batch_size = args.total_batch_size
    B = c.B if device_type == "cuda" else 2
    T = c.tdim
    grad_accum_steps = max(1, total_batch_size // (B * T * ddp_world_size))

    # max_steps = optimizer steps (each consumes grad_accum_steps micro-batches)
    steps_per_epoch = max(1, batches_per_epoch // grad_accum_steps)
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    # LR-schedule horizon: decoupled from the stop point (max_steps) when --lr_schedule_epochs
    # is set, so we can train fewer epochs while keeping the LR on a longer cosine curve.
    lr_max_steps = steps_per_epoch * args.lr_schedule_epochs if args.lr_schedule_epochs > 0 else max_steps

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    if master_process:
        n_params = sum(p.numel() for p in raw_m.parameters())
        print(f"Model: {n_params/1e6:.1f}M params | B:{B}, T:{T}, epochs:{args.epochs}, "
              f"steps/epoch:{steps_per_epoch}, max_steps:{max_steps}, grad_accum:{grad_accum_steps}")

        # Check target/source length ratio and warn if encoder LR scale may be needed
        try:
            import csv as _csv
            with open(args.data, newline="", encoding="utf-8") as _f:
                _reader = _csv.DictReader(_f)
                _src_lens, _tgt_lens = [], []
                for _i, _row in enumerate(_reader):
                    _src_lens.append(len(_row.get("source", "").split()))
                    _tgt_lens.append(len(_row.get("target", "").split()))
                    if _i >= 999:
                        break
                if _src_lens and _tgt_lens:
                    _ratio = (sum(_tgt_lens) / len(_tgt_lens)) / max(sum(_src_lens) / len(_src_lens), 1)
                    print(f"Target/source token ratio: {_ratio:.1f}x")
                    if _ratio > 10 and args.encoder_lr_scale == 1.0:
                        _suggested = max(0.1, round(1.0 / (_ratio ** 0.5), 2))
                        print(f"WARNING: Target/source ratio {_ratio:.0f}x > 10x but --encoder_lr_scale is 1.0.")
                        print(f"  Encoder gradients are amplified ~{_ratio:.0f}x. Consider --encoder_lr_scale {_suggested}")
                        print(f"  See ARCHITECTURE.md for details.")
        except Exception:
            pass

    # --- Run registry ---
    global _active_run_rec
    run_rec = None
    if master_process:
        run_rec = RunRecord.start(args, c)
        _active_run_rec = run_rec
        run_rec.set_training_info(
            n_params=sum(p.numel() for p in raw_m.parameters()),
            grad_accum_steps=grad_accum_steps,
            steps_per_epoch=steps_per_epoch,
            effective_max_steps=max_steps,
            ddp_world_size=ddp_world_size,
            total_batch_size=total_batch_size,
        )
        run_rec.set_data_info(
            train_rows=len(loader.dataset),
            eval_rows=len(eval_loader.dataset) if eval_loader else 0,
        )
        # Try to get version from dataset sidecar
        try:
            from dielectric_data.reader import get_dataset_info
            info = get_dataset_info(args.data)
            run_rec.set_data_info(version_id=info.get("version_id", ""))
        except Exception:
            pass
        # Auto-detect latest audit for the data CSV
        try:
            _audit_index_path = Path(args.data).resolve().parent / "audits" / "index.json"
            if _audit_index_path.exists():
                import json as _json
                _audits = _json.loads(_audit_index_path.read_text(encoding="utf-8"))
                _csv_name = Path(args.data).resolve().name
                _matching = [a for a in _audits if Path(a.get("csv", "")).name == _csv_name]
                if _matching:
                    _latest = max(_matching, key=lambda a: a.get("timestamp", ""))
                    run_rec.set_data_info(audit_id=_latest["file"])
        except Exception:
            pass
        if resume_path is not None:
            run_rec.set_resume_info(str(resume_path), resume_step)

        # Save run.sh for reproducibility
        try:
            run_sh = checkpoint_dir / "run.sh"
            if not run_sh.exists():
                run_sh.write_text(
                    "#!/bin/bash\n"
                    f"# Reproduce training run {run_rec.run_id}\n"
                    f"cd {Path.cwd()}\n"
                    f"{' '.join(sys.argv)}\n",
                    encoding="utf-8",
                )
        except Exception:
            pass

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Structural probe setup ---
    probe_sources = []
    probe_version_id = args.probe_version_id
    probe_training_formulas = None
    best_probe_tier0 = -1
    if args.probe_interval > 0 and master_process:
        try:
            import csv as _csv
            # probe_version_id comes from --probe_version_id (default d15_binrho_k7);
            # get_dataset_info() auto-detection is unreliable on custom CSVs.
            # Load eval sources and training formulas in one pass
            _train_formulas = set()
            with open(args.data, newline="", encoding="utf-8") as _f:
                _reader = _csv.DictReader(_f)
                for _row in _reader:
                    if _row.get("label", "") in ("eval", "test_1k", "test_100"):
                        probe_sources.append(_row["source"])
                    # Extract reduced formula from target for composition novelty
                    _tgt = _row.get("target", "")
                    _tgt_segs = _tgt.split("|")
                    # Formula is typically the second segment (after anon_elements) or first
                    for _seg in _tgt_segs[:3]:
                        _seg = _seg.strip()
                        # Simple heuristic: formula segments contain element symbols + numbers, no spaces in the formula itself
                        if _seg and not _seg.startswith(("SG ", "Eref ", "OX:", "CN:", "Ef:")) and not " " in _seg:
                            _train_formulas.add(_seg)
                            break
            probe_training_formulas = _train_formulas if _train_formulas else None
            print(f"Structural probe: {len(probe_sources)} eval sources, "
                  f"every {args.probe_interval} steps, n={args.probe_n}"
                  + (f", {len(_train_formulas)} training formulas" if _train_formulas else ""))
        except Exception as _e:
            print(f"Warning: Could not load probe sources: {_e}")
            probe_sources = []

    train_data_epoch = 0
    skip_count_epoch = 0   # steps skipped due to high grad norm in current epoch
    skip_count_total = 0   # total skips across all epochs
    current_epoch_int = 0  # tracks epoch boundary for reporting

    # EMA of model weights
    ema = None
    if args.ema_decay > 0:
        ema = EMA(raw_m, decay=args.ema_decay)
        # Restore saved EMA state from checkpoint if available
        _ema_state = locals().get("_saved_ema_state")
        if _ema_state is not None:
            ema.load_state_dict({
                k.removeprefix("_orig_mod."): v
                for k, v in _ema_state.items()
            })
            if master_process:
                print(f"EMA enabled (decay={args.ema_decay}, restored from checkpoint)")
        elif master_process:
            ema_msg = "initialized from model weights"
            if resume_step > 0:
                ema_msg += " (WARNING: no saved EMA in checkpoint — probes will be unreliable until EMA converges)"
            print(f"EMA enabled (decay={args.ema_decay}, {ema_msg})")

    if train_sampler is not None:
        train_sampler.set_epoch(train_data_epoch)
    loader_iter = iter(loader)

    def next_train_batch():
        nonlocal loader_iter, train_data_epoch
        try:
            return next(loader_iter)
        except StopIteration:
            train_data_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_data_epoch)
            loader_iter = iter(loader)
            return next(loader_iter)

    if resume_step > 0 and master_process:
        if args.skip_resume_replay:
            print(f"Resume @ step {resume_step}: SKIPPING data fast-forward "
                  f"(fresh data iteration) — first step imminent.", flush=True)
        else:
            print(f"Resume @ step {resume_step}: fast-forwarding "
                  f"{resume_step * grad_accum_steps:,} batches to restore data position "
                  f"(GPU-idle, slow; use --skip_resume_replay to skip)...", flush=True)
    for step in range(1, max_steps + 1):
        if step <= resume_step:
            if not args.skip_resume_replay:
                # Fast-forward: consume batches without computing (restores exact data order)
                for _ in range(grad_accum_steps):
                    next_train_batch()
            continue

        loss_accum = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum_steps):
            batch = next_train_batch()
            if use_decoder_only:
                input_ids, target_ids, padding_mask, _src_lens = batch
                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)
                padding_mask = padding_mask.to(device, non_blocking=True).bool()
                if ddp_world_size > 1:
                    m.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                with autocast_ctx:
                    logits, loss = m(input_ids, targets=target_ids,
                                     key_padding_mask=padding_mask)
                # Set src/dec_in to None so the NaN-detect path below has shapes for messages
                src = input_ids
                dec_in = input_ids
            else:
                if use_xattn:
                    src, dec_in, labels, src_mask, tgt_mask, wp_class = batch
                    wp_class = wp_class.to(device, non_blocking=True).long()
                else:
                    src, dec_in, labels, src_mask, tgt_mask = batch
                    wp_class = None
                src = src.to(device, non_blocking=True)
                dec_in = dec_in.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                src_mask = src_mask.to(device, non_blocking=True).bool()
                tgt_mask = tgt_mask.to(device, non_blocking=True).bool()

                if ddp_world_size > 1:
                    m.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                with autocast_ctx:
                    logits, loss = m(
                        dec_in, src, targets=labels, src_mask=src_mask, tgt_mask=tgt_mask,
                        **({"wp_class": wp_class} if wp_class is not None else {})
                    )

            # Check for NaN/Inf in micro-batch loss
            if args.detect_nan:
                if torch.isnan(loss) or torch.isinf(loss):
                    if master_process:
                        print(f"\n❌ CRITICAL: NaN/Inf in micro-batch at step {step}, micro_step {micro_step}")
                        print(f"   src shape: {src.shape}, dec_in shape: {dec_in.shape}")
                        print(f"   loss value: {loss.item()}")
                    raise RuntimeError(f"Loss is NaN/Inf at step {step}, micro_step {micro_step}")

            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()

        if ddp_world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Check for NaN/Inf in loss
        if args.detect_nan:
            loss_val = loss_accum.cpu().item()
            if torch.isnan(loss_accum) or torch.isinf(loss_accum):
                if master_process:
                    print(f"\n❌ CRITICAL: NaN/Inf detected at step {step}! Loss={loss_val}")
                raise RuntimeError(f"Loss is NaN/Inf at step {step}")

        norm = torch.nn.utils.clip_grad_norm_(raw_m.parameters(), args.grad_clip_norm)

        # Skip step entirely if gradient norm is too large (bad batch)
        if args.skip_norm > 0 and norm.item() > args.skip_norm:
            # Log per-layer gradient norms to diagnostic file
            if master_process:
                _diag_path = checkpoint_dir / "norm_diagnostics.log"
                with open(_diag_path, "a") as _df:
                    _df.write(f"step={step} total_norm={norm.item():.2f} loss={loss_accum.item():.4f}\n")
                    # Per-layer norms (top 10 by magnitude)
                    _layer_norms = []
                    for name, p in raw_m.named_parameters():
                        if p.grad is not None:
                            _layer_norms.append((name, p.grad.norm().item()))
                    _layer_norms.sort(key=lambda x: -x[1])
                    for name, ln in _layer_norms[:10]:
                        _df.write(f"  {ln:10.3f}  {name}\n")
                    # Batch metadata
                    _df.write(f"  src_len={src.shape[1]} tgt_len={dec_in.shape[1]} batch={src.shape[0]}\n\n")
            optimizer.zero_grad()
            skip_count_epoch += 1
            skip_count_total += 1
            if master_process:
                print(f"⚠️  SKIP step {step}: grad norm {norm.item():.1f} > {args.skip_norm}")
        else:
            lr = get_lr(step, max_steps=lr_max_steps, max_lr=c.lr, warmup_frac=args.warmup_frac)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * pg.get("lr_scale", 1.0)
            optimizer.step()
            # Update EMA weights after each optimizer step
            if ema is not None:
                ema.update(raw_m)

        if (step % args.log_interval == 0 or step == resume_step + 1) and master_process:
            epoch = step / steps_per_epoch
            loss_val = loss_accum.cpu().item()
            norm_val = norm.item()
            tag = "  <- first step, training underway" if step == resume_step + 1 else ""
            print(f"step {step}/{max_steps} epoch {epoch:.1f} loss {loss_val:.3f} "
                  f"norm {norm_val:.3f} lr {lr:.2e}{tag}")
            if run_rec is not None:
                run_rec.log_train(step, loss_val, norm_val, lr)

            # Report skip count at each epoch boundary
            epoch_int = int(epoch)
            if epoch_int > current_epoch_int:
                if skip_count_epoch > 0 or skip_count_total > 0:
                    print(f"📊 Epoch {current_epoch_int} skip summary: "
                          f"{skip_count_epoch} skips this epoch, "
                          f"{skip_count_total} total")
                current_epoch_int = epoch_int
                skip_count_epoch = 0

        if eval_loader is not None and args.eval_interval > 0 and step % args.eval_interval == 0 and master_process:
            # DDP-safe: build a deep-copied shadow model on rank 0 and swap in
            # EMA weights there. The DDP-wrapped raw_m is never modified, so
            # parameter-sync invariants are preserved.
            eval_target = raw_m
            eval_shadow = None
            if ema is not None:
                eval_shadow = copy.deepcopy(raw_m).eval()
                with torch.no_grad():
                    for name, p in eval_shadow.named_parameters():
                        if name in ema.shadow:
                            p.data.copy_(ema.shadow[name])
                eval_target = eval_shadow

            e_loss = eval_loss(eval_target, eval_loader, device,
                               eval_batches, use_xattn=use_xattn, use_decoder_only=use_decoder_only)
            print(f"  eval loss {e_loss:.3f}" +
                  (f" (EMA)" if ema is not None else ""))
            if run_rec is not None:
                run_rec.log_eval(step, e_loss)

            if eval_shadow is not None:
                del eval_shadow
                torch.cuda.empty_cache()

        # --- Structural validity probe ---
        # DDP-safe: deep-copied shadow model with EMA weights swapped in.
        if (args.probe_interval > 0 and probe_sources and
                step % args.probe_interval == 0 and master_process):
            probe_target = raw_m
            probe_shadow = None
            if ema is not None:
                probe_shadow = copy.deepcopy(raw_m).eval()
                with torch.no_grad():
                    for name, p in probe_shadow.named_parameters():
                        if name in ema.shadow:
                            p.data.copy_(ema.shadow[name])
                probe_target = probe_shadow

            probe_res = structural_probe(
                probe_target, sp, probe_sources, device,
                version_id=probe_version_id,
                n=args.probe_n,
                training_formulas=probe_training_formulas,
            )

            if probe_shadow is not None:
                del probe_shadow
                torch.cuda.empty_cache()

            if probe_res is not None:
                n_total = probe_res["total"]
                gen_ok = probe_res["gen_ok"]
                tier0 = probe_res["tier0"]
                comp_novel = probe_res.get("comp_novel", 0)
                n_formulas = probe_res.get("n_formulas", 0)
                n_protos = probe_res.get("n_protos", 0)
                pair_dist = probe_res.get("pair_dist", "?")
                print(f"  probe: gen={gen_ok}/{n_total} "
                      f"tier0={tier0}/{n_total} "
                      f"bvs={probe_res['bvs']}/{n_total} "
                      f"sg={probe_res['sg']}/{n_total} "
                      f"smact={probe_res['smact']}/{n_total} "
                      f"pair={pair_dist}/{n_total} "
                      f"novel={comp_novel}/{gen_ok} "
                      f"formulas={n_formulas} protos={n_protos}"
                      + (f" (EMA)" if ema is not None else ""))

                # Log to run record
                if run_rec is not None:
                    run_rec.data.setdefault("probe_curve", []).append({
                        "step": step, **probe_res,
                    })

                # Save best-probe checkpoint
                if tier0 > best_probe_tier0:
                    best_probe_tier0 = tier0
                    best_probe_path = os.path.join(
                        checkpoint_dir, f"{args.prefix}_ckpt_best_probe.pt")
                    _bp_dict = {
                        "model": raw_m.state_dict(), "step": step,
                        "config": raw_m.config, "probe": probe_res,
                    }
                    if ema is not None:
                        _bp_dict["ema"] = ema.state_dict()
                    torch.save(_bp_dict, best_probe_path)
                    print(f"  new best probe: tier0={tier0}/{n_total} -> {best_probe_path}")

            # --- Semimetal probe: only run when fine-tuning for topology (--best_probe_metric semi)
            semi_res = None
            if args.best_probe_metric == "semi":
                semi_res = semimetal_probe(
                    raw_m, sp, device, version_id=probe_version_id,
                )
            if semi_res is not None:
                sg_ok = semi_res["correct_sg"]
                sg_total = semi_res["total"]
                sg_gen = semi_res["gen_ok"]
                detail_str = " ".join(
                    f"{d['formula']}={'OK' if d.get('status')=='correct_sg' else d.get('gen_sg','?')}"
                    for d in semi_res["details"]
                )
                print(f"  semi-probe: sg_correct={sg_ok}/{sg_total} gen={sg_gen}/{sg_total} [{detail_str}]"
                      + (f" (EMA)" if ema is not None else ""))

                if run_rec is not None:
                    run_rec.data.setdefault("semi_probe_curve", []).append({
                        "step": step, **{k: v for k, v in semi_res.items() if k != "details"},
                    })

                # Save best-probe checkpoint based on semi-probe if configured
                if args.best_probe_metric == "semi" and sg_ok > best_probe_tier0:
                    best_probe_tier0 = sg_ok  # reuse variable for semi metric
                    best_probe_path = os.path.join(
                        checkpoint_dir, f"{args.prefix}_ckpt_best_probe.pt")
                    _bp_dict = {
                        "model": raw_m.state_dict(), "step": step,
                        "config": raw_m.config, "semi_probe": semi_res,
                    }
                    if ema is not None:
                        _bp_dict["ema"] = ema.state_dict()
                    torch.save(_bp_dict, best_probe_path)
                    print(f"  new best semi-probe: sg_correct={sg_ok}/{sg_total} -> {best_probe_path}")

        # --- Conditional-generation probe (bandgap + eps_0 compliance) ---
        # DDP-safe: probe runs on a deep-copied shadow model so the live
        # DDP-wrapped raw_m is never modified. The previous
        # ema.apply_to(raw_m) / ema.restore(raw_m) approach silently
        # desynced rank-0's parameters from the other ranks (even when the
        # restore was bit-exact, in-flight CUDA streams + DDP gradient
        # bucket invariants were broken), causing loss explosions ~50
        # steps after each probe. See `feedback_cond_probe_ddp_unsafe.md`.
        if (args.cond_probe_interval > 0 and
                step % args.cond_probe_interval == 0 and master_process):
            try:
                from cond_probe import run_cond_probe, save_probe_result
                # Build a probe-only shadow model on rank 0; never visible to DDP.
                probe_model = copy.deepcopy(raw_m).eval()
                if ema is not None:
                    with torch.no_grad():
                        for name, p in probe_model.named_parameters():
                            if name in ema.shadow:
                                p.data.copy_(ema.shadow[name])
                cp_res = run_cond_probe(
                    probe_model, sp, device, version_id=probe_version_id,
                )
                cp_summary = cp_res.summary()
                print(f"  cond-probe: parse={cp_summary['parse_rate']*100:.0f}% "
                      f"bg_match={cp_summary['bg_match_rate']*100:.0f}% "
                      f"eps_match={cp_summary['eps_match_rate']*100:.0f}% "
                      + (f"(EMA)" if ema is not None else ""))
                cp_path = os.path.join(
                    checkpoint_dir,
                    f"{args.prefix}_cond_probe_step{step:06d}.json")
                save_probe_result(cp_res, cp_path)
                if run_rec is not None:
                    run_rec.data.setdefault("cond_probe_curve", []).append({
                        "step": step, **cp_summary,
                    })
                del probe_model
                torch.cuda.empty_cache()
            except Exception as _e:
                print(f"  cond-probe failed: {_e}")

        if step % args.save_interval == 0 and master_process:
            ckpt_path = os.path.join(checkpoint_dir, f"{args.prefix}_ckpt_{step:06d}.pt")
            save_dict = {
                "model": raw_m.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "loss": round(loss_accum.item(), 2), "config": raw_m.config,
            }
            if ema is not None:
                save_dict["ema"] = ema.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            if run_rec is not None:
                run_rec.log_checkpoint(step, ckpt_path)
            # Optional pruning: keep only the N most recent numbered step
            # checkpoints. best_probe/final/ema (non-numeric suffixes) are
            # never matched by this glob, so they are always preserved.
            if args.max_keep_checkpoints > 0:
                import glob as _glob
                numbered = sorted(_glob.glob(
                    os.path.join(checkpoint_dir, f"{args.prefix}_ckpt_[0-9]*.pt")))
                for _old in numbered[:-args.max_keep_checkpoints]:
                    try:
                        os.remove(_old)
                        print(f"  pruned old checkpoint: {_old}")
                    except OSError as _e:
                        print(f"  prune failed for {_old}: {_e}")

    # Save final checkpoint (skip if no training steps were executed, e.g. resume_step >= max_steps)
    if master_process and resume_step < max_steps:
        final_dict = {
            "model": raw_m.state_dict(), "optimizer": optimizer.state_dict(),
            "step": max_steps, "loss": round(loss_accum.item(), 2), "config": raw_m.config,
        }
        if ema is not None:
            final_dict["ema"] = ema.state_dict()

        # Save regular final checkpoint
        final_path = os.path.join(checkpoint_dir, f"{args.prefix}_ckpt_final.pt")
        torch.save(final_dict, final_path)
        print(f"Saved final checkpoint: {final_path}")

        # Also save EMA-only checkpoint for inference
        if ema is not None:
            ema_backup = ema.apply_to(raw_m)
            ema_dict = {
                "model": raw_m.state_dict(),
                "step": max_steps, "loss": round(loss_accum.item(), 2),
                "config": raw_m.config, "is_ema": True,
            }
            ema_path = os.path.join(checkpoint_dir, f"{args.prefix}_ckpt_ema_final.pt")
            torch.save(ema_dict, ema_path)
            ema.restore(raw_m, ema_backup)
            print(f"Saved EMA checkpoint: {ema_path}")
    elif master_process and resume_step >= max_steps:
        print(f"No training performed: resume step ({resume_step}) >= max_steps ({max_steps}). "
              f"Increase --max_steps or --epochs (with --max_steps 0) to train further.")

    # --- Finalize run record ---
    if master_process and run_rec is not None:
        run_rec.finalize(
            status="completed" if resume_step < max_steps else "no_training",
            final_checkpoint=final_path if resume_step < max_steps else None,
            final_step=max_steps,
            skip_count=skip_count_total,
        )
        run_path = run_rec.save()
        _active_run_rec = None
        print(f"Run record saved: {run_path}")

    if ddp_world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except (KeyboardInterrupt, Exception) as exc:
        if _active_run_rec is not None:
            _active_run_rec.save_interrupted(error=str(exc))
            print(f"Interrupted run saved: {_active_run_rec.run_id}")
        raise

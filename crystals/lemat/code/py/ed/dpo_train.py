"""DPO (Direct Preference Optimization) fine-tuning for crystal structure generation.

Loads a supervised checkpoint, creates a trainable policy and a frozen reference
copy, and optimizes the DPO objective on preference pairs.

Usage:
    # Single GPU
    python dpo_train.py --data dpo_pairs.csv \
        --checkpoint checkpoints/ed_ckpt_final.pt --epochs 1 --lr 1e-6 --beta 0.5

    # Multi-GPU DDP (same pattern as ed_train.py)
    torchrun --nproc_per_node=4 dpo_train.py --data dpo_pairs.csv \
        --checkpoint checkpoints/ed_ckpt_final.pt --epochs 1 


Training Log notes: 
┌─────────┬───────────────────────────────┬───────────────────────────────────────────────────┐   
│ Metric  │         What it means         │                   Warning sign                    │
├─────────┼───────────────────────────────┼───────────────────────────────────────────────────┤   
│ entropy │ Avg per-token entropy of      │ Sharp drop → model concentrating on few tokens    │
│         │ policy distribution (nats)    │ (mode collapse)                                   │
├─────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
│ drift_c │ π_chosen − ref_chosen         │ Large positive → policy diverging from reference  │
│         │ log-prob drift                │ on good outputs                                   │
├─────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
│ drift_r │ π_rejected − ref_rejected     │ Large negative → policy aggressively suppressing  │
│         │ log-prob drift                │ rejected patterns (can destabilize generation)    │
└─────────┴───────────────────────────────┴───────────────────────────────────────────────────┘

  Healthy training: entropy stays relatively stable, drift values grow slowly. Mode collapse:
  entropy drops sharply while drift_r goes very negative.


"""

import argparse
import copy
import itertools
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from configs import get_device
from dpo_data import get_dpo_loader
from ed_model import EdGPT, get_lr


# ---------------------------------------------------------------------------
# DPO core functions
# ---------------------------------------------------------------------------

def compute_seq_log_probs(model, src, dec_in, labels, src_mask, tgt_mask):
    """Forward pass -> per-sequence sum of log-probs at label positions.

    Args:
        model: EdGPT (or DDP-wrapped).
        src:      (B, S) encoder input token IDs.
        dec_in:   (B, T) decoder input token IDs.
        labels:   (B, T) target token IDs (-100 = ignore).
        src_mask: (B, S) bool mask for encoder.
        tgt_mask: (B, T) bool mask for decoder.

    Returns:
        (B,) tensor of per-sequence summed log-probabilities.
    """
    logits, _ = model(dec_in, src, src_mask=src_mask, tgt_mask=tgt_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-probs at label positions; clamp(min=0) avoids indexing with -100
    token_lp = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (token_lp * mask).sum(dim=-1)  # shape (B,), summed (not length-normalised)


def compute_entropy(model, src, dec_in, labels, src_mask, tgt_mask):
    """Mean per-token entropy of the policy distribution (nats).

    A sharp drop signals mode collapse — the model concentrates mass on
    very few tokens.  Computed only at logging steps to avoid overhead.
    """
    with torch.no_grad():
        logits, _ = model(dec_in, src, src_mask=src_mask, tgt_mask=tgt_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        mask = (labels != -100).float()
        seq_len = mask.sum(dim=-1).clamp(min=1)
        return ((token_entropy * mask).sum(dim=-1) / seq_len).mean()  # scalar


def dpo_loss(pi_chosen_lp, pi_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta):
    """IPO loss (Azar et al.) with target margin 1/(2*beta).

    Instead of pushing the margin to infinity (standard DPO sigmoid),
    use a squared loss with a finite target margin.  This prevents
    over-optimisation that can destabilise generation quality.

    L = (margin - 1/(2*beta))^2
    where margin = (pi_w - ref_w) - (pi_l - ref_l)
    """
    margin = (pi_chosen_lp - ref_chosen_lp) - (pi_rejected_lp - ref_rejected_lp)
    target = 1.0 / (2.0 * beta)
    return ((margin - target) ** 2).mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DPO fine-tuning for crystal generation")
    p.add_argument("--data", required=True, help="Preference pairs CSV")
    p.add_argument("--checkpoint", required=True, help="Supervised checkpoint to start from")
    p.add_argument("--sp_model", default="checkpoints/model_sp.model",
                   help="SentencePiece model path")
    p.add_argument("--output_dir", default="./checkpoints", help="Checkpoint directory")
    p.add_argument("--prefix", default="dpo", help="Checkpoint prefix")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--beta", type=float, default=0.5, help="DPO temperature")
    p.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=500, help="Save every N steps")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--device", default="cuda", help="Device")
    p.add_argument("--min_entropy", type=float, default=0.5,
                   help="Early-stop if policy entropy drops below this (mode collapse)")
    return p.parse_args()


def load_model(ckpt_path, device):
    """Load model from checkpoint, stripping _orig_mod. prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = EdGPT(config).to(device)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    return model, config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_rank = int(os.environ.get("RANK", 0))
    master_process = ddp_rank == 0

    # --- Device setup ---
    if ddp_world_size > 1:
        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(get_device(args.device))

    device_type = device.type

    # --- Load policy and reference models ---
    if master_process:
        print(f"Loading checkpoint: {args.checkpoint}")
    policy_model, config = load_model(args.checkpoint, device)
    ref_model, _ = load_model(args.checkpoint, device)

    # Freeze reference model
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Policy model: train mode
    policy_model.train()

    if ddp_world_size > 1:
        policy_model = DDP(policy_model, device_ids=[local_rank])
        if args.compile:
            policy_model = torch.compile(policy_model)
        raw_policy = policy_model.module
    else:
        if args.compile:
            policy_model = torch.compile(policy_model)
        raw_policy = policy_model

    # --- Optimizer ---
    optimizer = raw_policy.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, device_type=device_type
    )

    # --- Data loader ---
    loader = get_dpo_loader(
        args.data, args.sp_model, args.batch_size, args.max_seq_len, shuffle=True
    )
    batches_per_epoch = len(loader)
    max_steps = batches_per_epoch * args.epochs

    if master_process:
        n_params = sum(p.numel() for p in raw_policy.parameters())
        print(f"Policy: {n_params/1e6:.1f}M params | B:{args.batch_size}, "
              f"epochs:{args.epochs}, batches/epoch:{batches_per_epoch}, "
              f"max_steps:{max_steps}, beta:{args.beta}, lr:{args.lr}")

    # --- Precision ---
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training loop ---
    loader_iter = itertools.cycle(iter(loader))
    for step in range(1, max_steps + 1):
        batch = next(loader_iter)
        (src, chosen_dec_in, chosen_labels,
         rejected_dec_in, rejected_labels,
         src_mask, chosen_tgt_mask, rejected_tgt_mask) = [
            t.to(device) for t in batch
        ]

        # Ensure masks are bool
        src_mask = src_mask.bool()
        chosen_tgt_mask = chosen_tgt_mask.bool()
        rejected_tgt_mask = rejected_tgt_mask.bool()

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            # Policy log-probs
            pi_chosen_lp = compute_seq_log_probs(
                policy_model, src, chosen_dec_in, chosen_labels,
                src_mask, chosen_tgt_mask
            )
            pi_rejected_lp = compute_seq_log_probs(
                policy_model, src, rejected_dec_in, rejected_labels,
                src_mask, rejected_tgt_mask
            )

            # Reference log-probs (no grad)
            with torch.no_grad():
                ref_chosen_lp = compute_seq_log_probs(
                    ref_model, src, chosen_dec_in, chosen_labels,
                    src_mask, chosen_tgt_mask
                )
                ref_rejected_lp = compute_seq_log_probs(
                    ref_model, src, rejected_dec_in, rejected_labels,
                    src_mask, rejected_tgt_mask
                )

            loss = dpo_loss(
                pi_chosen_lp, pi_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                args.beta
            )

        loss.backward()

        if ddp_world_size > 1:
            loss_val = loss.detach().clone()
            dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
        else:
            loss_val = loss.detach()

        norm = torch.nn.utils.clip_grad_norm_(raw_policy.parameters(), 1.0)

        lr = get_lr(step, max_steps=max_steps, max_lr=args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        # --- Logging ---
        if step % args.log_interval == 0 and master_process:
            with torch.no_grad():
                reward_margin = (
                    (pi_chosen_lp - ref_chosen_lp) - (pi_rejected_lp - ref_rejected_lp)
                ).mean().item()
                # Per-sequence KL drift from reference (policy − ref log-probs)
                chosen_drift = (pi_chosen_lp - ref_chosen_lp).mean().item()
                rejected_drift = (pi_rejected_lp - ref_rejected_lp).mean().item()
                # Policy entropy on chosen sequences (mode-collapse indicator)
                entropy = compute_entropy(
                    policy_model, src, chosen_dec_in, chosen_labels,
                    src_mask, chosen_tgt_mask
                ).item()
            epoch = step / batches_per_epoch
            print(f"step {step}/{max_steps} epoch {epoch:.1f} "
                  f"loss {loss_val.item():.4f} margin {reward_margin:.3f} "
                  f"norm {norm.item():.3f} lr {lr:.2e} "
                  f"entropy {entropy:.2f} "
                  f"drift_c {chosen_drift:.2f} drift_r {rejected_drift:.2f}")

            # Entropy-based early stopping
            if entropy < args.min_entropy:
                print(f"EARLY STOP: entropy {entropy:.3f} < {args.min_entropy} "
                      f"(mode collapse detected)")
                # Save checkpoint before stopping
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"{args.prefix}_ckpt_earlystop_{step:06d}.pt"
                )
                torch.save(
                    {"model": raw_policy.state_dict(), "step": step,
                     "loss": round(loss_val.item(), 4),
                     "config": raw_policy.config},
                    ckpt_path,
                )
                print(f"Saved early-stop checkpoint: {ckpt_path}")
                if ddp_world_size > 1:
                    dist.destroy_process_group()
                return

        # --- Checkpointing ---
        if step % args.save_interval == 0 and master_process:
            ckpt_path = os.path.join(
                args.output_dir, f"{args.prefix}_ckpt_{step:06d}.pt"
            )
            torch.save(
                {"model": raw_policy.state_dict(), "step": step,
                 "loss": round(loss_val.item(), 4), "config": raw_policy.config},
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # --- Final checkpoint ---
    if master_process:
        final_path = os.path.join(args.output_dir, f"{args.prefix}_ckpt_final.pt")
        torch.save(
            {"model": raw_policy.state_dict(), "step": max_steps,
             "loss": round(loss_val.item(), 4), "config": raw_policy.config},
            final_path,
        )
        print(f"Saved final checkpoint: {final_path}")

    if ddp_world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())

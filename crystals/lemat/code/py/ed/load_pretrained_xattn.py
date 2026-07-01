# load_pretrained_xattn.py
"""
Warm-start ExpertXAttnEdGPT from a pretrained EdGPT checkpoint.

Strategy:
  1. Copy all weight tensors whose keys and shapes match exactly
     (embeddings, LN, encoder, self-attention, MLP, q_proj, c_proj)
  2. For expert layers (last N decoder layers): broadcast the single
     kv_proj weight into all 4 expert kv_projs
     → all experts start identical; specialisation happens during training
"""

from ed_model import EdGPT
from expert_xattn_model import ExpertXAttnEdGPT


def load_pretrained_into_xattn(
    pretrained: EdGPT,
    xattn_model: ExpertXAttnEdGPT,
) -> ExpertXAttnEdGPT:
    """
    Warm-start ExpertXAttnEdGPT from a pretrained EdGPT.

    Args:
        pretrained:  trained EdGPT instance (or state dict loaded into one)
        xattn_model: freshly initialised ExpertXAttnEdGPT instance

    Returns:
        xattn_model with weights loaded in-place
    """
    pre_state = pretrained.state_dict()
    xattn_state = xattn_model.state_dict()

    # Step 1: copy all matching keys directly.
    # This covers: embeddings, encoder, LN, self-attention, MLP,
    # and standard Block layers (0 through n_standard-1) in full.
    copied, skipped = 0, 0
    for k, v in pre_state.items():
        if k in xattn_state and xattn_state[k].shape == v.shape:
            xattn_state[k] = v.clone()
            copied += 1
        else:
            skipped += 1

    # Step 2: broadcast kv_proj into expert kv_projs for expert layers.
    n_layers = pretrained.config.layers
    n_standard = xattn_model.n_standard_layers
    n_experts = xattn_model.n_xattn_experts

    broadcast_count = 0
    for layer_idx in range(n_standard, n_layers):
        # Source: Block.attn_x is MHA with kv_proj
        for suffix in ['weight', 'bias']:
            src_key = f'transformer.h.{layer_idx}.attn_x.kv_proj.{suffix}'
            if src_key not in pre_state:
                continue
            for expert_idx in range(n_experts):
                dst_key = f'transformer.h.{layer_idx}.attn_x.kv_projs.{expert_idx}.{suffix}'
                if dst_key in xattn_state:
                    xattn_state[dst_key] = pre_state[src_key].clone()
                    broadcast_count += 1

        # Also copy q_proj and c_proj if they weren't caught by exact match
        # (they should be, but guard against key name differences)
        for proj in ['q_proj', 'c_proj']:
            for suffix in ['weight', 'bias']:
                src_key = f'transformer.h.{layer_idx}.attn_x.{proj}.{suffix}'
                dst_key = src_key  # same key in ExpertCrossAttention
                if src_key in pre_state and dst_key in xattn_state:
                    if xattn_state[dst_key].shape == pre_state[src_key].shape:
                        xattn_state[dst_key] = pre_state[src_key].clone()

    xattn_model.load_state_dict(xattn_state)
    print(f"Warm-start: copied {copied} tensors, broadcast kv_proj into "
          f"{broadcast_count} expert projections across {n_layers - n_standard} layers, "
          f"skipped {skipped} non-matching keys")
    return xattn_model

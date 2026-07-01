# load_pretrained_moe.py

from ed_model import EdGPT
from wp_moe import MoEEdGPT
from configs import Config


def load_pretrained_into_moe(
    pretrained: EdGPT,
    moe_model:  MoEEdGPT,
) -> MoEEdGPT:
    """
    Warm-start MoEEdGPT from a pretrained EdGPT checkpoint.

    Strategy:
      1. Copy all weight tensors whose keys and shapes match exactly
         (embeddings, LN, attention, encoder — everything non-MoE)
      2. Copy pretrained MLP weights into EVERY expert in every layer
         → all experts start identical; specialisation happens during fine-tuning

    Args:
        pretrained: trained EdGPT instance
        moe_model:  freshly initialised MoEEdGPT instance

    Returns:
        moe_model with weights loaded in-place
    """
    pre_state = pretrained.state_dict()
    moe_state = moe_model.state_dict()

    # Step 1: copy matching keys directly
    for k, v in pre_state.items():
        if k in moe_state and moe_state[k].shape == v.shape:
            moe_state[k] = v.clone()

    # Step 2: broadcast pretrained MLP weights into each expert
    n_layers  = pretrained.config.layers
    n_experts = moe_model.n_experts
    mlp_keys  = ['c_fc.weight', 'c_fc.bias', 'c_proj.weight', 'c_proj.bias']

    for layer_idx in range(n_layers):
        for expert_idx in range(n_experts):
            for wk in mlp_keys:
                src = f'transformer.h.{layer_idx}.mlp.{wk}'
                dst = f'transformer.h.{layer_idx}.mlp.experts.{expert_idx}.{wk}'
                if src in pre_state and dst in moe_state:
                    moe_state[dst] = pre_state[src].clone()

    moe_model.load_state_dict(moe_state)
    return moe_model


# Usage
# pretrained = EdGPT(cfg); pretrained.load_state_dict(torch.load('checkpoint.pt'))
# moe = MoEEdGPT(cfg, n_experts=4)
# moe = load_pretrained_into_moe(pretrained, moe)

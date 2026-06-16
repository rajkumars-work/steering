"""
ExpertXAttnEdGPT: encoder-decoder transformer with expert cross-attention
in the last N decoder layers.

Identical to EdGPT except:
  - Decoder layers [0, n_standard) use standard Block
  - Decoder layers [n_standard, layers) use ExpertXAttnBlock
  - forward() accepts wp_class: (B, T) tensor
  - Encoder is completely unchanged
"""

import inspect

import torch
from torch import nn
from torch.nn import functional as F

from configs import Config
from ed_model import Block, EncoderBlock, MHA, MLP
from expert_xattn import ExpertXAttnBlock, N_DOF_CLASSES


class ExpertXAttnEdGPT(nn.Module):
    """
    EdGPT with per-DOF-class KV projections in decoder cross-attention.

    Args:
        c:                      Config (identical to EdGPT config)
        n_expert_xattn_layers:  how many decoder layers (from the end) use
                                expert cross-attention (default 4)
        n_xattn_experts:        number of KV experts per layer (default 4)
        xattn_eps:              epsilon smoothing for routing (default 0.05)
    """
    def __init__(self, c: Config = Config(),
                 n_expert_xattn_layers: int = 4,
                 n_xattn_experts: int = N_DOF_CLASSES,
                 xattn_eps: float = 0.05):
        super().__init__()
        self.config = c
        self.n_expert_xattn_layers = n_expert_xattn_layers
        self.n_xattn_experts = n_xattn_experts
        self.n_standard_layers = c.layers - n_expert_xattn_layers

        # Build mixed decoder: standard layers then expert layers
        decoder_layers = []
        for i in range(c.layers):
            if i < self.n_standard_layers:
                decoder_layers.append(Block(c.edim, c.heads, c.dropout))
            else:
                decoder_layers.append(ExpertXAttnBlock(
                    c.edim, c.heads, c.dropout,
                    n_experts=n_xattn_experts, eps=xattn_eps,
                ))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.edim),
            wpe=nn.Embedding(c.tdim, c.edim),
            ete=nn.Embedding(c.vocab_size, c.edim),
            epe=nn.Embedding(c.tdim, c.edim),
            h=nn.ModuleList(decoder_layers),
            e=nn.ModuleList([
                EncoderBlock(c.edim, c.heads, c.dropout)
                for _ in range(c.layers)
            ]),
            ln_f=nn.LayerNorm(c.edim),
            ln_e_f=nn.LayerNorm(c.edim),
        ))
        self.lm_head = nn.Linear(c.edim, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def encode(self, o, src_mask=None):
        """Pre-compute encoder output."""
        B, T = o.shape
        pos = torch.arange(0, T, dtype=torch.long, device=o.device)
        o = self.transformer.ete(o) + self.transformer.epe(pos)
        for enc_block in self.transformer.e:
            o = enc_block(o, src_mask)
        return self.transformer.ln_e_f(o)

    def forward(self, x, o=None, targets=None, src_mask=None, tgt_mask=None,
                encoder_out=None, wp_class=None):
        """
        Args:
            x:           (B, T)     decoder input token ids
            o:           (B, S)     encoder input token ids
            targets:     (B, T)     target token ids for CE loss
            src_mask:    (B, S)     encoder padding mask
            tgt_mask:    (B, T)     decoder padding mask
            encoder_out: (B, S, C)  pre-computed encoder output (skips encode())
            wp_class:    (B, T)     DOF class per decoder token, dtype=long
                                    Values: 0=FIXED, 1=LINE, 2=PLANE, 3=GENERAL
                                    Non-atom tokens → 0. Can be None (uniform fallback).
        """
        if encoder_out is not None:
            o = encoder_out
        elif o is not None:
            o = self.encode(o, src_mask)

        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)

        for i, block in enumerate(self.transformer.h):
            if i < self.n_standard_layers:
                x = block(x, o, src_mask, tgt_mask)
            else:
                x = block(x, o, wp_class=wp_class, src_mask=src_mask,
                          tgt_mask=tgt_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = (
            F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if targets is not None else None
        )
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95),
            eps=1e-8, fused=use_fused,
        )

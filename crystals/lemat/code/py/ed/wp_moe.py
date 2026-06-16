"""
Wyckoff-conditioned Mixture of Experts extension for EdGPT.

Drop-in replacement components:
  MoEBlock   replaces   Block
  MoEEdGPT   replaces   EdGPT
"""

import math
from enum import IntEnum
import torch
from torch import nn
from torch.nn import functional as F
from configs import Config
from ed_model import MHA, MLP, EncoderBlock   # reuse unchanged components


# ─────────────────────────────────────────────
# 1. DOF Class Definitions & WP Letter Mapping
# ─────────────────────────────────────────────

class WPDOFClass(IntEnum):
    FIXED   = 0   # 0 DOF — coordinates fully determined by symmetry
    LINE    = 1   # 1 DOF — one free parameter
    PLANE   = 2   # 2 DOF — two free parameters
    GENERAL = 3   # 3 DOF — fully general position


# Approximate mapping; override with pymatgen for production
_WP_TO_DOF: dict[str, WPDOFClass] = {
    'a': WPDOFClass.FIXED,   'b': WPDOFClass.FIXED,
    'c': WPDOFClass.FIXED,   'd': WPDOFClass.FIXED,
    'e': WPDOFClass.LINE,    'f': WPDOFClass.LINE,
    'g': WPDOFClass.LINE,    'h': WPDOFClass.LINE,
    'i': WPDOFClass.PLANE,   'j': WPDOFClass.PLANE,
    'k': WPDOFClass.PLANE,   'l': WPDOFClass.PLANE,
}

def wp_letter_to_dof(letter: str) -> int:
    """Map a WP letter to its DOF class index. Defaults to GENERAL."""
    return int(_WP_TO_DOF.get(letter.lower(), WPDOFClass.GENERAL))


# ─────────────────────────────────────────────
# 2. Wyckoff Router
# ─────────────────────────────────────────────

class WyckoffRouter(nn.Module):
    """
    Soft router conditioned on token representation + WP DOF class embedding.

    The WP embedding shifts the gate input so that tokens sharing a DOF class
    consistently favour the same expert, while the token repr allows fine-grained
    per-position adjustments within that class.

    Args:
        edim:      model embedding dimension
        n_experts: number of expert MLPs to route over
    """
    def __init__(self, edim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.wp_embed  = nn.Embedding(4, edim)          # 4 DOF classes
        self.gate      = nn.Linear(edim, n_experts, bias=False)
        self.log_temp  = nn.Parameter(torch.zeros(1))  # learnable softmax temperature

    def forward(self, x: torch.Tensor, wp_class: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (B, T, C)  token representations (after LayerNorm)
            wp_class: (B, T)     WP DOF class index per position, dtype=torch.long
        Returns:
            weights:  (B, T, n_experts)  soft expert weights, sum to 1
        """
        wp_e     = self.wp_embed(wp_class)                        # (B, T, C)
        gate_in  = x + wp_e                                       # additive conditioning
        temp     = self.log_temp.exp().clamp(min=0.1, max=10.0)
        weights  = F.softmax(self.gate(gate_in) / temp, dim=-1)   # (B, T, n_experts)
        return weights


# ─────────────────────────────────────────────
# 3. MoE MLP
# ─────────────────────────────────────────────

class MoEMLP(nn.Module):
    """
    Mixture-of-Experts FFN with Wyckoff-conditioned soft routing.

    All experts always produce an output; they are mixed by learned weights.
    Soft routing is preferred over hard (top-1) routing here because:
      - Stable gradients flow to all experts during warm-start fine-tuning
      - Handles ambiguous/transitional WP assignments gracefully
      - No need for auxiliary load-balancing loss (we have explicit labels)

    Args:
        edim:      model embedding dimension
        n_experts: number of parallel expert MLPs (default 4, one per DOF class)
    """
    def __init__(self, edim: int, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([MLP(edim) for _ in range(n_experts)])
        self.router  = WyckoffRouter(edim, n_experts)

    def forward(self, x: torch.Tensor, wp_class: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (B, T, C)
            wp_class: (B, T) long — DOF class per token. If None, averages all experts.
        Returns:
            (B, T, C)
        """
        if wp_class is None:
            # Uniform fallback for inference without WP labels
            return torch.stack([e(x) for e in self.experts], dim=0).mean(0)

        weights      = self.router(x, wp_class)          # (B, T, n_experts)
        expert_outs  = torch.stack(
            [e(x) for e in self.experts], dim=-1
        )                                                 # (B, T, C, n_experts)
        out = (expert_outs * weights.unsqueeze(-2)).sum(-1)  # (B, T, C)
        return out


# ─────────────────────────────────────────────
# 4. MoE Decoder Block
# ─────────────────────────────────────────────

class MoEBlock(nn.Module):
    """
    Decoder block with MoEMLP replacing standard MLP.
    Structurally identical to Block in ed_model.py; only the FFN differs.
    """
    def __init__(self, edim: int, heads: int, dropout: float = 0.1, n_experts: int = 4):
        super().__init__()
        self.ln_1        = nn.LayerNorm(edim)
        self.attn        = MHA(edim, heads, dropout=dropout)
        self.attn_x      = MHA(edim, heads, dropout=dropout, causal=False, is_cross_attn=True)
        self.ln_3        = nn.LayerNorm(edim)
        self.ln_2        = nn.LayerNorm(edim)
        self.mlp         = MoEMLP(edim, n_experts=n_experts)   # ← MoE replaces MLP
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, o, wp_class=None, src_mask=None, tgt_mask=None):
        x = x + self.resid_dropout(
            self.attn(self.ln_1(x), key_padding_mask=tgt_mask)
        )
        if o is not None:
            x = x + self.resid_dropout(
                self.attn_x(self.ln_3(x), o, key_padding_mask=src_mask)
            )
        x = x + self.resid_dropout(self.mlp(self.ln_2(x), wp_class))
        return x


# ─────────────────────────────────────────────
# 5. Full MoE Encoder-Decoder Model
# ─────────────────────────────────────────────

class MoEEdGPT(nn.Module):
    """
    EdGPT with Wyckoff-conditioned MoE decoder.

    Key differences from EdGPT:
      - Decoder uses MoEBlock (MoE FFN) instead of Block
      - forward() and generate() accept wp_class: (B, T) tensor
      - Encoder is completely unchanged

    Args:
        c:         Config (same as EdGPT)
        n_experts: number of expert MLPs per block (default 4)
    """
    def __init__(self, c: Config = Config(), n_experts: int = 4):
        super().__init__()
        self.config    = c
        self.n_experts = n_experts

        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(c.vocab_size, c.edim),
            wpe   = nn.Embedding(c.tdim, c.edim),
            ete   = nn.Embedding(c.vocab_size, c.edim),
            epe   = nn.Embedding(c.tdim, c.edim),
            h     = nn.ModuleList([
                MoEBlock(c.edim, c.heads, c.dropout, n_experts=n_experts)
                for _ in range(c.layers)
            ]),
            e     = nn.ModuleList([
                EncoderBlock(c.edim, c.heads, c.dropout)
                for _ in range(c.layers)
            ]),
            ln_f   = nn.LayerNorm(c.edim),
            ln_e_f = nn.LayerNorm(c.edim),
        ))
        self.lm_head = nn.Linear(c.edim, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def encode(self, o, src_mask=None):
        B, T = o.shape
        pos  = torch.arange(0, T, dtype=torch.long, device=o.device)
        o    = self.transformer.ete(o) + self.transformer.epe(pos)
        for enc_block in self.transformer.e:
            o = enc_block(o, src_mask)
        return self.transformer.ln_e_f(o)

    def forward(self, x, o=None, targets=None, src_mask=None, tgt_mask=None,
                encoder_out=None, wp_class=None):
        """
        Args:
            x:           (B, T)     decoder input token ids
            o:           (B, S)     encoder input token ids
            targets:     (B, T)     targets for CE loss
            src_mask:    (B, S)     encoder padding mask
            tgt_mask:    (B, T)     decoder padding mask
            encoder_out: (B, S, C)  pre-computed encoder output (skips encode())
            wp_class:    (B, T)     WP DOF class index per decoder position
                                    dtype=torch.long, values in {0,1,2,3}
                                    Tokens outside atom blocks → 0 (FIXED, safe default)
        """
        if encoder_out is not None:
            o = encoder_out
        elif o is not None:
            o = self.encode(o, src_mask)

        B, T = x.shape
        pos  = torch.arange(0, T, dtype=torch.long, device=x.device)
        x    = self.transformer.wte(x) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x, o, wp_class=wp_class, src_mask=src_mask, tgt_mask=tgt_mask)

        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss   = (
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
        import inspect
        param_dict    = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups  = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

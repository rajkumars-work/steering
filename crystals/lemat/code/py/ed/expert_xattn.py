"""
Expert Cross-Attention: per-DOF-class KV projections for decoder cross-attention.

Components:
  ExpertCrossAttention  — replaces MHA(is_cross_attn=True) in decoder blocks
  ExpertXAttnBlock      — decoder block using ExpertCrossAttention
"""

import torch
from torch import nn
from torch.nn import functional as F

from ed_model import MHA, MLP

N_DOF_CLASSES = 4


class ExpertCrossAttention(nn.Module):
    """
    Cross-attention with per-DOF-class KV projections.

    Each DOF class gets its own Linear(edim, 2*edim) for K,V from encoder output.
    Q projection and output projection are shared across all DOF classes.
    Hard routing with epsilon smoothing selects the dominant KV projection per
    decoder position based on wp_class.

    Args:
        edim:       model embedding dimension
        heads:      number of attention heads
        n_experts:  number of KV expert projections (default 4, one per DOF class)
        eps:        epsilon smoothing for non-selected experts (default 0.05)
        dropout:    attention dropout probability
    """
    def __init__(self, edim: int, heads: int, n_experts: int = N_DOF_CLASSES,
                 eps: float = 0.05, dropout: float = 0.1):
        super().__init__()
        assert edim % heads == 0
        self.n_head = heads
        self.n_embd = edim
        self.head_dim = edim // heads
        self.n_experts = n_experts
        self.eps = eps
        self.attn_dropout = dropout

        # Shared Q projection (from decoder hidden states)
        self.q_proj = nn.Linear(edim, edim)
        # Per-expert KV projections (from encoder output)
        self.kv_projs = nn.ModuleList([
            nn.Linear(edim, 2 * edim) for _ in range(n_experts)
        ])
        # Shared output projection
        self.c_proj = nn.Linear(edim, edim)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                wp_class: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:               (B, T, C)  decoder hidden states (after LayerNorm)
            enc_out:         (B, S, C)  encoder output
            wp_class:        (B, T)     DOF class per decoder position, long tensor
                             If None, uniform weighting across all experts.
            key_padding_mask:(B, S)     encoder padding mask (True = valid)
        Returns:
            (B, T, C)
        """
        B, T, C = x.shape
        S = enc_out.size(1)
        H, Hd = self.n_head, self.head_dim

        # Shared Q: (B, H, T, Hd)
        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)

        # Compute K, V for each expert: list of (B, H, S, Hd) pairs
        expert_k = []
        expert_v = []
        for proj in self.kv_projs:
            kv = proj(enc_out)                              # (B, S, 2C)
            k, v = kv.split(C, dim=-1)
            expert_k.append(k.view(B, S, H, Hd).transpose(1, 2))
            expert_v.append(v.view(B, S, H, Hd).transpose(1, 2))

        # Prepare attention mask
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, S)

        drop_p = self.attn_dropout if self.training else 0.0

        # Compute attention output per expert: list of (B, H, T, Hd)
        expert_outs = []
        for k, v in zip(expert_k, expert_v):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=drop_p,
            )
            expert_outs.append(y)

        # Stack: (B, H, T, Hd, n_experts)
        stacked = torch.stack(expert_outs, dim=-1)

        # Build routing weights: (B, T, n_experts) -> (B, 1, T, 1, n_experts)
        if wp_class is not None:
            hard = F.one_hot(wp_class, self.n_experts).float()       # (B, T, n_experts)
            uniform = torch.ones_like(hard) / self.n_experts
            weights = (1 - self.eps) * hard + self.eps * uniform
        else:
            weights = torch.ones(B, T, self.n_experts, device=x.device) / self.n_experts

        weights = weights[:, None, :, None, :]                      # (B, 1, T, 1, n_experts)

        # Weighted sum over experts
        y = (stacked * weights).sum(dim=-1)                          # (B, H, T, Hd)

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class ExpertXAttnBlock(nn.Module):
    """
    Decoder block with ExpertCrossAttention replacing the standard cross-attention.
    Self-attention and MLP are unchanged (standard shared weights).

    Drop-in replacement for Block in the last N decoder layers.
    """
    def __init__(self, edim: int, heads: int, dropout: float = 0.1,
                 n_experts: int = N_DOF_CLASSES, eps: float = 0.05):
        super().__init__()
        self.ln_1 = nn.LayerNorm(edim)
        self.attn = MHA(edim, heads, dropout=dropout)           # self-attention (unchanged)
        self.ln_3 = nn.LayerNorm(edim)
        self.attn_x = ExpertCrossAttention(                     # expert cross-attention
            edim, heads, n_experts=n_experts, eps=eps, dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(edim)
        self.mlp = MLP(edim)                                    # standard MLP (unchanged)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, o, wp_class=None, src_mask=None, tgt_mask=None):
        """
        Args:
            x:        (B, T, C)  decoder hidden states
            o:        (B, S, C)  encoder output (or None)
            wp_class: (B, T)     DOF class per decoder token
            src_mask: (B, S)     encoder padding mask
            tgt_mask: (B, T)     decoder padding mask
        """
        x = x + self.resid_dropout(
            self.attn(self.ln_1(x), key_padding_mask=tgt_mask)
        )
        if o is not None:
            x = x + self.resid_dropout(
                self.attn_x(self.ln_3(x), o, wp_class=wp_class,
                            key_padding_mask=src_mask)
            )
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x

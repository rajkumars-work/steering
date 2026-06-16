"""Decoder-only transformer model.

Architectural counterpart to the encoder-decoder `EdGPT` in `ed_model.py`.
Same building blocks (`MHA`, `MLP`) but with no encoder and no cross-
attention — input is the concatenation of source tokens and target tokens
in a single causal stack, prefix-LM style.

Loss is computed over target positions only (source positions in the
labels tensor are filled with -100, which `F.cross_entropy` ignores by
default).

Created for the encoder-decoder ablation experiment described in
`dielectric/docs/DecoderOnlyAblationPlan.md`. Goal: test whether RECAST's
LeMat-GenBench numbers depend on the encoder-decoder split (cross-attention)
or whether a simpler decoder-only architecture achieves the same results.
"""

import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

from configs import Config
# Reuse the building blocks from the encoder-decoder model.
from ed_model import MHA, MLP


class DBlock(nn.Module):
    """Decoder-only block: causal self-attention + MLP. No cross-attention.

    QK-norm on self-attn: encoder-decoder models inherit stabilization from
    the cross-attention pathway; a pure causal stack lacks that and is
    prone to attention-logit explosion under DDP. Required.
    """

    def __init__(self, edim, heads, dropout: float = 0.1, qk_norm: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(edim)
        self.attn = MHA(edim, heads, dropout=dropout, causal=True, qk_norm=qk_norm)
        self.ln_2 = nn.LayerNorm(edim)
        self.mlp = MLP(edim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.resid_dropout(
            self.attn(self.ln_1(x), key_padding_mask=key_padding_mask)
        )
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class DecoderOnlyGPT(nn.Module):
    """Causal decoder-only transformer, prefix-LM training.

    Input layout per row: `[src_tokens, BOS, tgt_tokens]` (concatenated
    along the time axis). Loss only on positions where the corresponding
    label is not -100 (i.e., the target portion).
    """

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.config = c

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(c.vocab_size, c.edim),
                wpe=nn.Embedding(c.tdim, c.edim),
                h=nn.ModuleList(
                    [DBlock(c.edim, c.heads, c.dropout) for _ in range(c.layers)]
                ),
                ln_f=nn.LayerNorm(c.edim),
            )
        )
        self.lm_head = nn.Linear(c.edim, c.vocab_size, bias=False)
        # Tie input embedding with output head, same as EdGPT.
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, x, targets=None, key_padding_mask=None):
        """
        Args:
            x: (B, T) concatenated [src; BOS; tgt] token ids.
            targets: (B, T) labels — -100 for source positions, real ids
                     for target positions (the label_ids[i] expected at
                     each target output position).
            key_padding_mask: (B, T) bool, True at valid (non-pad) positions.

        Returns:
            (logits, loss).
        """
        B, T = x.shape
        if T > self.config.tdim:
            raise ValueError(
                f"Sequence length {T} exceeds tdim={self.config.tdim}. "
                f"Either truncate inputs or raise tdim."
            )
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pe = self.transformer.wpe(pos)
        te = self.transformer.wte(x)
        h = te + pe

        for block in self.transformer.h:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.transformer.ln_f(h)
        # Force fp32 lm_head + cross_entropy in a single autocast-disabled block.
        # Tied wte<->lm_head + DDP large-effective-batch can grow lm_head outputs
        # past bf16 safe range; computing the entire path (matmul, clamp, CE) in
        # fp32 with hard clamp guarantees bounded, well-conditioned loss.
        loss = None
        with torch.amp.autocast(device_type="cuda", enabled=False):
            logits = self.lm_head(h.float())
            logits = torch.clamp(logits, min=-30.0, max=30.0)
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
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
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95),
            eps=1e-8, fused=use_fused,
        )
        return optimizer

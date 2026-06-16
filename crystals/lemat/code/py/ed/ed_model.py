"""Encoder-decoder transformer model.

See ARCHITECTURE.md for detailed documentation of v2 features
(QK-norm, gated cross-attention, encoder LR scaling).
"""

import inspect
import math
import torch
from torch import nn
from torch.nn import functional as F
from configs import Config


def get_lr(it, *, max_steps: int, max_lr: float, warmup_frac: float = 0.05):
    min_lr = max_lr * 0.1
    warmup_steps = max(1, int(max_steps * warmup_frac))
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class EdGPT(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.config = c

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(c.vocab_size, c.edim),
                wpe=nn.Embedding(c.tdim, c.edim),
                ete=nn.Embedding(c.vocab_size, c.edim),
                epe=nn.Embedding(c.tdim, c.edim),
                h=nn.ModuleList(
                    [Block(c.edim, c.heads, c.dropout,
                           qk_norm=getattr(c, 'qk_norm', False),
                           gated_cross_attn=getattr(c, 'gated_cross_attn', False))
                     for _ in range(getattr(c, 'dec_layers', None) or c.layers)]
                ),
                e=nn.ModuleList(
                    [EncoderBlock(c.edim, c.heads, c.dropout)
                     for _ in range(getattr(c, 'enc_layers', None) or c.layers)]
                ),
                ln_f=nn.LayerNorm(c.edim),
                ln_e_f=nn.LayerNorm(c.edim),
            )
        )
        self.lm_head = nn.Linear(c.edim, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def encode(self, o, src_mask=None):
        """Pre-compute encoder output for use in generation loops."""
        B, T = o.shape
        pos = torch.arange(0, T, dtype=torch.long, device=o.device)
        pe = self.transformer.epe(pos)
        te = self.transformer.ete(o)
        o = te + pe
        for encoder_block in self.transformer.e:
            o = encoder_block(o, src_mask)
        o = self.transformer.ln_e_f(o)
        return o

    def forward(
        self, x, o=None, targets=None, src_mask=None, tgt_mask=None, encoder_out=None
    ):
        if encoder_out is not None:
            o = encoder_out
        elif o is not None:
            o = self.encode(o, src_mask)

        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pe = self.transformer.wpe(pos)
        te = self.transformer.wte(x)
        x = te + pe

        for block in self.transformer.h:
            x = block(x, o, src_mask, tgt_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = (
            F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if targets is not None
            else None
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
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


class Block(nn.Module):
    def __init__(self, edim, heads, dropout: float = 0.1,
                 qk_norm: bool = False, gated_cross_attn: bool = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(edim)
        self.attn = MHA(edim, heads, dropout=dropout)
        self.attn_x = MHA(edim, heads, dropout=dropout, causal=False,
                          is_cross_attn=True, qk_norm=qk_norm)
        self.ln_3 = nn.LayerNorm(edim)
        self.ln_2 = nn.LayerNorm(edim)
        self.mlp = MLP(edim)
        self.resid_dropout = nn.Dropout(dropout)
        # Gated cross-attention: learnable gate initialized to 0 (Flamingo-style).
        # Model gradually learns to use cross-attention rather than being forced to from step 0.
        self.gated_cross_attn = gated_cross_attn
        if gated_cross_attn:
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, o, src_mask=None, tgt_mask=None):
        x = x + self.resid_dropout(self.attn(self.ln_1(x), key_padding_mask=tgt_mask))
        if o is not None:
            cross_out = self.resid_dropout(
                self.attn_x(self.ln_3(x), o, key_padding_mask=src_mask)
            )
            if self.gated_cross_attn:
                cross_out = cross_out * self.cross_attn_gate.tanh()
            x = x + cross_out
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, edim, heads, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(edim)
        self.attn = MHA(edim, heads, dropout=dropout, causal=False)
        self.ln_2 = nn.LayerNorm(edim)
        self.mlp = MLP(edim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = x + self.resid_dropout(self.attn(self.ln_1(x), key_padding_mask=src_mask))
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class MHA(nn.Module):
    def __init__(
        self,
        edim,
        heads,
        causal=True,
        dropout=0.1,
        is_cross_attn: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert edim % heads == 0, "embedding dimension must be divisible by head count"
        self.is_cross_attn = is_cross_attn
        self.n_head = heads
        self.n_embd = edim
        self.causal = causal
        self.attn_dropout = dropout
        self.qk_norm = qk_norm
        if is_cross_attn:
            self.q_proj = nn.Linear(edim, edim)
            self.kv_proj = nn.Linear(edim, 2 * edim)
        else:
            self.c_attn = nn.Linear(edim, 3 * edim)
        self.c_proj = nn.Linear(edim, edim)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # QK-norm: L2-normalize Q and K per head to prevent attention logit explosion
        if qk_norm:
            self.q_norm = nn.LayerNorm(edim // heads)
            self.k_norm = nn.LayerNorm(edim // heads)

    def forward(self, x, o=None, key_padding_mask=None):
        B, T, C = x.size()

        if self.is_cross_attn and o is not None:
            T_enc = o.size(1)
            q = self.q_proj(x)
            kv = self.kv_proj(o)
            k, v = kv.split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, None, None, :]
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=key_padding_mask,
                is_causal=False,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
        else:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if self.causal and key_padding_mask is not None:
                causal_mask = torch.tril(
                    torch.ones(T, T, dtype=torch.bool, device=x.device)
                )
                key_padding_mask = key_padding_mask[:, None, None, :]
                combined_mask = causal_mask[None, None, :, :] & key_padding_mask
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined_mask,
                    is_causal=False,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )
            elif key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, None, None, :]
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=key_padding_mask,
                    is_causal=False,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=self.causal,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, edim):
        super().__init__()
        self.c_fc = nn.Linear(edim, 4 * edim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * edim, edim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

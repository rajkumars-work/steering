import torch
from dataclasses import dataclass


@dataclass
class Config:
    name: str = "default"
    edim: int = 1024
    tdim: int = 2048
    vocab_size: int = 50304
    layers: int = 12
    heads: int = 16
    dropout: float = 0.1
    gpt3: bool = True
    B: int = 16
    lr: float = 6e-4 * 3
    params: int = 1e8
    grad_checkpoint: bool = False
    # v2 architecture features. See ARCHITECTURE.md for detailed explanation.
    qk_norm: bool = True            # L2-normalize Q and K in cross-attention (default on — no downside)
    gated_cross_attn: bool = False  # learnable gate on cross-attention (init=0)
    # Asymmetric encoder/decoder depth for ed-only. None ⇒ fall back to `layers`
    # (so existing configs keep symmetric 12+12). Set to specific ints (e.g.
    # enc_layers=6, dec_layers=18) for the encoder-light ablation.
    enc_layers: int | None = None
    dec_layers: int | None = None


# RECAST-Lite (ed-lite): asymmetric encoder-decoder with most params in the
# decoder, matching the production architecture as of 2026-05-06.
# See `dielectric/docs/paper/ArchitectureStudy.md` for the ablation that
# motivated this choice. The previous symmetric 12+12 split is preserved
# below as `config_ed_v1` for reproducing pre-2026-05-06 trainings.
config_ed = Config(
    name="ed",
    edim=768,
    layers=18,            # used as decoder-default fallback if dec_layers unset; also the residual-init scaling factor
    enc_layers=6,
    dec_layers=18,
    heads=12,
    tdim=1024,
    dropout=0.1,
    vocab_size=16000,
    gpt3=True,
    B=64,
    lr=5e-5,
    grad_checkpoint=False,
)


# Legacy symmetric 12+12 ed config — kept for reproducing pre-RECAST-Lite
# training runs. New code should prefer `config_ed`.
config_ed_v1 = Config(
    name="ed",
    edim=768,
    layers=12,
    heads=12,
    tdim=1024,
    dropout=0.1,
    vocab_size=16000,
    gpt3=True,
    B=4,
    lr=1e-4,
    grad_checkpoint=False,
)


def get_device(device=None):
    if device and device != "auto":
        print("Using device", device)
        return device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device", device)
    return device

#!/usr/bin/env bash
# Fetch the public models for Tier-2 regeneration into ./models (CLIP/VAE go to the HF cache).
set -e
D="$(cd "$(dirname "$0")/.." && pwd)/models"; mkdir -p "$D"
wget -c -O "$D/DiT-XL-2-256x256.pt" https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt   # fallback ckpt
wget -O "$D/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth" \
  "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["openai/clip-vit-large-patch14","stabilityai/sd-vae-ft-mse","facebook/DiT-XL-2-256"]:
    snapshot_download(r); print("cached", r)
PY
echo "models ready in $D (+ HF cache)"

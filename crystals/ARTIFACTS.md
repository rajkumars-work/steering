# Artifacts & integrity manifest

What the crystal track depends on, where the durable copy lives, and the hash to
verify against. Nothing here is committed to the repo (weights and reference data
are fetched); this file is how you confirm you fetched the right thing.

## The trained model (the one irreplaceable artifact)

| file | size | sha256 |
|---|---|---|
| `ed_ckpt_final.pt` (CUES, alex_nolemat_lowhull) | 909 MB | `9b8e4b2e9b5dbaaff6e618acffcd6761547d6568328b2787a4052165d8554177` |

- **Canonical home:** HuggingFace model hub — `rajkumars47/cues-alex-nolemat-lowhull`
  (uploaded with a write token; `fetch_checkpoint.sh` pulls it). Verify after fetch:
  `sha256sum ed_ckpt_final.pt` must match the value above.
- Tokenizer (committed, small): `model_sp.model` (491 KB), `model_sp.vocab` (230 KB).

Verify a fetched checkpoint:
```bash
echo "9b8e4b2e9b5dbaaff6e618acffcd6761547d6568328b2787a4052165d8554177  ed_ckpt_final.pt" | sha256sum -c
```

## Reference data — already durable on HF, pin the revision

Re-fetchable; no need to re-host. Pin these revisions so the hull/novelty scoring
matches what we ran:

| dataset | revision |
|---|---|
| `LeMaterial/LeMat-Bulk` | `0dc17eea904b860ad7288141e9870f67f8e6bb2c` |
| `LeMaterial/LeMat-Bulk-MLIP-Hull` | `70d505bb294c16658a9551e2b81ace69a28d9790` |

## Public models — re-fetchable, pinned

| model | revision | used by |
|---|---|---|
| `facebook/DiT-XL-2-256` | `eab87f77abd5aef071a632f08807fbaab0b704d0` | images |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` | images (aesthetic, concepts) |
| `stabilityai/sd-vae-ft-mse` | `31f26fdeee1355a5c34592e401dd41e45d25a493` | images (DiT decode) |
| `facebook/UMA` | `7210de6fe86ad94854b21b881fefbcfdfeab373b` | crystals (MLIP scoring) |

Also public, not in the HF hub cache: ResNet-50 (torchvision weights `IMAGENET1K_V2`,
the image key π) and the LAION improved-aesthetic-predictor MLP (`scripts/download_models.sh`).

## Durability status (as of this snapshot)
- Checkpoint: volatile `/opt/dlami/nvme/...` **+ persistent backup** `~/durable/cues_alex_nolemat_lowhull/` **+ HF (pending upload)**.
- Everything else: re-fetchable from HF / torchvision at the pinned revisions above.

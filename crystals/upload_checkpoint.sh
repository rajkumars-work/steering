#!/usr/bin/env bash
# Maintainer-only: publish the CUES steering checkpoint to the HF model hub.
# Counterpart of fetch_checkpoint.sh (which readers run). Run this once, from a
# machine that has the 909 MB checkpoint and an HF *write* token.
#
#   export HF_TOKEN=$(cat ~/.ssh/hf_july.tkn)   # a WRITE token
#   bash crystals/upload_checkpoint.sh [/path/to/ed_ckpt_final.pt]
#
# Default source is the durable backup on this box; pass a path to override.
# Uses the huggingface_hub Python API (stable across CLI renames).
set -euo pipefail

HF_MODEL="${HF_MODEL:-rajkumars47/cues-alex-nolemat-lowhull}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$HOME/durable/cues_alex_nolemat_lowhull/ed_ckpt_final.pt}"
SHA_EXPECT="9b8e4b2e9b5dbaaff6e618acffcd6761547d6568328b2787a4052165d8554177"

[ -f "$SRC" ] || { echo "checkpoint not found: $SRC" >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN to a WRITE token first (export HF_TOKEN=\$(cat ~/.ssh/hf_july.tkn))" >&2; exit 1; }

echo "verifying sha256 of $SRC ..."
echo "$SHA_EXPECT  $SRC" | sha256sum -c - || {
  echo "sha256 mismatch — refusing to upload the wrong checkpoint" >&2; exit 1; }

echo "creating repo $HF_MODEL (if absent) and uploading ..."
CKPT="$SRC" TOK="$HERE/checkpoints/alex_nolemat_lowhull" CARD="$HERE/MODEL_CARD.md" \
HF_MODEL="$HF_MODEL" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_MODEL"]
api.create_repo(repo, repo_type="model", exist_ok=True)   # public by default
uploads = [
    (os.environ["CKPT"], "ed_ckpt_final.pt"),                          # the irreplaceable artifact
    (os.path.join(os.environ["TOK"], "model_sp.model"), "model_sp.model"),
    (os.path.join(os.environ["TOK"], "model_sp.vocab"), "model_sp.vocab"),
    (os.environ["CARD"], "README.md"),                                 # self-describing HF page
]
for src, dst in uploads:
    print(f"  uploading {dst} ...", flush=True)
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
print("done")
PY

echo "done -> https://huggingface.co/$HF_MODEL"
echo "readers now run: bash crystals/fetch_checkpoint.sh"

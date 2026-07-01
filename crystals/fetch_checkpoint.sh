#!/usr/bin/env bash
# Fetch the CUES steering checkpoint from the HF model hub into checkpoints/alex_nolemat_lowhull/.
# TODO: set HF_MODEL once the weights are uploaded (huggingface-cli upload by the maintainer).
set -e
HF_MODEL="${HF_MODEL:-rajkumars47/cues-alex-nolemat-lowhull}"   # placeholder repo id
DST="$(cd "$(dirname "$0")" && pwd)/checkpoints/alex_nolemat_lowhull"; mkdir -p "$DST"
python -c "from huggingface_hub import snapshot_download as s; s('$HF_MODEL', local_dir='$DST')"
echo "checkpoint -> $DST"

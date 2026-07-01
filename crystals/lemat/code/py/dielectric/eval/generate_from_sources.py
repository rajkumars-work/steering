#!/usr/bin/env python3
"""Generate structures from a text file of source strings.

Outputs generated.extxyz which can be fed to run_screen.py --input.

Usage:
    python eval/generate_from_sources.py \
        --sources eval/synthetic_sources_chons_lowk.txt \
        --ckpt sweep_novelty/v3_elements_only \
        --version d5_elements_only_causal_tgt \
        --outdir eval/chons_lowk_v3
"""
import argparse
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGHUP, signal.SIG_IGN)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ED_ROOT = _PROJECT_ROOT.parent / "ed"
for p in (str(_PROJECT_ROOT), str(_ED_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    p = argparse.ArgumentParser(description="Generate structures from synthetic source strings")
    p.add_argument("--sources", required=True, help="Text file with one source string per line")
    p.add_argument("--ckpt", required=True, help="Checkpoint dir")
    p.add_argument("--version", default="d5_elements_only_causal_tgt", help="Version ID")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--top-k", type=int, default=10, help="Top-k sampling (default: 10)")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    import os
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu != 0 and _cvd != str(args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.execvp(sys.executable, [sys.executable] + sys.argv)

    import torch
    from ase.io import write as ase_write
    from eval.screening import load_model, generate_one

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load sources
    with open(args.sources) as f:
        sources = [(line.strip(), "mp") for line in f if line.strip()]
    print(f"Loaded {len(sources)} source strings", flush=True)

    # Load model
    model, sp = load_model(args.ckpt, device)

    # Generate
    generated = []
    t0 = time.time()
    for i, (src, origin) in enumerate(sources):
        g = generate_one(model, sp, src, origin, args.version, device, top_k=args.top_k)
        if g.atoms is not None:
            g.atoms.info["synthetic_source"] = True
            generated.append(g.atoms)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sources)}] ok={len(generated)}", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"Generated: {len(generated)}/{len(sources)} ({time.time()-t0:.0f}s)", flush=True)

    if generated:
        out_path = outdir / "generated.extxyz"
        ase_write(str(out_path), generated, format="extxyz")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

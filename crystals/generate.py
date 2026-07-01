#!/usr/bin/env python3
"""Generate crystals steered to a target Combined score.

Mixes two (element-set, natoms) distributions — a HIGH-Combined pool and a LOW-Combined
pool — at fraction lambda chosen so the expected Combined hits --target, then generates
crystals from the checkpoint with those naked prompts.

    python generate.py --target 0.5 --n 100 --out generated.extxyz

Then score with:  python score.py generated.extxyz

--target in [0.1, 0.8].  lambda = (target - LO) / (HI - LO), where HI/LO are the validated
anchor Combined values of the two distributions (defaults below; override with --hi/--lo).
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# prefer code bundled in the package; fall back to a live checkout
for p in [os.path.join(HERE, "lemat/code/py/ed"), os.path.join(HERE, "lemat/code/py/dielectric"),
          "/home/ubuntu/code/py/ed", "/home/ubuntu/code/py/dielectric"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

DEF_CKPT = os.path.join(HERE, "checkpoints/alex_nolemat_lowhull")
DEF_HIGH = os.path.join(HERE, "lemat/data/distributions/HIGH_rareearth.json")
DEF_LOW = os.path.join(HERE, "lemat/data/distributions/LOW_broad_HPtRh.json")
HI_DEFAULT, LO_DEFAULT = 0.85, 0.14      # measured n=100 anchors (ladder-calibrated); reachable range ~0.15-0.85


def load_pool(path):
    d = json.load(open(path))
    t = d["tuples"] if isinstance(d, dict) and "tuples" in d else d
    return [(x["elements"], int(x["natoms"]), int(x.get("count", 1))) for x in t]


def sample(pool, n, seed):
    if n <= 0:
        return []
    rng = random.Random(seed)
    keys = [(e, nn) for e, nn, _ in pool]
    w = [c for _, _, c in pool]
    return [f"{e} | {nn} | " for e, nn in rng.choices(keys, weights=w, k=n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True, help="target Combined in [0.1, 0.8]")
    ap.add_argument("--n", type=int, default=100, help="number of structures to generate")
    ap.add_argument("--out", default="generated.extxyz")
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--high", default=DEF_HIGH)
    ap.add_argument("--low", default=DEF_LOW)
    ap.add_argument("--hi", type=float, default=HI_DEFAULT, help="HIGH pool anchor Combined")
    ap.add_argument("--lo", type=float, default=LO_DEFAULT, help="LOW pool anchor Combined")
    ap.add_argument("--version", default="d15_binrho_k7")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lam = (args.target - args.lo) / (args.hi - args.lo)
    lam = max(0.0, min(1.0, lam))
    k = max(0, min(args.n, round(lam * args.n)))
    print(f"target={args.target}  ->  lambda={lam:.3f}  ->  {k}/{args.n} from HIGH, {args.n-k} from LOW", flush=True)

    high, low = load_pool(args.high), load_pool(args.low)
    prompts = sample(high, k, args.seed) + sample(low, args.n - k, args.seed + 7)
    random.Random(args.seed).shuffle(prompts)

    from eval.screening import load_model, generate_one
    from ase.io import write
    print(f"loading checkpoint {args.ckpt} ...", flush=True)
    model, sp = load_model(args.ckpt, args.device)
    out = []
    for i, src in enumerate(prompts):
        try:
            g = generate_one(model, sp, src, "alex", args.version, args.device, top_k=10)
            if g is not None and getattr(g, "atoms", None) is not None:
                a = g.atoms
                a.info["source"] = src
                out.append(a)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(prompts)} ({len(out)} valid)", flush=True)
    write(args.out, out)
    print(f"\nwrote {len(out)} structures -> {args.out}", flush=True)
    print(f"expected Combined ~= {args.target} (verify with: python score.py {args.out})", flush=True)


if __name__ == "__main__":
    main()

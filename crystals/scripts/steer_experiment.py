#!/usr/bin/env python3
"""Steerability experiment (paper §2.3): the PROMPT-POOL distribution determines
Combined. Model can be disjoint from the prompt dataset.

Steps:
 2. For each cluster/pool of prompts: generate from the model, score Combined
    (our stability + genbench novelty/uniqueness). -> per-pool Combined.
 3. Mix the highest- and lowest-Combined pools at ratios lambda -> generate+score
    -> show Combined(lambda) interpolates (target any value in between).

Model + scorer loaded once. Usage:
    python steer_experiment.py --pools <dir of *.txt> --ckpt <dir> --version d15_binrho_k7
                               --n 40 --out <dir> [--mix]
"""
from __future__ import annotations
import sys, os, glob, json, time, argparse, random
sys.path[:0] = ["/home/ubuntu/code/py/dielectric", "/home/ubuntu/code/py/ed",
                "/home/ubuntu/packages/lemat-genbench/src"]
from pymatgen.io.ase import AseAtomsAdaptor

def gen_pool(model, sp, prompts, version, device, top_k=10):
    from eval.screening import generate_one
    out = []
    for src in prompts:
        try:
            g = generate_one(model, sp, src, "alex", version, device, top_k=top_k)
            if g is not None and getattr(g, "atoms", None) is not None:
                out.append(g.atoms)
        except Exception:
            pass
    return out

def score(atoms_list, calc, sun_metric):
    from chem.stability import compute_e_above_hull
    structs = []
    for a in atoms_list:
        eah = compute_e_above_hull(a, calc=calc, timeout=120).get("e_above_hull")
        s = AseAtomsAdaptor.get_structure(a)
        if eah is not None:
            s.properties["e_above_hull_mean"] = float(eah)
        structs.append(s)
    if not structs:
        return {"n": 0, "combined": None}
    m = sun_metric.compute(structs).metrics
    return {"gen": len(atoms_list), "n": len(structs),
            "stable": m.get("stable_count"), "metastable": m.get("metastable_count"),
            "sun": round(m.get("sun_rate", 0), 4), "msun": round(m.get("msun_rate", 0), 4),
            "combined": round(m.get("combined_sun_msun_rate", 0), 4)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", default="d15_binrho_k7"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True)
    ap.add_argument("--mix", action="store_true", help="after per-pool, mix hi+lo at lambda ladder")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    random.seed(1234)

    from eval.screening import load_model
    from chem.stability import load_stability_calc
    from lemat_genbench.metrics.sun_metric import SUNMetric
    print("loading model + mace + SUNMetric(LeMat-Bulk) ...", flush=True)
    model, sp = load_model(args.ckpt, args.device)
    calc = load_stability_calc(device=args.device)
    sun = SUNMetric()  # LeMat-Bulk reference loaded once, cached across .compute() calls

    pool_prompts = {os.path.basename(f)[:-4]: [l.strip() for l in open(f) if l.strip()][:args.n]
                    for f in sorted(glob.glob(os.path.join(args.pools, "*.txt")))}
    results = {}
    for name, prompts in pool_prompts.items():
        t = time.time()
        atoms = gen_pool(model, sp, prompts, args.version, args.device)
        r = score(atoms, calc, sun)
        results[name] = r
        print(f"  [{name}] gen {r.get('gen')}/{len(prompts)} | combined={r.get('combined')} "
              f"(stable {r.get('stable')}, meta {r.get('metastable')}) {time.time()-t:.0f}s", flush=True)
    json.dump(results, open(os.path.join(args.out, "per_pool.json"), "w"), indent=2)

    if args.mix and len(results) >= 2:
        valid = {k: v for k, v in results.items() if v.get("combined") is not None}
        hi = max(valid, key=lambda k: valid[k]["combined"])
        lo = min(valid, key=lambda k: valid[k]["combined"])
        print(f"\nMIX: hi={hi}({valid[hi]['combined']})  lo={lo}({valid[lo]['combined']})", flush=True)
        hp = pool_prompts[hi]; lp = pool_prompts[lo]; mix = {}
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = int(round(lam * args.n))
            prompts = random.sample(hp, min(k, len(hp))) + random.sample(lp, min(args.n - k, len(lp)))
            atoms = gen_pool(model, sp, prompts, args.version, args.device)
            r = score(atoms, calc, sun)
            target = round(lam * valid[hi]["combined"] + (1 - lam) * valid[lo]["combined"], 4)
            mix[lam] = {"target_linear": target, **r}
            print(f"  lambda={lam}: target~{target}  achieved combined={r.get('combined')}", flush=True)
        json.dump(mix, open(os.path.join(args.out, "mix_ladder.json"), "w"), indent=2)

if __name__ == "__main__":
    main()

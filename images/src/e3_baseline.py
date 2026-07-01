"""E3 — baseline for Claim 3 (the headline): make "only showing reaches the joint" measured.

For animal AND nature, at matched sample size + CIs, four ways to reach the joint:
  telling_animal  — sample the animal class-group (one property prompt)   [have ~13.8%]
  telling_nature  — sample the nature class-group (the other property)    [have ~11.2%]
  compositional   — energy-based conjunction of an animal-class and a      [NEW build]
                    nature-class conditional, composed at sampling time (compose.py)
  showing         — bin-mixing over classes high on BOTH                  [have ~35.6%]

Same bars for every pool: bar = each property's own-axis telling mean (pooled across
seeds). Joint hit = (sim_animal >= bar_a) AND (sim_nature >= bar_n). Report joint
hit-rate per pool with: across-seed mean +/- 95% CI (>=3 seeds) AND bootstrap-over-
images 95% CI on the pooled images. If showing beats compositional guidance, that is
the paper's strongest single result.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G
from compose import generate_composed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
N = 128                 # images per pool per seed
CFG = 4.0               # guidance for telling/showing pools
COMPOSE_SCALE = 4.0     # per-concept weight for the energy-based conjunction (matches CFG)
SEEDS = [13, 29, 41]


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def boot_ci(mask, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(mask, float)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5), \
        float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    ga, gn = d["g_sim_animal"], d["g_sim_nature"]
    za, zn = zscore(ga), zscore(gn)
    animal_classes = np.argsort(za)[::-1][:300]
    nature_classes = np.argsort(zn)[::-1][:300]
    joint_classes = np.argsort(np.minimum(za, zn))[::-1][:30]
    print("showing joint classes:", sorted(joint_classes.tolist()), flush=True)

    pipe = G.load_pipe(); L = Labeler()

    # collect per-seed labels for each pool
    pools = ["telling_animal", "telling_nature", "compositional", "showing"]
    data = {p: {"animal": [], "nature": [], "per_seed_animal": [], "per_seed_nature": []} for p in pools}

    for s in SEEDS:
        rng = np.random.default_rng(s)
        # telling_animal
        ids = G.sample_mu(G.uniform_pool(animal_classes), N, seed=s)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=s))
        data["telling_animal"]["per_seed_animal"].append(l["sim_animal"]); data["telling_animal"]["per_seed_nature"].append(l["sim_nature"])
        # telling_nature
        ids = G.sample_mu(G.uniform_pool(nature_classes), N, seed=s + 1)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=s + 1))
        data["telling_nature"]["per_seed_animal"].append(l["sim_animal"]); data["telling_nature"]["per_seed_nature"].append(l["sim_nature"])
        # showing
        ids = G.sample_mu(G.uniform_pool(joint_classes), N, seed=s + 2)
        l = L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=s + 2))
        data["showing"]["per_seed_animal"].append(l["sim_animal"]); data["showing"]["per_seed_nature"].append(l["sim_nature"])
        # compositional: pair a random animal class with a random nature class per sample
        ca = rng.choice(animal_classes, N).tolist()
        cb = rng.choice(nature_classes, N).tolist()
        imgs = generate_composed(pipe, ca, cb, scale=COMPOSE_SCALE, seed=s + 3)
        l = L.labels(imgs)
        data["compositional"]["per_seed_animal"].append(l["sim_animal"]); data["compositional"]["per_seed_nature"].append(l["sim_nature"])
        print(f"seed {s} done", flush=True)

    # pooled arrays
    for p in pools:
        data[p]["animal"] = np.concatenate(data[p]["per_seed_animal"])
        data[p]["nature"] = np.concatenate(data[p]["per_seed_nature"])

    # bars = each property's own-axis telling mean (pooled across seeds)
    bar_a = float(data["telling_animal"]["animal"].mean())
    bar_n = float(data["telling_nature"]["nature"].mean())

    res = {"N_per_seed": N, "cfg": CFG, "compose_scale": COMPOSE_SCALE, "seeds": SEEDS,
           "bar_animal": bar_a, "bar_nature": bar_n,
           "showing_classes": sorted(joint_classes.tolist()), "pools": {}}
    print(f"\nbars: animal>={bar_a:.3f} nature>={bar_n:.3f}")
    print(f"{'pool':16s} {'mAnimal':>8s} {'mNature':>8s} {'joint%':>7s}  across-seed CI / bootstrap CI")
    for p in pools:
        a, nn = data[p]["animal"], data[p]["nature"]
        hit_mask = (a >= bar_a) & (nn >= bar_n)
        pooled_hit = float(hit_mask.mean()) * 100
        # per-seed hit rates -> across-seed CI
        per_seed = []
        for ai, ni in zip(data[p]["per_seed_animal"], data[p]["per_seed_nature"]):
            per_seed.append(float(((ai >= bar_a) & (ni >= bar_n)).mean()) * 100)
        m, lo_s, hi_s = across_seed_ci(per_seed)
        lo_b, hi_b = boot_ci(hit_mask, seed=7)
        res["pools"][p] = {
            "mean_animal": float(a.mean()), "mean_nature": float(nn.mean()),
            "joint_hit_pct_pooled": pooled_hit,
            "joint_hit_per_seed": per_seed,
            "across_seed_mean": m, "across_seed_CI95": [lo_s, hi_s],
            "bootstrap_CI95": [lo_b, hi_b]}
        print(f"{p:16s} {a.mean():8.3f} {nn.mean():8.3f} {pooled_hit:7.1f}  "
              f"seed[{lo_s:.1f},{hi_s:.1f}] boot[{lo_b:.1f},{hi_b:.1f}]", flush=True)

    sh = res["pools"]["showing"]["across_seed_mean"]
    comp = res["pools"]["compositional"]["across_seed_mean"]
    best_tell = max(res["pools"]["telling_animal"]["across_seed_mean"],
                    res["pools"]["telling_nature"]["across_seed_mean"])
    res["showing_minus_compositional_pts"] = sh - comp
    res["showing_minus_best_telling_pts"] = sh - best_tell
    print(f"\nshowing {sh:.1f}%  compositional {comp:.1f}%  best-telling {best_tell:.1f}%")
    print(f"showing - compositional = {sh-comp:+.1f} pts ; showing - best telling = {sh-best_tell:+.1f} pts")

    json.dump(res, open(f"{OUT}/e3_baseline.json", "w"), indent=2)
    print("E3-BASELINE-DONE")


if __name__ == "__main__":
    main()

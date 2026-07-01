"""E9 — stronger compositional baseline + the privileged-info check (review.2 C5, review.3 C4).
Also bumps the E3 Claim-3 headline to >=10 seeds (review round-2 follow-up).

The round-1 win was showing 44% vs telling/compositional ~13%. Two reviewer asks:
 (1) give Claim 3 the STRONGEST recognized compositional baseline — Composable / energy-based
     diffusion (proper conjunction of the two class-conditional score directions, compose.py),
     not just guidance-averaging;
 (2) PRIVILEGED-INFO check — showing is handed the audit's per-class scores (it knows which
     classes score high on both). Give *telling* the same information: pick the best class(es)
     from the audit and ask for them. If telling-with-audit closes the gap, the dichotomy is
     definitional; if showing still wins with both sides using the audit, it is operational.

Pools (animal AND nature, CFG=4, N/seed x SEEDS, same bars for all = each property's own-axis
telling mean):
  telling_animal      naive single-property prompt (no audit)
  telling_nature      the other naive single-property prompt
  compositional       Composable Diffusion conjunction, eps_u + s(eps_a-eps_u)+s(eps_b-eps_u)
  telling_audit_best  PRIVILEGED: point mass on the single class with the highest audit min(z_a,z_n)
  telling_audit_top5  PRIVILEGED: uniform over the top-5 audit joint classes (a small asked-for set)
  showing             uniform over the top-30 audit joint classes (bin-mixing)
Report joint hit-rate +/- across-seed and bootstrap 95% CI for every pool. The continuum
best1 -> top5 -> showing(top30) shows whether the spread of showing is what does the work.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G
from compose import generate_composed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
N = 128
CFG = 4.0
COMPOSE_SCALE = 4.0
SEEDS = [13, 29, 41, 57, 71, 83, 97, 109, 127, 149]   # 10 seeds


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def boot_ci(mask, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(mask, float)
    m = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    ga, gn = d["g_sim_animal"], d["g_sim_nature"]
    za, zn = zscore(ga), zscore(gn)
    animal_classes = np.argsort(za)[::-1][:300]
    nature_classes = np.argsort(zn)[::-1][:300]
    joint_rank = np.argsort(np.minimum(za, zn))[::-1]
    joint_top30 = joint_rank[:30]
    audit_best1 = [int(joint_rank[0])]
    audit_top5 = joint_rank[:5].tolist()
    print("audit best single class:", audit_best1, "top5:", audit_top5, flush=True)

    pipe = G.load_pipe(); L = Labeler()
    pools = ["telling_animal", "telling_nature", "compositional",
             "telling_audit_best", "telling_audit_top5", "showing"]
    data = {p: {"per_seed_animal": [], "per_seed_nature": []} for p in pools}

    def std_pool(classes, seed):
        ids = G.sample_mu(G.uniform_pool(classes), N, seed=seed)
        return L.labels(G.generate(pipe, ids, guidance_scale=CFG, seed=seed))

    for s in SEEDS:
        rng = np.random.default_rng(s)
        for name, classes, off in [("telling_animal", animal_classes, 0),
                                    ("telling_nature", nature_classes, 1),
                                    ("telling_audit_best", audit_best1, 4),
                                    ("telling_audit_top5", audit_top5, 5),
                                    ("showing", joint_top30, 2)]:
            l = std_pool(classes, s + off)
            data[name]["per_seed_animal"].append(l["sim_animal"])
            data[name]["per_seed_nature"].append(l["sim_nature"])
        # compositional (energy-based conjunction): pair random animal+nature classes per sample
        ca = rng.choice(animal_classes, N).tolist(); cb = rng.choice(nature_classes, N).tolist()
        l = L.labels(generate_composed(pipe, ca, cb, scale=COMPOSE_SCALE, seed=s + 3))
        data["compositional"]["per_seed_animal"].append(l["sim_animal"])
        data["compositional"]["per_seed_nature"].append(l["sim_nature"])
        print(f"seed {s} done", flush=True)

    for p in pools:
        data[p]["animal"] = np.concatenate(data[p]["per_seed_animal"])
        data[p]["nature"] = np.concatenate(data[p]["per_seed_nature"])

    bar_a = float(data["telling_animal"]["animal"].mean())
    bar_n = float(data["telling_nature"]["nature"].mean())
    res = {"N_per_seed": N, "cfg": CFG, "compose_scale": COMPOSE_SCALE, "n_seeds": len(SEEDS),
           "bar_animal": bar_a, "bar_nature": bar_n,
           "audit_best1": audit_best1, "audit_top5": audit_top5,
           "showing_classes": sorted(joint_top30.tolist()), "pools": {}}
    print(f"\nbars: animal>={bar_a:.3f} nature>={bar_n:.3f}")
    print(f"{'pool':20s} {'mA':>6s} {'mN':>6s} {'joint%':>7s}  across-seed / bootstrap CI")
    for p in pools:
        a, nn = data[p]["animal"], data[p]["nature"]
        hit_mask = (a >= bar_a) & (nn >= bar_n)
        pooled = float(hit_mask.mean()) * 100
        per_seed = [float(((ai >= bar_a) & (ni >= bar_n)).mean()) * 100
                    for ai, ni in zip(data[p]["per_seed_animal"], data[p]["per_seed_nature"])]
        m, lo_s, hi_s = across_seed_ci(per_seed)
        lo_b, hi_b = boot_ci(hit_mask, seed=7)
        res["pools"][p] = {"mean_animal": float(a.mean()), "mean_nature": float(nn.mean()),
                           "joint_hit_pct_pooled": pooled, "joint_hit_per_seed": per_seed,
                           "across_seed_mean": m, "across_seed_CI95": [lo_s, hi_s],
                           "bootstrap_CI95": [lo_b, hi_b]}
        print(f"{p:20s} {a.mean():6.3f} {nn.mean():6.3f} {pooled:7.1f}  "
              f"seed[{lo_s:.1f},{hi_s:.1f}] boot[{lo_b:.1f},{hi_b:.1f}]", flush=True)

    sh = res["pools"]["showing"]["across_seed_mean"]
    for ref in ["compositional", "telling_audit_best", "telling_audit_top5"]:
        res[f"showing_minus_{ref}_pts"] = sh - res["pools"][ref]["across_seed_mean"]
    print(f"\nshowing - compositional      = {res['showing_minus_compositional_pts']:+.1f} pts")
    print(f"showing - telling_audit_best = {res['showing_minus_telling_audit_best_pts']:+.1f} pts (privileged)")
    print(f"showing - telling_audit_top5 = {res['showing_minus_telling_audit_top5_pts']:+.1f} pts (privileged)")

    json.dump(res, open(f"{OUT}/e9_privileged.json", "w"), indent=2)
    print("E9-PRIVILEGED-DONE")


if __name__ == "__main__":
    main()

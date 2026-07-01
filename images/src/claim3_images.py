"""
Claim 3 (images) — is the entangled joint "animal AND nature" reachable only by showing?

Operationalization, native to DiT's class-only conditioning and matching the framework:
  TELLING  = the coarse per-PROPERTY prompt. You can ask for "animal" (sample the animal
             class-group) or "nature" (sample the nature class-group), but a single prompt
             cannot name the intersection. Two telling baselines, one per property.
  SHOWING  = a fine exemplar distribution mu_P on the BINS that already score high on BOTH
             (animal classes that are also nature-rich: otter, pelican, ...). Bin selection
             can target the intersection; a property prompt cannot.

If the claim holds: each single-property telling pool lands high on its own axis but near
chance on the other, so its JOINT hit rate is low; showing lands high on both, so its joint
hit rate is much higher. If false: telling reaches the joint about as well as showing.

Joint hit = (sim_animal >= ta) AND (sim_nature >= tn), where the bars ta, tn are each pool's
own-axis telling mean — i.e. "as animal as a typical animal prompt AND as nature as a typical
nature prompt." Same bars applied to every pool.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
N = 160          # images per pool
CFG = 4.0        # same guidance for every pool, so only the class distribution differs


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def pool_mu(classes):
    mu = np.zeros(1000)
    mu[classes] = 1.0 / len(classes)
    return mu


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    ga, gn = d["g_sim_animal"], d["g_sim_nature"]
    za, zn = zscore(ga), zscore(gn)

    animal_classes = np.argsort(za)[::-1][:300]          # "animal" property group
    nature_classes = np.argsort(zn)[::-1][:300]          # "nature" property group
    joint_classes = np.argsort(np.minimum(za, zn))[::-1][:30]  # high on BOTH -> showing bins

    print("showing (joint-rich) class ids:", sorted(joint_classes.tolist()))
    print(f"  their mean g_animal={ga[joint_classes].mean():.3f} g_nature={gn[joint_classes].mean():.3f}")

    pipe = G.load_pipe()
    L = Labeler()

    pools = {
        "telling_animal": pool_mu(animal_classes),
        "telling_nature": pool_mu(nature_classes),
        "showing_joint":  pool_mu(joint_classes),
    }

    lab = {}
    for name, mu in pools.items():
        ids = G.sample_mu(mu, N, seed=13)
        imgs = G.generate(pipe, ids, guidance_scale=CFG, seed=13)
        l = L.labels(imgs)
        lab[name] = {"animal": np.asarray(l["sim_animal"]), "nature": np.asarray(l["sim_nature"])}
        print(f"{name}: mean animal={lab[name]['animal'].mean():.3f}  nature={lab[name]['nature'].mean():.3f}")

    # bars: each property's own-axis telling mean
    ta = float(lab["telling_animal"]["animal"].mean())
    tn = float(lab["telling_nature"]["nature"].mean())

    res = {"N": N, "cfg": CFG, "bar_animal": ta, "bar_nature": tn,
           "showing_classes": sorted(joint_classes.tolist()), "pools": {}}
    print(f"\nbars: animal>={ta:.3f}, nature>={tn:.3f}")
    print(f"{'pool':16s} {'mean_animal':>11s} {'mean_nature':>11s} {'JOINT hit %':>11s}")
    for name, l in lab.items():
        hit = float(np.mean((l["animal"] >= ta) & (l["nature"] >= tn))) * 100
        res["pools"][name] = {"mean_animal": float(l["animal"].mean()),
                              "mean_nature": float(l["nature"].mean()),
                              "joint_hit_pct": hit}
        print(f"{name:16s} {l['animal'].mean():11.3f} {l['nature'].mean():11.3f} {hit:11.1f}")

    sj = res["pools"]["showing_joint"]["joint_hit_pct"]
    best_tell = max(res["pools"]["telling_animal"]["joint_hit_pct"],
                    res["pools"]["telling_nature"]["joint_hit_pct"])
    res["showing_minus_best_telling_pct"] = sj - best_tell
    print(f"\nshowing {sj:.1f}%  vs  best telling {best_tell:.1f}%  ->  gap {sj-best_tell:+.1f} pts")

    with open(f"{OUT}/claim3_images.json", "w") as f:
        json.dump(res, f, indent=2)
    print("CLAIM3-IMAGES-DONE")


if __name__ == "__main__":
    main()

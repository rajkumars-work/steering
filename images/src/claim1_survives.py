"""
Claim 1, the part that matters: does the data-audit's budget split SURVIVE the model?

The split T (within-bin) + E (between-bin) = V is an identity on the data, so on its own
it only checks our bookkeeping. The claim with teeth: read T and E from the DATA audit,
then GENERATE from the model, bin its outputs the same way, and recompute T and E from the
generated images. If the data-side budget is real, the model reproduces the same split.

Target: brightness. Bins: ImageNet classes (the model's conditioning = the bin).
A subset of classes, generated uniformly, at the raw conditional (CFG=1) so no extra steering.

Note on E: a per-class mean estimated from n samples carries sampling noise that inflates the
between-class variance by (within-var)/n. We report the naive and the noise-corrected E.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
CLASSES = list(range(0, 1000, 25))   # 40 classes, an unbiased spread
N = 40                               # images per class
CFG = 1.0                            # raw conditional, no extra steering


def split(means, vars):
    """uniform-weight within/between split from per-class means and variances"""
    T = float(np.mean(vars))               # within-bin: average per-class variance
    E = float(np.var(means))               # between-bin: spread of the class means
    return T, E


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    gd = d["g_brightness"][CLASSES]
    vd = d["v_brightness"][CLASSES]
    T_data, E_data = split(gd, vd)
    print(f"DATA audit over {len(CLASSES)} classes (brightness):")
    print(f"  within T = {T_data:.5f}   between E = {E_data:.5f}   total = {T_data+E_data:.5f}")

    pipe = G.load_pipe()
    L = Labeler()
    gen_means, gen_vars = [], []
    for c in CLASSES:
        imgs = G.generate(pipe, [c] * N, guidance_scale=CFG, seed=1000 + c)
        b = np.asarray(L.labels(imgs)["brightness"])
        gen_means.append(float(b.mean()))
        gen_vars.append(float(b.var()))
    gen_means = np.array(gen_means); gen_vars = np.array(gen_vars)

    T_model, E_naive = split(gen_means, gen_vars)
    E_model = E_naive - T_model / N        # remove sampling-noise inflation
    print(f"\nMODEL's own outputs, same bins (brightness):")
    print(f"  within T = {T_model:.5f}   between E = {E_model:.5f} (naive {E_naive:.5f})   total = {T_model+E_model:.5f}")

    # how well do the per-class means line up (the bridge), as a bonus
    r = float(np.corrcoef(gd, gen_means)[0, 1])

    res = {"classes": CLASSES, "n_per_class": N, "cfg": CFG, "target": "brightness",
           "data":  {"T": T_data, "E": E_data, "V": T_data + E_data},
           "model": {"T": T_model, "E": E_model, "E_naive": E_naive, "V": T_model + E_model},
           "ratio": {"T_model_over_data": T_model / T_data, "E_model_over_data": E_model / E_data},
           "per_class_mean_r2": r * r}
    print(f"\n  T model/data = {T_model/T_data:.2f}   E model/data = {E_model/E_data:.2f}")
    print(f"  per-class mean alignment r^2 = {r*r:.3f}")
    with open(f"{OUT}/claim1_survives.json", "w") as f:
        json.dump(res, f, indent=2)
    print("CLAIM1-SURVIVES-DONE")


if __name__ == "__main__":
    main()

"""E12 REVISION — bins-vs-knobs over a PROPERTY PANEL; ratio vs E/(E+T) (round-5 revision).

The first E12 found the bins-vs-knobs balance is property-dependent (bins win for aesthetic 4.4x,
knob wins for brightness 0.38x). The budget predicts this: showing's reach ~ sqrt(chi2*E),
telling's ~ sqrt(chi2*T), so the bin/knob shift ratio should RISE with the between-bin fraction
E/(E+T). Make it a law over a panel of targets spanning the E/(E+T) spectrum.

Panel = the 7 audit targets (E/(E+T) from E8, no new audit needed; imagenet stream is gated/slow
here so we cannot cheaply add new pixel targets — flagged). For each target:
  knob_shift = best single class (argmax g) with CFG pushed to its limit (best of {1,7,15}) minus
               that class at CFG=1   (the per-prompt knob's max reach)
  bin_shift  = top-8 recipe at CFG=1 minus the J_P baseline   (bin choice's reach)
  ratio = bin_shift / knob_shift
PRIMARY OUTPUT: scatter of ratio vs E/(E+T), per-point CIs (>=5 seeds).
PRE-REGISTERED: ratio rises monotonically with E/(E+T); ratio>1 (data wins) at high E/(E+T),
ratio<1 (knob wins) at low E/(E+T). Brightness/filesize are directly guidance-coupled (annotate).
CLIP-ROBUST brightness: also measure brightness on non-clipped pixels (lum<250) so the low-E point
isn't an over-exposure artifact; report raw and clip-robust.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
# E/(E+T) at the 1000-class key (from E8 / claim1)
EFRAC = {"sim_animal": 0.723, "sim_food": 0.632, "sim_vehicle": 0.568, "sim_nature": 0.530,
         "filesize_per_mp": 0.299, "aesthetic": 0.272, "brightness": 0.168}
GUIDANCE_COUPLED = {"brightness", "filesize_per_mp"}   # low-level pixel stats CFG amplifies directly
TARGETS = ["sim_animal", "sim_food", "sim_vehicle", "sim_nature", "filesize_per_mp", "aesthetic", "brightness"]
CFGS_KNOB = [1.0, 7.0, 15.0]
TOPK = 8
N = 64
SEEDS = [11, 23, 37, 53, 67]


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def clip_robust_brightness(imgs):
    out = np.empty(len(imgs), np.float32)
    for i, im in enumerate(imgs):
        a = np.asarray(im.convert("RGB"), np.float32)
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        nc = lum[lum < 250]
        out[i] = (nc.mean() if nc.size > 50 else lum.mean()) / 255.0
    return out


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    cnt = d["count"].astype(float); JP = cnt / cnt.sum()
    sup = np.where(JP > 0)[0]
    pipe = G.load_pipe(); L = Labeler()

    # shared J_P baseline @ CFG1 (per seed); all standard targets read from these images
    base_imgs = {}
    base_lab = {t: [] for t in TARGETS}
    base_bright_robust = []
    for s in SEEDS:
        imgs = G.generate(pipe, G.sample_mu(JP, N, seed=s), guidance_scale=1.0, seed=s)
        lab = L.labels(imgs)
        for t in TARGETS:
            base_lab[t].append(np.asarray(lab[t], float))
        base_bright_robust.append(clip_robust_brightness(imgs))
        print(f"baseline seed {s} done", flush=True)

    def measure(target, imgs, lab=None):
        if target == "brightness_robust":
            return clip_robust_brightness(imgs)
        if lab is None:
            lab = L.labels(imgs)
        return np.asarray(lab[target], float)

    # entries: each target + brightness_robust as an extra
    entries = list(TARGETS) + ["brightness_robust"]
    res = {"N": N, "seeds": SEEDS, "cfgs_knob": CFGS_KNOB, "topk": TOPK, "efrac": EFRAC,
           "guidance_coupled": sorted(GUIDANCE_COUPLED), "panel": {}}

    for t in entries:
        gkey = "brightness" if t == "brightness_robust" else t
        g = d[f"g_{gkey}"]
        best_class = int(sup[np.argmax(g[sup])])
        topk = sup[np.argsort(g[sup])[::-1][:TOPK]]

        # baseline mean for this entry, per seed
        if t == "brightness_robust":
            B0_seed = [float(b.mean()) for b in base_bright_robust]
        else:
            B0_seed = [float(b.mean()) for b in base_lab[t]]

        knob_means = {c: [] for c in CFGS_KNOB}
        bin_means = []
        for s in SEEDS:
            for c in CFGS_KNOB:
                imgs = G.generate(pipe, [best_class] * N, guidance_scale=c, seed=3000 + s + int(c))
                knob_means[c].append(float(measure(t, imgs).mean()))
            imgs = G.generate(pipe, G.sample_mu(G.uniform_pool(topk), N, seed=4000 + s),
                              guidance_scale=1.0, seed=4000 + s)
            bin_means.append(float(measure(t, imgs).mean()))

        knob_shift_seed = [max(knob_means[c][k] - knob_means[1.0][k] for c in CFGS_KNOB) for k in range(len(SEEDS))]
        bin_shift_seed = [bin_means[k] - B0_seed[k] for k in range(len(SEEDS))]
        ratio_seed = [bin_shift_seed[k] / knob_shift_seed[k] if abs(knob_shift_seed[k]) > 1e-9 else float("nan")
                      for k in range(len(SEEDS))]
        km, klo, khi = across_seed_ci(knob_shift_seed)
        bm, blo, bhi = across_seed_ci(bin_shift_seed)
        rv = np.array([r for r in ratio_seed if np.isfinite(r)])
        rm, rlo, rhi = across_seed_ci(rv) if len(rv) >= 2 else (float(np.mean(rv)), float("nan"), float("nan"))
        res["panel"][t] = {"efrac": EFRAC[gkey], "best_class": best_class,
                           "knob_shift": km, "knob_shift_CI95": [klo, khi],
                           "bin_shift": bm, "bin_shift_CI95": [blo, bhi],
                           "ratio_bin_over_knob": rm, "ratio_CI95": [rlo, rhi],
                           "guidance_coupled": gkey in GUIDANCE_COUPLED}
        print(f"{t:18s} E/(E+T)={EFRAC[gkey]:.3f}  knob={km:.4g} bin={bm:.4g}  "
              f"ratio={rm:.2f} CI[{rlo:.2f},{rhi:.2f}]", flush=True)

    json.dump(res, open(f"{OUT}/e12_panel.json", "w"), indent=2)

    # primary figure: ratio vs E/(E+T)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(v["efrac"], v["ratio_bin_over_knob"], v["ratio_CI95"], k, v["guidance_coupled"])
           for k, v in res["panel"].items() if k != "brightness_robust"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for ef, r, ci, name, gc in pts:
        err = [[max(0, r - ci[0])], [max(0, ci[1] - r)]] if np.isfinite(ci[0]) else None
        ax.errorbar(ef, r, yerr=err, fmt="s" if gc else "o",
                    c="#d62728" if gc else "#1f77b4", capsize=3, ms=8)
        ax.annotate(name.replace("sim_", ""), (ef, r), textcoords="offset points", xytext=(6, 5), fontsize=9)
    # clip-robust brightness as an open marker at brightness's E/(E+T)
    br = res["panel"]["brightness_robust"]
    ax.errorbar(br["efrac"], br["ratio_bin_over_knob"], fmt="s", mfc="none", c="#d62728", ms=10,
                label="brightness (clip-robust)")
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("E/(E+T)  (between-bin fraction of the budget)")
    ax.set_ylabel("bin shift / knob shift   (data vs knob)")
    ax.set_yscale("log")
    ax.set_title("E12 — bins-vs-knobs ratio rises with E/(E+T)\n(>1 data wins, <1 knob wins; red = guidance-coupled pixel stat)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_e12_ratio_vs_efrac.png", dpi=130)
    print("saved figure -> fig_e12_ratio_vs_efrac.png")

    # report monotonicity (Spearman of ratio vs efrac over real targets)
    efs = [p[0] for p in pts]; rs = [p[1] for p in pts]
    rx = np.argsort(np.argsort(efs)); ry = np.argsort(np.argsort(rs))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    res["spearman_ratio_vs_efrac"] = rho
    json.dump(res, open(f"{OUT}/e12_panel.json", "w"), indent=2)
    print(f"Spearman(ratio, E/(E+T)) = {rho:.3f}")
    print("E12-PANEL-DONE")


if __name__ == "__main__":
    main()

"""E14 — strongest + audit-calibrated knob on the bins-vs-knobs panel (round-6, r1#1/r2#2, r1#2).

The reviewer worry: telling = a CFG-sweep/naming knob, so E12's 3-8x "bins beat knobs" might be
"beats THIS knob," not "beats ANY knob." Give telling its STRONGEST shot with AUDIT PARITY and show
the ratio survives. Per target we compare bin-selection (showing) against three telling knobs:

  knob_cfg_default   best audit class @ CFG=4                       (the E12 reference point)
  knob_cfg_strong    best audit class, CFG chosen by the audit from a WIDE grid {1..25}
                     (audit-calibrated: parity with showing, which also uses the audit)
  knob_soft          a LEARNED soft-prompt (textual-inversion-style continuous per-block
                     conditioning) optimized to the target scorer, warm-started at the best
                     audit class, anchored on the top audit classes (soft_prompt.py). The
                     strongest per-prompt knob a reviewer imagines closes the gap.

For every knob we report (>=3 seeds, CIs):
  - realized mean-shift Delta = mean(knob) - mean(best_class @ CFG=1)   [same zero as E12]
  - realized WITHIN-BIN chi2_tell(q||p) vs the best-class CFG=1 distribution p (E7 histogram +
    noise floor), and the within-bin ceiling sqrt(chi2 * v_b); VERIFY Delta <= ceiling.
  - ResNet-key stay-fraction: fraction of knob images whose ResNet class == best_class (does the
    knob stay within the bin, or has it started doing bin-selection? -- the honest mechanism check).
bin_shift (showing) = top-8 audit recipe @ CFG=1 minus the J_P baseline (as E12).
RATIO recomputed against the STRONGEST knob = bin_shift / max(all knob Deltas).

PRE-REGISTERED (commit 5bafc8b, 2026-07-01 22:11 UTC): even the strongest/audited knob stays
chi2_tell = O(1)-O(10) (not O(100)); Delta <= sqrt(chi2*T) holds for every knob; the bin/knob
ratio stays >1 for the bin-win targets (aesthetic, sim_animal, sim_vehicle) and <1 for brightness;
the equal-effort sqrt(E/T) ordering is unchanged. If a learned knob realizes chi2 >> 10 AND closes
a bin-win gap, report it plainly (it would genuinely qualify the headline).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G
import soft_prompt as SP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")

TARGETS = ["brightness", "aesthetic", "sim_animal", "sim_vehicle"]
BIN_WIN = {"aesthetic", "sim_animal", "sim_vehicle"}          # expected ratio > 1
CFG_WIDE = [1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 25.0]       # wide grid for the strongest CFG knob
TOPK = 8
K_ANCHOR = 16                                                 # soft-prompt anchor classes
N = 64
SEEDS = [11, 23, 37]
BINS = 12
# E/(E+T) at the 1000-class key (from E8) and within-bin budget v_b per target for the ceiling
EFRAC = {"sim_animal": 0.723, "sim_food": 0.632, "sim_vehicle": 0.568, "sim_nature": 0.530,
         "filesize_per_mp": 0.299, "aesthetic": 0.272, "brightness": 0.168}


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def chi2_hist(samp_q, samp_p, edges, alpha=0.5):
    cq, _ = np.histogram(samp_q, bins=edges); cp, _ = np.histogram(samp_p, bins=edges)
    p = (cp + alpha) / (cp.sum() + alpha * len(cp)); q = (cq + alpha) / (cq.sum() + alpha * len(cq))
    return float((((q - p) ** 2) / p).sum())


def boot_chi2_ci(samp_q, samp_p, edges, reps=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        qi = samp_q[rng.integers(0, len(samp_q), len(samp_q))]
        vals.append(chi2_hist(qi, samp_p, edges))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def measure_arm(imgs_by_seed, keys_by_seed, target, p_samp, base_mean, vb, best_class):
    """imgs_by_seed: list over seeds of label arrays for `target`. Returns arm dict."""
    q = np.concatenate(imgs_by_seed)
    alls = np.concatenate([q, p_samp]); lo, hi = float(alls.min()), float(alls.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, BINS + 1)
    x2 = chi2_hist(q, p_samp, edges)
    h = len(p_samp) // 2
    floor = chi2_hist(p_samp[:h], p_samp[h:], edges)
    c2lo, c2hi = boot_chi2_ci(q, p_samp, edges, seed=7)
    delta_seed = [float(a.mean()) - base_mean for a in imgs_by_seed]
    dm, dlo, dhi = across_seed_ci(delta_seed)
    ceiling = (max(0.0, x2) * vb) ** 0.5
    stay = float(np.mean([np.mean(np.asarray(k) == best_class) for k in keys_by_seed])) \
        if keys_by_seed is not None else None
    return {"delta": dm, "delta_CI95": [dlo, dhi], "delta_per_seed": delta_seed,
            "chi2_within": x2, "chi2_within_CI95": [c2lo, c2hi], "noise_floor_chi2": floor,
            "chi2_minus_floor": max(0.0, x2 - floor), "ceiling_sqrt_chi2_vb": ceiling,
            "delta_le_ceiling": bool(abs(dm) <= ceiling + 1e-9), "key_stay_frac": stay}


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    cnt = d["count"].astype(float); JP = cnt / cnt.sum(); sup = np.where(JP > 0)[0]
    pipe = G.load_pipe(); L = Labeler()
    tables = SP.install_soft_slot(pipe)

    # shared J_P baseline @ CFG=1 (per seed) -> B0 for every target's bin arm
    print("=== shared J_P baseline @ CFG=1 ===", flush=True)
    base_lab = {t: [] for t in TARGETS}
    for s in SEEDS:
        lab = L.labels(G.generate(pipe, G.sample_mu(JP, N, seed=s), guidance_scale=1.0, seed=s))
        for t in TARGETS:
            base_lab[t].append(np.asarray(lab[t], float))
        print(f"  baseline seed {s} done", flush=True)

    res = {"targets_order": TARGETS, "cfg_wide": CFG_WIDE, "topk": TOPK, "k_anchor": K_ANCHOR,
           "N": N, "seeds": SEEDS, "commit": "5bafc8b", "panel": {}}

    for t in TARGETS:
        g = d[f"g_{t}"]; v = d[f"v_{t}"]
        best_class = int(sup[np.argmax(g[sup])])
        topk = sup[np.argsort(g[sup])[::-1][:TOPK]]
        anchors = [int(c) for c in sup[np.argsort(g[sup])[::-1][:K_ANCHOR]]]
        vb = float(v[best_class])
        print(f"\n=== {t}  best_class={best_class}  E/(E+T)={EFRAC[t]:.3f}  vb={vb:.4g} ===", flush=True)

        # p = within-bin baseline: best class @ CFG=1 (pooled over seeds), and its mean
        p_seed = []
        for s in SEEDS:
            lab = L.labels(G.generate(pipe, [best_class] * N, guidance_scale=1.0, seed=500 + s))
            p_seed.append(np.asarray(lab[t], float))
        p_samp = np.concatenate(p_seed); base_mean = float(p_samp.mean())

        # --- knob_cfg_strong: scan wide CFG grid (cheap), audit-pick the best, full eval ---
        scan = {}
        for c in CFG_WIDE:
            lab = L.labels(G.generate(pipe, [best_class] * 48, guidance_scale=c, seed=600 + int(c)))
            scan[c] = float(np.asarray(lab[t], float).mean())
        c_star = max(CFG_WIDE, key=lambda c: scan[c])
        print(f"  CFG scan {{c:mean}} -> best CFG*={c_star} (mean {scan[c_star]:.4f}); "
              f"CFG4 mean {scan.get(4.0, float('nan')):.4f}", flush=True)

        arms_lab = {"knob_cfg_default": [], "knob_cfg_strong": []}
        arms_key = {"knob_cfg_default": [], "knob_cfg_strong": []}
        for s in SEEDS:
            for name, c in [("knob_cfg_default", 4.0), ("knob_cfg_strong", c_star)]:
                imgs = G.generate(pipe, [best_class] * N, guidance_scale=c, seed=1500 + s + int(c))
                arms_lab[name].append(np.asarray(L.labels(imgs)[t], float))
                arms_key[name].append(L.key(imgs))

        # --- knob_soft: learn a soft-prompt to maximise the target, then full eval ---
        print(f"  optimizing soft-prompt (K={K_ANCHOR} anchors)...", flush=True)
        theta, f_es = SP.optimize_soft(pipe, L, t, best_class, anchors, tables,
                                       n_es=32, iters=25, sigma0=0.35, seed=0)
        best_rows = SP.class_rows(tables, best_class)
        anchor_rows = [SP.class_rows(tables, c) for c in anchors]
        SP.set_soft_from_theta(tables, best_rows, anchor_rows, theta)
        soft_lab, soft_key = [], []
        for s in SEEDS:
            imgs = G.generate(pipe, [SP.SOFT_ID] * N, guidance_scale=4.0, seed=1800 + s)
            soft_lab.append(np.asarray(L.labels(imgs)[t], float)); soft_key.append(L.key(imgs))
        arms_lab["knob_soft"] = soft_lab; arms_key["knob_soft"] = soft_key

        # --- bin arm (showing): top-8 recipe @ CFG=1 vs J_P baseline B0 ---
        bin_seed = []
        for s in SEEDS:
            lab = L.labels(G.generate(pipe, G.sample_mu(G.uniform_pool(topk), N, seed=2500 + s),
                                      guidance_scale=1.0, seed=2500 + s))
            bin_seed.append(np.asarray(lab[t], float))
        B0_seed = [float(b.mean()) for b in base_lab[t]]
        bin_shift_seed = [float(bin_seed[k].mean()) - B0_seed[k] for k in range(len(SEEDS))]
        bm, blo, bhi = across_seed_ci(bin_shift_seed)

        # measure every knob arm
        arms = {}
        for name in ["knob_cfg_default", "knob_cfg_strong", "knob_soft"]:
            arms[name] = measure_arm(arms_lab[name], arms_key[name], t, p_samp, base_mean,
                                     vb, best_class)
            a = arms[name]
            print(f"  {name:17s} Delta={a['delta']:+.4f}  chi2={a['chi2_within']:.2f}"
                  f"(floor {a['noise_floor_chi2']:.2f})  ceil={a['ceiling_sqrt_chi2_vb']:.4f}"
                  f"  Delta<=ceil={a['delta_le_ceiling']}  stay={a['key_stay_frac']}", flush=True)

        strongest_delta = max(arms[n]["delta"] for n in arms)
        ratio_vs_strong = bm / strongest_delta if abs(strongest_delta) > 1e-9 else float("inf")
        res["panel"][t] = {
            "efrac": EFRAC[t], "best_class": best_class, "vb": vb,
            "T_proxy_vb": vb, "baseline_mean_cfg1": base_mean,
            "cfg_star": c_star, "cfg_scan": {str(c): scan[c] for c in CFG_WIDE},
            "knobs": arms, "soft_theta_norm": float(np.linalg.norm(theta)),
            "bin_shift": bm, "bin_shift_CI95": [blo, bhi], "bin_shift_per_seed": bin_shift_seed,
            "strongest_knob_delta": strongest_delta,
            "ratio_bin_over_strongest_knob": ratio_vs_strong,
            "expected_bin_win": t in BIN_WIN}
        print(f"  >> bin_shift={bm:+.4f}  strongest_knob={strongest_delta:+.4f}  "
              f"RATIO={ratio_vs_strong:.2f}x  (expect {'>1' if t in BIN_WIN else '<1'})", flush=True)
        json.dump(res, open(f"{OUT}/e14_strong_knob.json", "w"), indent=2)   # checkpoint each target

    # ---- figure: E12-style ratio vs E/(E+T), strongest-knob ratio ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pts = [(res["panel"][t]["efrac"], res["panel"][t]["ratio_bin_over_strongest_knob"], t)
               for t in TARGETS]
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for ef, r, name in pts:
            gc = name == "brightness"
            ax.scatter(ef, r, s=90, c="#d62728" if gc else "#1f77b4", marker="s" if gc else "o", zorder=3)
            ax.annotate(name.replace("sim_", ""), (ef, r), textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.axhline(1.0, ls="--", c="k", lw=0.8)
        ax.set_xlabel("E/(E+T)  (between-bin fraction, data-side)")
        ax.set_ylabel("bin shift / STRONGEST knob shift")
        ax.set_yscale("log")
        ax.set_title("E14 — bins vs the STRONGEST (audit + soft-prompt) knob\n(>1 data wins; red = brightness, guidance-coupled pixel stat)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_e14_ratio_vs_efrac.png", dpi=130)
        print("saved fig_e14_ratio_vs_efrac.png")
    except Exception as e:
        print("figure skipped:", e)

    json.dump(res, open(f"{OUT}/e14_strong_knob.json", "w"), indent=2)
    print("E14-STRONG-KNOB-DONE")


if __name__ == "__main__":
    main()

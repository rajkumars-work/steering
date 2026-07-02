"""E15 — key-selection diagnostic: does data-side E/(E+T) tell you if a key is good? (round-6, r1#3/r2#1)

Both reviewers: the whole budget is relative to the key pi and you give no way to CHOOSE it. Turn
E8 into a SELECTION DIAGNOSTIC. Claim: E/(E+T) (data-side, no model) is a key-quality score -- a key
that concentrates the target's variance into E is one that SHOWING can exploit (a bin-selection knob
gains reach), while a key with low E (e.g. a random partition) leaves showing no room and the ratio
collapses to the knob floor.

Design. Across the E8 candidate keys (imagenet_1000, kmeans_100, kmeans_50, random_50 -- reconstructed
with the SAME seeds so E/(E+T) matches E8), for a few targets spanning the spectrum
(sim_animal high-E, aesthetic mid, brightness low-E):
  x = data-side E/(E+T) under the key (from E8, no generation).
  y = REALIZED bin/knob ratio under the key:
        bin_shift(key) = mean(top-8-BINS recipe of that key, CFG=1) - B0(J_P @ CFG=1)
          (top-8 bins by bin-mean, each expanded to its member classes proportional to count;
           at the 1000-class key this is exactly E14's top-8-class recipe; for a random key the
           best 8 bins are ~global-mean so the shift -> 0, the floor)
        knob_shift = the STRONGEST audited knob's shift (from E14, key-independent).
        ratio(key) = bin_shift(key) / knob_shift.
Report predicted (E8 sqrt(chi2*E)) alongside realized. PRE-REGISTERED (commit 5bafc8b): the realized
ratio RISES with E/(E+T); the random key sits at the low end (its E-collapse is the floor). Then the
selection recipe: enumerate cheap candidate keys, audit each (shadow only, no GPU), pick the one
maximizing E/(E+T) for your target; a low max across all candidates => the target is intrinsically
knob-shaped. Honest caveat: needs *a* cheap label to bin by; unsupervised key discovery = future work.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labelers import Labeler
import generate as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
TARGETS = ["sim_animal", "aesthetic", "brightness"]
TOPB = 8
N = 64
SEEDS = [11, 23, 37]
ALL_T = Labeler.TARGETS


def across_seed_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    se = v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def kmeans(F, k, seed, iters=100):        # identical to E8 (same seed -> same partition)
    rng = np.random.default_rng(seed)
    cent = F[rng.choice(len(F), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((F[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
        new = np.array([F[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab


def key_split_efrac(labels, w, g, v):
    """E/(E+T) via law of total variance (matches E8)."""
    p = w / w.sum(); gbar = float((p * g).sum()); T = E = 0.0
    for b in np.unique(labels):
        m = labels == b; W = w[m].sum()
        if W == 0:
            continue
        pB = W / w.sum(); GB = float((w[m] * g[m]).sum() / W)
        VB = float((w[m] * (v[m] + (g[m] - GB) ** 2)).sum() / W)
        T += pB * VB; E += pB * (GB - gbar) ** 2
    return E / (E + T) if (E + T) > 0 else float("nan")


def top8_recipe(labels, w, g):
    """mu over the 1000 classes: pick the TOP-8 bins by bin-mean, weight member classes by count."""
    bins = np.unique(labels)
    GB = np.array([(w[labels == b] * g[labels == b]).sum() / max(w[labels == b].sum(), 1e-9) for b in bins])
    top = bins[np.argsort(GB)[::-1][:min(TOPB, len(bins))]]
    mu = np.zeros(1000)
    for b in top:
        m = labels == b
        mu[m] = w[m]                    # proportional to class count within the selected bins
    return mu / mu.sum()


def main():
    d = np.load(f"{OUT}/claim1_perclass.npz")
    w = d["count"].astype(float); JP = w / w.sum()
    Gt = {t: d[f"g_{t}"].astype(float) for t in ALL_T}
    Vt = {t: d[f"v_{t}"].astype(float) for t in ALL_T}
    F = np.stack([(Gt[t] - Gt[t].mean()) / (Gt[t].std() + 1e-12) for t in ALL_T], axis=1)

    keys = {"imagenet_1000": np.arange(1000),
            "kmeans_100": kmeans(F, 100, seed=0),
            "kmeans_50": kmeans(F, 50, seed=0),
            "random_50": np.random.default_rng(1).integers(0, 50, size=1000)}

    # strongest-knob shift per target (key-independent) from E14; fall back to E12 panel
    knob = {}
    if os.path.exists(f"{OUT}/e14_strong_knob.json"):
        e14 = json.load(open(f"{OUT}/e14_strong_knob.json"))
        for t in TARGETS:
            if t in e14["panel"]:
                knob[t] = e14["panel"][t]["strongest_knob_delta"]
    if any(t not in knob for t in TARGETS):
        e12 = json.load(open(f"{OUT}/e12_panel.json"))
        for t in TARGETS:
            knob.setdefault(t, e12["panel"][t]["knob_shift"])
    print("knob shifts (strongest, from E14/E12):", {t: round(knob[t], 4) for t in TARGETS}, flush=True)

    pipe = G.load_pipe(); L = Labeler()

    # shared J_P baseline @ CFG=1 -> B0 per target
    base = {t: [] for t in TARGETS}
    for s in SEEDS:
        lab = L.labels(G.generate(pipe, G.sample_mu(JP, N, seed=s), guidance_scale=1.0, seed=s))
        for t in TARGETS:
            base[t].append(float(np.asarray(lab[t], float).mean()))
        print(f"baseline seed {s} done", flush=True)

    res = {"keys": list(keys), "targets": TARGETS, "topb": TOPB, "N": N, "seeds": SEEDS,
           "commit": "5bafc8b", "knob_shift": knob, "points": []}
    for t in TARGETS:
        B0 = base[t]
        for kname, lab in keys.items():
            ef = key_split_efrac(lab, w, Gt[t], Vt[t])
            mu = top8_recipe(lab, w, Gt[t])
            shift_seed = []
            for si, s in enumerate(SEEDS):
                imgs = G.generate(pipe, G.sample_mu(mu, N, seed=3000 + s), guidance_scale=1.0, seed=3000 + s)
                shift_seed.append(float(np.asarray(L.labels(imgs)[t], float).mean()) - B0[si])
            bm, blo, bhi = across_seed_ci(shift_seed)
            ratio = bm / knob[t] if abs(knob[t]) > 1e-12 else float("nan")
            pt = {"target": t, "key": kname, "efrac": ef, "bin_shift": bm, "bin_shift_CI95": [blo, bhi],
                  "knob_shift": knob[t], "ratio_bin_over_knob": ratio,
                  "bin_shift_per_seed": shift_seed}
            res["points"].append(pt)
            print(f"  {t:12s} {kname:14s} E/(E+T)={ef:.3f}  bin_shift={bm:+.4f}  ratio={ratio:.2f}", flush=True)
            json.dump(res, open(f"{OUT}/e15_key_selection.json", "w"), indent=2)

    # figure: E/(E+T) vs realized ratio, colored by target, random key marked
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        cols = {"sim_animal": "#1f77b4", "aesthetic": "#2ca02c", "brightness": "#d62728"}
        for t in TARGETS:
            pts = [p for p in res["points"] if p["target"] == t]
            xs = [p["efrac"] for p in pts]; ys = [p["ratio_bin_over_knob"] for p in pts]
            ax.plot(xs, ys, "-", c=cols[t], alpha=0.4, zorder=1)
            for p in pts:
                mk = "x" if p["key"] == "random_50" else "o"
                ax.scatter(p["efrac"], p["ratio_bin_over_knob"], c=cols[t], marker=mk, s=70, zorder=3,
                           label=t if p["key"] == "imagenet_1000" else None)
                ax.annotate(p["key"].replace("_", ""), (p["efrac"], p["ratio_bin_over_knob"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axhline(1.0, ls="--", c="k", lw=0.8)
        ax.set_xlabel("data-side E/(E+T) under the key")
        ax.set_ylabel("realized bin / knob ratio")
        ax.set_title("E15 — a key's E/(E+T) predicts the bin/knob ratio it buys\n(x = random-partition control at the floor)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/fig_e15_efrac_vs_ratio.png", dpi=130)
        print("saved fig_e15_efrac_vs_ratio.png")
    except Exception as e:
        print("figure skipped:", e)

    # spearman(efrac, ratio) pooled across all points
    ef = np.array([p["efrac"] for p in res["points"]]); rt = np.array([p["ratio_bin_over_knob"] for p in res["points"]])
    rx = np.argsort(np.argsort(ef)); ry = np.argsort(np.argsort(rt))
    res["spearman_ratio_vs_efrac_pooled"] = float(np.corrcoef(rx, ry)[0, 1])
    json.dump(res, open(f"{OUT}/e15_key_selection.json", "w"), indent=2)
    print(f"Spearman(ratio, E/(E+T)) pooled = {res['spearman_ratio_vs_efrac_pooled']:.3f}")
    print("E15-KEY-SELECTION-DONE")


if __name__ == "__main__":
    main()

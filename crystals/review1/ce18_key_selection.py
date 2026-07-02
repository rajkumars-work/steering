#!/usr/bin/env python3
"""C-E18 — key-selection diagnostic for choosing the chemistry partition pi (crystal side; mirrors
image E15). Reviews r1#3/r2#1 ("how do you choose the key?").

Two claims, both mostly data-side (reuse C-E8 + C-E14; no new generation):

(A) VALIDATION — data-side E/(E+T) predicts realized steerability. Under the operative key
    (element-set + atom-count), the data-side between-chemistry fraction E/(E+T) orders the three
    properties the same way as the C-14 realized bins-vs-knobs ratio: gap (0.979 -> 801x) >
    density (0.955 -> 91x) > stability (0.573 -> 41x). So a purely data-side quantity you can compute
    BEFORE any generation predicts which property is most chemistry-steerable. (n=3 properties;
    illustrative + mechanistic: it is exactly the budget law showing-reach ~ sqrt(chi^2 * E).)

(B) SELECTION RECIPE — pick the key that maximizes the usable showing budget. Across the C-E8 keys
    (els_nat, els_only, anion), for each property score each key by the fraction of total property
    variance that is BOTH between-chemistry AND carries over to generation:
        key_quality(key, prop) = between_frac * carryover_r2      (both in [0,1])
    This is the realized-reach^2 proxy (reach ~ sqrt(E_gen) = sqrt(between_frac * R^2 * Var)). Recipe:
    enumerate cheap chemical keys, compute key_quality on data + a small carry-over probe, pick the max.
    Honest caveat: needs a cheap chemical label; unsupervised key discovery is future work.

Pure CPU. Output: review1/ce18_key_selection.json + figures/fig_ce18_key_selection.png.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CRYS = os.path.dirname(HERE)                          # steering/crystals
def _find(*cands):
    for c in cands:
        if os.path.exists(c): return c
    raise FileNotFoundError(cands)
KEYS = _find(os.path.join(CRYS, "results", "ce8_keys.json"), os.path.join(HERE, "ce8_keys.json"))
PANEL = _find(os.path.join(CRYS, "results", "ce14_panel.json"), os.path.join(HERE, "ce14_panel.json"))
OUT = os.path.join(HERE, "ce18_key_selection.json")
FIGDIR = os.path.join(CRYS, "figures"); os.makedirs(FIGDIR, exist_ok=True)
FIG = os.path.join(FIGDIR, "fig_ce18_key_selection.png")

keys = json.load(open(KEYS))["keys"]
panel = json.load(open(PANEL))["panel"]

# ---- (A) validation: data-side E/(E+T) vs realized C-14 ratio, per property (operative key) ----
# gap is not in the C-E8 key file (composition-surrogate, near-perfect carry-over by construction);
# use its C-14 E/(E+T) and treat carry-over ~ 1.0.
val_props = {
    "gap":       {"efrac": panel["gap"]["E_over_EplusT"],       "ratio": panel["gap"]["bin_over_knob"],       "carryover_r2": None},
    "density":   {"efrac": panel["density"]["E_over_EplusT"],   "ratio": panel["density"]["bin_over_knob"],   "carryover_r2": keys["els_nat"]["carryover_r2"]["density"]["r2"]},
    "stability": {"efrac": panel["stability"]["E_over_EplusT"], "ratio": panel["stability"]["bin_over_knob"], "carryover_r2": keys["els_nat"]["carryover_r2"]["e_above_hull"]["r2"]},
}
order = sorted(val_props, key=lambda p: val_props[p]["efrac"])
efr = [val_props[p]["efrac"] for p in order]
rat = [val_props[p]["ratio"] for p in order]
# rank agreement (Spearman on n=3): both should be increasing
spearman = float(np.corrcoef(np.argsort(np.argsort(efr)), np.argsort(np.argsort(rat)))[0, 1])
monotone = all(efr[i] < efr[i+1] and rat[i] < rat[i+1] for i in range(len(order)-1))

# ---- (B) selection recipe: key_quality = between_frac * carryover_r2, per (key, property) ----
PROP_MAP = {"density": "density", "bvs_gii": "bvs_gii", "e_above_hull": "e_above_hull"}
recipe = {}
table = {}
for prop in PROP_MAP:
    scores = {}
    for kname, kd in keys.items():
        bf = kd["data_TE_split"][prop]["between_frac"]
        r2 = kd["carryover_r2"][prop]["r2"]
        scores[kname] = {"between_frac": bf, "carryover_r2": r2, "key_quality": round(bf * r2, 4)}
    best = max(scores, key=lambda k: scores[k]["key_quality"])
    table[prop] = scores
    recipe[prop] = {"best_key": best, "key_quality": scores[best]["key_quality"],
                    "ranking": sorted(scores, key=lambda k: -scores[k]["key_quality"])}

res = {
    "validation": {"per_property": val_props, "order_by_efrac": order,
                   "monotone_efrac_predicts_ratio": monotone, "spearman_n3": round(spearman, 3),
                   "note": "n=3 properties under the operative els_nat key; illustrative + mechanistic (budget law)."},
    "selection": {"key_quality_def": "between_frac * carryover_r2 (usable showing-budget fraction)",
                  "table": table, "recipe": recipe},
    "recipe_text": None,
}
# recipe sentence
picks = "; ".join(f"{p}->{recipe[p]['best_key']} (q={recipe[p]['key_quality']})" for p in PROP_MAP)
res["recipe_text"] = (
    "Choosing the key (crystal pi-selection): enumerate cheap chemical partitions (element-set, "
    "element-set+atom-count, anion family), and for the target property pick the key maximizing "
    "key_quality = between_frac x carry-over R^2 (the fraction of property variance that is both "
    "between-chemistry AND survives generation). On the C-E8 keys this recommends: " + picks + ". "
    "This validates the operative element-set+atom-count key for density and stability, and flags that "
    "the coarse anion key collapses density steerability (between_frac 0.955->0.282). The predictor is "
    "the same data-side E/(E+T) that (validation) tracks the realized C-14 ratio across properties "
    "(gap>density>stability). Caveat: this assumes a cheap chemical label exists; unsupervised key "
    "discovery is future work."
)
json.dump(res, open(OUT, "w"), indent=2)

# ---- figure: two panels ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# (A) E/(E+T) vs realized ratio
ax1.scatter(efr, rat, s=90, color="#1f4e79", zorder=3)
for p in order:
    ax1.annotate(p, (val_props[p]["efrac"], val_props[p]["ratio"]),
                 textcoords="offset points", xytext=(8, 4), fontsize=9)
ax1.set_yscale("log"); ax1.set_xlabel("data-side between-chemistry fraction  E/(E+T)")
ax1.set_ylabel("realized bins-vs-knobs ratio (C-14, log)")
ax1.set_title(f"(A) E/(E+T) predicts steerability  (monotone={monotone})")
ax1.grid(True, which="both", alpha=0.25)

# (B) key_quality per property per key (grouped bars)
props = list(PROP_MAP); knames = list(keys)
x = np.arange(len(props)); w = 0.26
colors = {"els_nat": "#1f4e79", "els_only": "#3a7ca5", "anion": "#c0605a"}
for i, kn in enumerate(knames):
    vals = [table[p][kn]["key_quality"] for p in props]
    ax2.bar(x + (i-1)*w, vals, w, label=kn, color=colors.get(kn, None))
    for pi, p in enumerate(props):
        if recipe[p]["best_key"] == kn:                # star the recipe pick for this property
            ax2.text(x[pi] + (i-1)*w, vals[pi] + 0.01, "*", ha="center", fontsize=13, color="k")
ax2.set_xticks(x); ax2.set_xticklabels(props, fontsize=9)
ax2.set_ylabel("key_quality = between_frac x carry-over R^2")
ax2.set_title("(B) pick max key_quality per property  (*=recipe pick)")
ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG, dpi=150)

print(json.dumps({"validation": res["validation"], "recipe": recipe}, indent=2))
print("\nwrote", OUT, "and", FIG)
print("CE18-DONE")

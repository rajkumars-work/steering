#!/usr/bin/env python3
"""C-E17 — bound the stability chemistry/knob ratio under weak carry-over (review r2#3).

Stability has the weakest carry-over (single-MACE R^2 = 0.488 [0.172, 0.706]; C-E9 ensemble label
0.42 [0.23, 0.66]) AND the smallest bins-vs-knobs ratio (41x, C-E14 metastable-rate re-run). r2 asks:
how much of that 41x could be a carry-over artifact vs a genuine showing advantage? Deliver a
conservative FLOOR on the real data-lever, discounted by the carry-over CI-lower bound.

Coupling model (which carry-over R^2 discounts the realized *contrast*):
  The showing advantage `bin_shift` is a contrast (a difference of chemistry-group means measured on
  generations, with chemistries SELECTED by the data-side ranking). For a linear-Gaussian
  selection-on-proxy model, selecting on a proxy with data<->gen correlation r attenuates the realized
  contrast by exactly r = sqrt(R^2). This matches the budget law directly: showing reach ~ sqrt(chi^2 * E),
  and carry-over retains a fraction R^2 of the between-chemistry variance E in generation, so the reach
  scales as sqrt(R^2 * E) = sqrt(R^2) * sqrt(E) = r * (reach at perfect carry-over). => PRIMARY model
  discounts the contrast by sqrt(R^2).
  We also report two sensitivity variants: (var) discount by R^2 itself (would be right for a variance,
  over-strict for a contrast); (raw) multiply by R^2_lower as an absolute retention fraction (crediting
  a perfect-carry-over baseline) -- the most punishing, least principled.

The realized bin_shift (0.3849) already reflects the ACTUAL (unknown) carry-over, whose POINT estimate
is 0.488. So the honest discount is RELATIVE: credit only the CI-lower carry-over instead of the point
estimate -> factor = f(R^2_lower) / f(R^2_point). At the point estimate the discount is 1.0 (no change).

Pure CPU re-analysis of C-E14 + C-E2/C-E9. No generation. Output: review1/ce17_stability_bound.json.
"""
import os, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(os.path.dirname(HERE), "results", "ce14_panel.json")  # steering/crystals/results
if not os.path.exists(PANEL):
    PANEL = os.path.join(HERE, "ce14_panel.json")
OUT = os.path.join(HERE, "ce17_stability_bound.json")

# ---- inputs (all with CIs) ----
p = json.load(open(PANEL))["panel"]["stability"]
BIN, KNOB = p["bin_shift"], p["knob_shift"]                 # 0.3849, 0.0094 (metastable-rate units)
BIN_CI, KNOB_CI = p["bin_ci95"], p["knob_ci95"]            # [0.2964,0.4676], [-0.0524,0.0712]
RATIO_OBS = p["bin_over_knob"]                              # 40.88

# carry-over R^2 for e_above_hull, point + 95% CI
CARRYOVER = {
    "single_mace_ce2": {"r2": 0.488, "ci95": [0.172, 0.706]},   # C-E2 converged, 3 seeds
    "ensemble_ce9":    {"r2": 0.42,  "ci95": [0.23,  0.66]},    # C-E9 cleaner MLIP-ensemble label
}

def discount_factors(r2_point, r2_lower):
    """relative discount: credit only the CI-lower carry-over instead of the point estimate."""
    r2_lower = max(r2_lower, 0.0)
    return {
        "attenuation_sqrtR2": math.sqrt(r2_lower) / math.sqrt(r2_point),  # PRIMARY (contrast ~ r)
        "variance_R2":        r2_lower / r2_point,                        # over-strict (contrast != var)
        "raw_R2lower":        r2_lower,                                   # ultra-conservative absolute
    }

res = {"inputs": {"bin_shift": BIN, "bin_ci95": BIN_CI, "knob_shift": KNOB, "knob_ci95": KNOB_CI,
                  "ratio_observed": RATIO_OBS, "carryover": CARRYOVER,
                  "note": ("ratio_observed uses the point tag-knob (0.0094); the tag-knob CI includes 0 "
                           "(no reliable knob effect), so the ratio's upper CI is unstable -- the robust "
                           "inference is bin_ci95 excludes 0 while knob_ci95 includes 0. C-E17 discounts "
                           "the SHOWING side for carry-over; knob uncertainty is handled separately below.")},
       "bounds": {}}

for src, c in CARRYOVER.items():
    r2p, r2lo = c["r2"], c["ci95"][0]
    facs = discount_factors(r2p, r2lo)
    entry = {"r2_point": r2p, "r2_lower": r2lo, "models": {}}
    for model, fac in facs.items():
        dbin = BIN * fac
        dbin_lo = BIN_CI[0] * fac                       # discount the showing lower CI too
        ratio_pt   = dbin / KNOB                          # vs the point tag-knob
        ratio_floor = dbin_lo / KNOB_CI[1] if KNOB_CI[1] > 0 else None  # vs worst-case knob (most conservative)
        entry["models"][model] = {
            "discount_factor": round(fac, 4),
            "discounted_bin_shift": round(dbin, 4),
            "ratio_vs_point_knob": round(ratio_pt, 1),
            "ratio_floor_worstcase_knob": round(ratio_floor, 1) if ratio_floor else None,
        }
    res["bounds"][src] = entry

# headline: primary (sqrt-R2 attenuation) under single-MACE CI-lower carry-over, vs point knob
prim = res["bounds"]["single_mace_ce2"]["models"]["attenuation_sqrtR2"]
res["headline"] = {
    "floor_ratio": prim["ratio_vs_point_knob"],
    "model": "sqrt(R^2) contrast attenuation (matches showing-reach ~ sqrt(chi^2*E))",
    "carryover_source": "single-MACE C-E2, CI-lower R^2=0.172 vs point 0.488",
    "sensitivity": {
        "variance_R2_model": res["bounds"]["single_mace_ce2"]["models"]["variance_R2"]["ratio_vs_point_knob"],
        "raw_R2lower_model": res["bounds"]["single_mace_ce2"]["models"]["raw_R2lower"]["ratio_vs_point_knob"],
        "ensemble_label_ce9_primary": res["bounds"]["ensemble_ce9"]["models"]["attenuation_sqrtR2"]["ratio_vs_point_knob"],
    },
    "worstcase_knob_ratio_raw_model": res["bounds"]["single_mace_ce2"]["models"]["raw_R2lower"]["ratio_floor_worstcase_knob"],
    "sentence": None,  # filled below
}
h = res["headline"]
h["sentence"] = (
    f"Even crediting only the CI-lower carry-over (single-MACE R^2=0.172 vs the 0.488 point estimate), "
    f"stability's chemistry-selection advantage is >= {h['floor_ratio']:.0f}x the measured "
    f"within-chemistry tag-knob (sensitivity: {h['sensitivity']['variance_R2_model']:.0f}x under a "
    f"variance-discount, {h['sensitivity']['raw_R2lower_model']:.0f}x under the most punishing raw-R^2 "
    f"discount); under the cleaner C-E9 ensemble label the floor is "
    f"{h['sensitivity']['ensemble_label_ce9_primary']:.0f}x. All these compare against the tag-knob's "
    f"point effect (0.0094); the only way the ratio falls below 1x is to also credit the tag-knob its "
    f"full upper-CI (0.071) -- but that knob CI includes 0, i.e. telling has no statistically reliable "
    f"within-chemistry effect, so that is not a genuine telling win. Contrast with gap (800x) and "
    f"density (91x), whose carry-over is near-perfect (R^2>=0.98) so they need essentially no discount: "
    f"the headline should trust gap/density and caveat stability as >= {h['floor_ratio']:.0f}x."
)

json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res["headline"], indent=2))
print("\nCE17-DONE ->", OUT)

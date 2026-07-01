# Round 5 — pre-registered predictions (committed BEFORE the runs)

## C-E13 — coverage vs number of properties (2 → 3)

**Pair/triple selection (data-side, `ce13_tripleselect.json`, n=36 chemistries).** Correlation
matrix shows band gap is anti-correlated with *everything* (eps₀ −0.42, eps_inf −0.92, density
−0.29, BVS −0.44), while the polarizable cluster {eps₀, eps_inf, density} is mutually *positively*
correlated (+0.3…+0.6), and density↔BVS is sample-unstable (−0.91 vs +0.50). **⚠ So no third
property is anti-correlated with both gap and eps₀ — the data is effectively ~2-D.** Best-available
third axis = **BVS-GII** (least positive with eps₀, +0.35). The 3-property run uses
{gap, eps₀, BVS-GII}; the imperfect 3rd axis makes the result a **lower bound** on the N-scaling
effect. A clean test needs a formation-energy / bulk-modulus surrogate (not available on-box).

**Corners (high in each, pre-registered):** gap ≥ 2.5 eV, eps₀ ≥ 35, BVS-GII ≥ 0.5.
coverage = min(frac_gap-corner, frac_eps-corner, frac_bvs-corner). N=120/method × 3 seeds.

**Prediction:** at n=3, one chemistry cannot occupy all three corners (gap is anti with the other
two), so **telling coverage ≈ 0** and **conditioning ≈ 0** (a single multi-tag request collapses;
only gap/density are taggable, not eps₀/BVS). **Showing (mix of the three per-corner specialists)
holds coverage > 0.** Headline trend vs n=2 (C-E12: showing 0.40, conditioning 0.017, telling 0):
showing stays clearly positive while telling/conditioning remain ≈0; showing's absolute coverage
may fall (min over 3 < min over 2) but the showing-vs-rest gap persists. Also report whether the
best single chemistry's within-bin spread reaches each corner (variance-reach).

## C-E14 — bins beat knobs (does steering live in the data?)

One property = band gap. **knob shift** = mean_gap(broad chems + bg-vhigh) − mean_gap(broad chems +
bg-vlow) [same chemistry, tag flipped]; **bin shift** = mean_gap(top-gap chems) − mean_gap(bottom-gap
chems) [chemistry selection, no tag]. N=120 × 3 seeds, bootstrap CIs.

**Prediction:** **bin shift ≫ knob shift** (ratio bin/knob > 1, plausibly ≫ 1) — the tag knob moves
the batch mean weakly (C-E12 showed bg-vhigh barely raised gap — only 1.7% reached the high-gap
corner), whereas chemistry selection spans the data's full gap range. This is the direct number
behind "steering lives in the data, not the property-tag knobs."

Committed before both runs (see git hash of this commit). Report whatever happens, including if
the knob turns out stronger than expected or showing fails to hold coverage at n=3.

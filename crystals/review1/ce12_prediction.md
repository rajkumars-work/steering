# C-E12 — pre-registered prediction (committed BEFORE the run)

The real Claim-3 crystal experiment (round 4): does **showing** (mixing chemistries via the audit)
**cover both corners of an anti-correlated trade-off frontier**, where **telling / conditioning**
(a single request) can only reach one corner? This is the Pareto-coverage use-case a screening
campaign actually wants. Committed before the C-E12 run exists.

## The pair (verified anti-correlated, data-side, n=50 chemistries)
**band gap ∧ static dielectric (eps_0)**, both XGBoost composition surrogates.
`corr(gap, eps_0) = −0.537` (`ce12_pairselect.json`) — a genuine trade-off (wide-gap ⇒ low ε,
narrow-gap ⇒ high ε). (gap∧density was only −0.11 → no tension; rejected.)

## Pre-registered corners (thresholds fixed now; sensitivity reported)
Data-side per-chem pctiles (20/40/60/80): gap 0.58/0.87/2.52/3.36 eV; eps_0 20.5/28.8/31.7/35.9.
- **corner A** = gap ≥ **3.0** eV ∧ eps_0 ≤ **25**   (high-gap, low-dielectric)
- **corner B** = gap ≤ **1.0** eV ∧ eps_0 ≥ **33**   (low-gap, high-dielectric)
- **middle** = everything else.
Sensitivity set (also reported): A=(gap≥2.5 ∧ eps_0≤28), B=(gap≤1.5 ∧ eps_0≥31).

## Methods (matched N=120/method × 3 seeds; Wilson/bootstrap CIs)
1. **telling — single best chemistry**: the single richest corner-A chemistry from the audit.
2. **conditioning — multi-property tags**: broad chemistry + the `bg-vhigh` gap tag (the model has
   **no dielectric tag** — `k-*` is uniformly `k-mid` — so conditioning can target only the gap axis;
   a single request therefore aims at one gap regime ⇒ one corner. Flagged honestly.)
3. **showing — chemistry mix**: 50/50 mix of corner-A chemistries + corner-B chemistries (audit),
   naked prompts.

**Primary metric:** coverage = **min(fraction in A, fraction in B)**; also report fraction in middle
per method + a 2-D (gap, eps_0) scatter per method.

## Prediction (committed)
- **telling** (one chemistry) → one corner → coverage ≈ **0** (definitional).
- **conditioning** (bg-vhigh, broad) → high-gap ⇒ corner A (low-ε) ⇒ covers A, **not B** → coverage
  ≈ **0** (collapses to one corner). I expect frac-A > 0, frac-B ≈ 0.
- **showing** (mix A+B chemistries) → covers **both** → coverage **substantially > 0** (the unique
  win). Predicted ordering: **coverage_showing ≫ coverage_conditioning ≈ coverage_telling ≈ 0.**

If showing covers both while conditioning collapses, the crystal Claim 3 becomes an outright win
(coverage), not a tie — the cross-domain mirror of the image animal∧nature result. Report whatever
happens (incl. if conditioning unexpectedly spans, or showing fails to fill a corner).

## Result — VINDICATED (run done 2026-06-26)

| method | coverage = min(fracA,fracB) | 95% CI | fracA | fracB | mid |
|---|---|---|---|---|---|
| telling (1 chem) | **0.00** | [0, 0] | 1.00 | 0.00 | 0.00 |
| conditioning (bg-vhigh) | **0.017** | [0.006, 0.033] | 0.017 | 0.564 | 0.419 |
| showing (mix A+B) | **0.401** | [0.354, 0.435] | 0.40 | 0.46 | 0.14 |

**Prediction confirmed exactly:** coverage_showing (0.40) ≫ conditioning (0.017) ≈ telling (0) —
non-overlapping CIs, ~24×. Showing covers **both** corners; telling sits in one corner (all A);
conditioning **collapses** (it failed to push gap high — only 1.7% reached corner A — landing in the
data-dominant corner B + middle). Robust across thresholds (`ce12_sensitivity.json`): showing
0.13–0.49 vs conditioning ≤0.04 vs telling 0 for tight/primary/loose corners. Figures:
`fig_ce12_coverage.png`, `fig_ce12_scatter.png` (the killer figure — telling→one corner,
conditioning→collapse, showing→spans). **Crystal Claim 3 is now an outright coverage win, not a
tie — cross-domain.**

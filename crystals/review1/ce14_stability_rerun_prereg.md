# C-E14 STABILITY CELL RE-RUN — pre-registered prediction (committed BEFORE the run)

Per handoff `# C-E14 STABILITY RE-RUN` (2026-06-29): the stability row in `ce14_panel.json` is an
artifact (it used **mean raw `e_above_hull`**, which is mostly noise, and ranked bins by the noisy
`ce2_peritem` audit mean → garbage 1.2–1.4 eV/atom generations, wrong direction → spurious knob-win
0.46×). Re-run **just the stability cell** with the corrected design. gap (801×) and density (91×)
cells are unchanged and stand.

## Corrected design

- **Metric → metastable rate** = fraction of generated structures with `e_above_hull ≤ 0.1` eV/atom
  (the `metastable` flag, `claim1_survival_spectrum.py:112`; same scorer `compute_e_above_hull`,
  single-MACE, used on BOTH sides → matched metric).
- **bin side (data lever):** sample `(elements, natoms)` tuples from the **an_lh distribution**
  (`dist_verify/distributions/an_lh.dist.json`, count-weighted), filtered to the within-dist
  push_HIGH / push_LOW element sets — bin_hi ⊆ {Dy,Er,Y,Hg,P,Bi,Sb,In,Nd,Tl}, bin_lo ⊆
  {H,Pt,Rh,Mn,In,Ho,Nd,Pr,Mg,Ni}. This is the **same universe `binning_experiment.sample()` drew
  from** for the validated 84% / 1.7%. Generate **tag-free** (`"<els> | <n> | "`, top_k=10).
  **Sanity gate:** bin_hi must regenerate ≥50% metastable; if ~0% the path is still wrong → flag.
  - **CORRECTION (2 attempts burned, this note amended before relaunch):** the first design drew
    bin pools from *training* chemistries whose elements ⊆ the push_HIGH/LOW top-10 `final_elems`,
    with training atom counts. That invented **out-of-distribution** combos (N up to 20, e.g.
    `Nd Tl Y | 20`) the model cannot realize → `atoms None` → 0 scorable → all-NaN. `final_elems`
    is only an element *summary*; the validated 84% came from a refined 17-tuple pool sampled from
    the an_lh distribution. Fix = sample the an_lh distribution tuples directly (probe: 39/40 = 97%
    generation). **No change to the claim, metric, or prediction** — only the chemistry-sampling
    source is corrected to match the validated pipeline.
- **ROOT CAUSE of the all-NaN runs = load order (deterministic, not transient):** loading the MACE
  stability calc *before* the generation model makes `generate_one` return `atoms=None` for **every**
  prompt. Proven by A/B with the web app suspended: **calc-first 0/8 vs model-first 8/8.** The script
  now loads `load_model` **before** `load_stability_calc`. (The earlier MP-outage and web/app.py
  contention hypotheses were both wrong — suspending the web app did not fix it; the order did.)
  Fix is order-only: load model first, then MACE, and **leave** the global default at float64 — do
  NOT reset it to float32 (MACE builds inputs at the global default at scoring time, so a reset
  re-breaks NaCl scoring). Verified: model-first + no-reset → NaCl=0.001 and generation 6/6.
- **Belt-and-suspenders:** the script still **preflights** scorer (NaCl metastable) and generator
  (≥30% of 12 bin_hi prompts yield atoms) and exits 75 = RETRYABLE rather than grind into NaN; an
  **auto-retry supervisor** (`run_ce14_stab_super.sh`) retries on exit 75 within a 16h deadline.
  With the load-order fix these gates should pass on attempt 1.
- **knob side (hull tag):** broad audit chemistries, flip `hull-vlow` (high stability) vs `hull-high`
  (low stability) at fixed chemistry; same metastable metric.
- N=80 × 3 seeds, pooled bootstrap CIs. bin_shift = rate(bin_hi) − rate(bin_lo); knob_shift =
  rate(hull-vlow) − rate(hull-high); **ratio = bin_shift / knob_shift**, plotted at stability's
  E/(E+T)=0.573 on the panel scatter (replacing the artifact point).

## Prediction (strong DATA-WIN, reversing the artifact)

- **bin metastable rate:** bin_hi ≳ 0.7, bin_lo ≲ 0.1 → **bin_shift ≳ 0.6** (consistent with the
  within-dist validated 84% → 1.7%). Passes the sanity gate.
- **knob metastable rate:** hull-vlow > hull-high but modestly; **knob_shift < 0.2** (the hull tag is
  a weak lever on the *realized* metastable rate relative to chemistry selection).
- **ratio = bin_shift / knob_shift > 1 (plausibly ≫ 1)** — stability becomes a **data-win**, not the
  spurious 0.46 knob-win. This is the direction required by the budget law (chemistry carries the
  metastability signal) and consistent with gap/density.
- If instead the hull tag turns out to be the stronger lever (knob_shift ≥ bin_shift), report that
  honestly — the artifact would then have been only the *metric*, not the direction.

Committed before the run (see git hash of this commit).

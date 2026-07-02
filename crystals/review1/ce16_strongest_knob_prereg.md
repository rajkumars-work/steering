# C-E16 — strongest / audit-calibrated knob on the bins-vs-knobs panel (pre-registered)

Per Round-6 handoff (r1#1/r2#2 "knob is a strawman"; r1#2 side-info parity). The 801×/91×/41×
bin/knob ratios (C-E14) pit chemistry-selection (showing) against the property-**tag** knob (telling).
Reviewers: give telling its **strongest** shot **with audit parity** and show the ratio survives.

## The strongest audit-calibrated within-chemistry knob = best-of-N selection

Showing uses the property scorer (the "audit") to **select chemistries**. The parity knob gives telling
the **same scorer**, but restricted to act **within a fixed chemistry** (so it can only exploit the
within-chemistry budget T, not the between-chemistry budget E). The strongest such knob is **best-of-N
test-time selection**: at fixed chemistry, generate N tag-free candidates, keep the single best by the
property scorer. This is (i) audit-calibrated (uses the scorer), (ii) strictly within-chemistry (parity),
(iii) as strong as telling can get without touching chemistry, and — crucially — (iv) has an **analytic
telling effort**:

  χ²_tell(N) = χ²( p_bestofN ‖ p_base ) = N²/(2N−1) − 1

(the max-order-statistic reweighting p_sel(x)=N·F(x)^{N-1}p(x) gives ∫p_sel²/p−1 = N²∫₀¹u^{2N-2}du−1).
N=1→0, N=2→0.33, N=5→1.78, N=10→4.26, N=16→7.26. This is exactly the χ² the handoff asks us to report
so the main session can draw the **equal-effort √(E/T) floor** alongside the realized ratio.

The property-**tag** knob (bg-*/rho-*/hull-*, the original C-E14 knob) is kept as the labelled "weak
knob" reference point (its χ² is not analytic; reported as Δ only).

## Design

- **One tag-free candidate pool per chemistry**, generated once, scored for all three properties, so
  best-of-N is an order-statistic over the pool (no per-N regeneration). Pool: 3 seeds × ~30 broad
  audit chemistries (count-weighted, same universe as the C-E14 knob condition) × M=12 candidates.
- **Properties / scorers (identical to C-E14):** gap = composition-XGBoost surrogate (eV); density =
  geometry (g/cc); stability = single-MACE metastable rate (fraction e_above_hull ≤ 0.1), to match the
  C-E14 stability re-run metric. best-of-N "goodness": maximize gap, maximize density, **minimize
  e_above_hull** (→ maximize metastability) for stability.
- **best-of-k sweep** k ∈ {1,2,4,8,12}: per chemistry, average the max-of-random-k-subset over many
  draws; average over chemistries → Δ_tell(k) = |mean_bestofk − mean_base|, in the property's C-14 units
  (eV / g·cc⁻¹ / metastable-rate). χ²_tell(k) analytic.
- **Budget quantities from the same pools:** T = mean over chemistries of within-chemistry property
  variance; E = variance over chemistries of the chemistry-mean property; report E/(E+T) (cross-check vs
  C-E8/C-E14) and √(E/T).
- **Showing side reused from C-E14** (bin_shift: gap 2.709, density 8.404, stability 0.385).
- **Report (bootstrap CI over chemistries, 3 seeds):** Δ_tell(k) & χ²_tell(k) for the sweep; the
  **strongest-knob Δ** (k=12); the recomputed **bin/knob ratio vs the strongest knob** = bin_shift /
  Δ_tell(12); the budget-ceiling check **Δ_tell(k) ≤ √(χ²_tell(k)·T)**; and √(E/T).

## Predictions (pre-registered)

1. **The best-of-N knob is much stronger than the tag knob** (Δ_tell(12) ≫ the C-E14 tag knob_shift) —
   we are genuinely steel-manning telling.
2. **Even against this strongest knob, chemistry still wins on gap & density**: bin/knob ratio vs
   best-of-12 stays **> 1** for gap and density (plausibly still ≫ 1). Report honestly if a strong knob
   closes it.
3. **The budget ceiling holds**: Δ_tell(k) ≤ √(χ²_tell(k)·T) for every k, every property (telling cannot
   out-reach its √(χ²·T) frontier).
4. **The equal-effort ratio √(E/T) is unchanged by knob strength** and orders gap > density > stability,
   matching the E/(E+T) ranking — the structural invariant the realized ratio only approximates.
5. **Stability is the one where a strong within-chemistry knob is most competitive** (lowest E/(E+T)); its
   realized ratio vs best-of-12 may fall toward or below 1 — consistent with the budget law (low E/(E+T)
   ⇒ telling relatively most useful), NOT a counterexample. Report whatever happens.

Committed before the run (see git hash of this commit).

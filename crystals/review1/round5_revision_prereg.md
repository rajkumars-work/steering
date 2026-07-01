# Round 5 REVISION — pre-registration (2026-06-29)

Applied the image-side design fixes before finalizing.

## C-E13 — revised metric (number of corners covered) + premise gate
- Primary metric: **number of corners covered** (corner fraction's bootstrap 95% CI excludes 0);
  min(fractions) demoted to secondary. Pre-registered expectation (IF premise holds):
  telling=1, conditioning=0, showing=N; structural gap = N−1.
- **Mandatory premise gate:** every pairwise corr among the 3 properties must be ≲0.
- **Outcome — premise FAILS:** pairwise gap-eps0 −0.42, gap-BVS −0.44, **eps0-BVS +0.35** (not all
  negative; data ~2-D). Best-available least-correlated triple {gap, eps0, BVS}, contamination
  flagged. Revised metric on the existing run: telling=2, conditioning=3, showing=3 — the clean
  separation does NOT appear because the premise fails (variance-reach + the +0.35 contamination).
  **Conclusion: the clean 3-property coverage isn't demonstrable on crystals with available
  surrogates; the n=2 C-E12 result stands as the crystal coverage evidence.** (`ce13_revised.json`.)

## C-E14 — revised to a property panel; ratio vs E/(E+T) (committed BEFORE the run)
Panel of taggable (knob-able) properties: **band gap (bg-*), density (rho-*), stability (hull-*)**
(dielectric k-* is constant → no knob, excluded). Per property: knob shift (tag lo→hi, chemistry
fixed) vs bin shift (chemistry selection lo→hi); ratio = bin/knob; E/(E+T) from the C-E2 budget.
**Prediction:** bin/knob ratio **rises with E/(E+T)** (showing reach ∝√(χ²·E), telling ∝√(χ²·T));
ratio > 1 (data/chemistry wins) for high-E/(E+T) properties; watch for any <1 (tag-knob wins) at
low E/(E+T). The crystal knob is a property-*tag* (weaker than image CFG), so expect data to win
for most chemistry-determined properties — the panel tests whether the *ordering* by E/(E+T) holds.

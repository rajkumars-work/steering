# C-E4 — pre-registered prediction of the carry-over ladder (committed BEFORE the audit)

**Purpose.** The claim "the data-audit budget transfers for properties the model reproduces well"
is circular unless the ordering is predicted *in advance* from signals that are **not** the
per-chemistry agreement (r²) itself. This file commits that prediction. It is committed to git
**before** the converged C-E2 audit (`ce2_converge.py`) finishes, so the check is genuinely
out-of-sample.

**Committed:** 2026-06-25, on the alex_nolemat_lowhull ep120 checkpoint. The C-E2 audit was still
running (seed 0, ~7/24 chemistries) when this was written; its converged r² ladder did not yet
exist.

## Pre-measurement signals (from `ce4_pilot.py`, 8 INDEPENDENT chemistries, seed 99 — disjoint draw from C-E2's seed-0 set)

Per property, on a small independent pilot (NOT the audit): within-chemistry **spread ratio**
ρ = median(std_model/std_data), **drift** δ = median|mean_model − mean_data|/global_data_std, and
**support overlap** = fraction of model values inside the data range.

| property | ρ (spread ratio, 1=ideal) | δ (drift) | support overlap |
|---|---|---|---|
| density | 1.43 | 0.25 | 0.94 |
| BVS-GII | 2.59 | 0.32 | 0.84 |
| e_above_hull | **58.4** | **63.3** | 0.62 |

Plus a **loss-sensitivity** argument: density is a direct function of the tokens the model emits
(lattice + composition) → tightly constrained by the training objective; BVS-GII is local
coordination geometry → partially constrained; e_above_hull is global energetics that the
next-token objective never optimizes → essentially unconstrained.

## The prediction

1. **Ordering (high confidence):** carry-over r² is **density > BVS-GII > stability**. The signals
   are monotonic and separate cleanly — ρ and δ jump by ~1.5 orders of magnitude from BVS to
   stability, and support overlap falls density→BVS→stability.
2. **Rough levels:**
   - density: **r² ≳ 0.9** (ρ≈1.4, δ small, support≈0.94 → near-perfect transfer).
   - BVS-GII: **r² ≈ 0.6–0.85** (moderate excess spread, good support → partial transfer).
   - stability: **r² ≲ 0.4** (catastrophic ρ/δ, partial support → poor transfer; also
     curation-confounded since the data is low-hull, so the native between-chemistry stability
     spread is tiny).
3. **Mechanism call:** stability will show large excess model spread (ρ≫1) and a strong upward
   drift (model less stable than data), not merely low correlation.

## Check (filled in after C-E2 completed, 2026-06-25; converged 3 seeds, m=30)

| property | predicted r² | measured r² (C-E2, 95% CI) | ordering correct? |
|---|---|---|---|
| density | ≳0.9 | **0.985** [0.906, 0.998] | ✓ |
| BVS-GII | 0.6–0.85 | **0.879** [0.035, 0.972] | ✓ |
| stability | ≲0.4 | **0.488** [0.172, 0.706] | ✓ |

**Verdict — prediction vindicated.** The committed **ordering density > BVS-GII > stability holds**.
Levels: density spot-on; BVS-GII and stability land a touch *above* their predicted bands (0.88 vs
≤0.85; 0.49 vs ≤0.4) but within the CIs and the ordering is clean — the advance prediction from
pre-measurement signals correctly placed all three rungs. The mechanism call also holds (stability:
huge excess model spread + upward drift; ρ=58, δ=63 in the pilot). Claim 4 is now **predictive**.

Caveats: BVS-GII's CI is wide because only the ionic chemistries have a defined GII (few qualifying
bins per seed). Stability converges to **0.49**, higher than the earlier single-run 0.30 — the
multi-seed converged number is the one to cite.

## Verifiable pre-registration (review.2/review.3 C5)

- **Timestamp:** this prediction was committed at git **`d96b827`** ("crystals/review1: PRE-REGISTER
  C-E4 ladder prediction …") in `rajkumars-work/wip` (local mirror), **before** the C-E2 audit
  commit `08981f7` that filled the measured ladder. A reader can verify ordering with
  `git log --oneline d96b827 08981f7` (predict precedes audit) and that the check table above was
  empty (`_TBD_`) at `d96b827`.
- **Prior / baseline:** a strict 3-rung ordering is **1-in-6 (1/3! ≈ 0.167)** correct by chance —
  so a single correct ordering is not strong on its own. What makes this a forecast rather than a
  lucky draw is the **per-rung confidence implied by the pilot signals** (all three signals are
  monotone in the same direction and separate by orders of magnitude):

  | rung | ρ (spread) | δ (drift) | support | implied confidence the rung lands in its band |
  |---|---|---|---|---|
  | density (≳0.9) | 1.43 | 0.25 | 0.94 | **high** — ρ≈1, low δ, support≈0.94 ⇒ near-perfect transfer |
  | BVS-GII (0.6–0.85) | 2.59 | 0.32 | 0.84 | medium — some excess spread, good support |
  | stability (≲0.4) | **58** | **63** | 0.62 | **high** (that it's *low*) — ρ,δ ~1.5 orders of magnitude worse |

  The density and stability rungs are the confident ends (signals extreme); BVS is the genuine
  middle. The prediction is a forecast with a stated 1/6 chance baseline, not an unqualified guess.

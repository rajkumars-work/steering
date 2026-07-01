# Claim 1 (crystals): does the data-audit budget split survive the model?

**Date:** 2026-06-18
**Model:** `alex_nolemat_lowhull` ep120 (`/opt/dlami/nvme/recast/train/mix_ep120_ckpt`, version `d15_binrho_k7`)
**Driver:** `claim1_survival_spectrum.py` (this dir) · **Output:** `/opt/dlami/nvme/recast/train/claim1_survival_spectrum.json`

The crystal mirror of the ImageNet brightness experiment, for the paper's **Claim 1**: a
within/between budget split read off the **data** should be reproduced by the **model's own
outputs** — *but only to the extent the model has converged on the property.* If the model can't
reliably produce a property, its statistics won't be preserved; that's a model limitation, not a
hole in the framework. So instead of one property we walk a **difficulty ladder** and watch the
agreement degrade.

## Headline

Across a property ladder of increasing difficulty, **per-chemistry agreement degrades exactly as
predicted** — the data audit predicts the model where the model is good, and progressively fails
where the model is poor:

| property (easy → hard) | what it tests | per-chemistry mean **r²** (data vs model) |
|---|---|---|
| **density** | geometry (mass / cell volume) | **0.991** |
| **BVS-GII** | bond-valence self-consistency | **0.833** |
| **e_above_hull** | thermodynamic stability | **0.296** |

The realization is graceful, not binary: the model nails density, mostly tracks bond-valence
self-consistency, and only loosely reproduces stability.

## (A) Convergence ladder — pass rate (data vs model)

How often each check passes, averaged over chemistry bins. The model's rate falls monotonically
as the check gets harder:

| check | data | model |
|---|---|---|
| `smact_pass` (composition charge-balance) | 0.85 | 0.83 |
| `bvs_pass` (bond-valence assignable) | 0.93 | 0.83 |
| `validity_pass` (SMACT ∧ BVS ∧ min-dist) | 0.80 | 0.69 |
| `metastable` (e_above_hull ≤ 0.1) | 0.99 | 0.67 |

The model is essentially at parity on composition (SMACT), slips on geometry-dependent checks
(BVS, validity), and falls furthest on stability — the property generative models are worst at.

## (B) Budget survival — within (T) / between (E) split + r²

For each *continuous* property, the within-chemistry variance `T` (mean of per-bin variances) and
the between-chemistry variance `E` (variance of per-bin means), read off the data and recomputed on
the model's own outputs, on **one measure per property for both sides**:

| property | bins | T_data | E_data | T_model | E_model | E_model/E_data | **r²** |
|---|---|---|---|---|---|---|---|
| **density** (g/cm³) | 24 | 0.450 | 8.04 | 0.911 | 9.65 | **1.2×** | **0.991** |
| **BVS-GII** (v.u.) | 15 | 0.0036 | 0.425 | 0.065 | 0.471 | **1.1×** | **0.833** |
| **e_above_hull** (eV/atom) | 22 | 9.6e-5 | 2.8e-4 | 0.808 | 0.613 | **~2200×** | **0.296** |

Reading the table:

- **density** — the between-chemistry budget is essentially reproduced (E 8.04 → 9.65), per-bin
  means align almost perfectly (r²=0.99). The model has fully converged on geometry, so the data's
  density budget survives intact. This is the crystal analog of the image result's "total +
  dominant part carries through."
- **BVS-GII** — still strong: the between-chemistry budget is reproduced to ~10% (E 0.425 → 0.471)
  and per-bin means track well (r²=0.83). The model mostly respects bond-valence self-consistency,
  with some excess within-bin scatter (T 0.0036 → 0.065).
- **e_above_hull** — the hard end. The data's between-chemistry spread is **tiny** (E=2.8e-4)
  because the training set was *curated on low hull energy* — every data structure is near the hull
  by construction, so there is almost no between-chemistry stability budget to reproduce. The model
  produces a far broader, noisier stability distribution (E_model 0.61, ~2200× larger) and only
  loosely tracks per-chemistry means (r²=0.30). Both effects — the model's weaker convergence on
  stability **and** the curation-flattened data spread — push the same way.

A standalone, higher-resolution stability run (`claim1_survival_crystals.py`, `EAH_MAX=5.0` filter,
n=230 data / 235 generated over 23 bins) corroborates the hard end: DATA T/E/V =
9.3e-5 / 2.7e-4 / 3.6e-4, MODEL = 0.395 / 0.910 / 1.305, r²=0.303.

## The point

This turns the framing into a *measured* claim: the data audit predicts the model **exactly where
the model is good** (density, r²=0.99), **partly where it's so-so** (BVS, r²=0.83), and **not where
it's poor** (stability, r²=0.30). The stability end is also curation-confounded (the data was
selected on low hull, so its native stability budget is artificially narrow), which is a second,
independent reason its statistics don't transfer. The degradation is the result — a single number
on any one property would have been the wrong test.

## Methodology

- **Bins** = chemistry `(element set, atom count)`; `K_BINS=24` bins with `≥ MIN_MEMBERS=10` real
  structures, sampled at `SEED=0`. Per bin: `M_DATA=10` real structures and `M_GEN=12` generated.
- **One measure per property, both sides.** Data structures are decoded from the training row with
  the **same** `parse_target` decoder the model's outputs go through, then every property is
  computed the identical way for data and generated structures:
  - **density** = mass / cell-volume (pure geometry; no model, no hull).
  - **SMACT / BVS-pass / validity** = `chem.validity.check_validity` (SMACT charge-balance,
    bond-valence assignability, min-distance) in one call.
  - **BVS-GII** = the Global Instability Index: per site, the real-valued bond-valence sum from
    bond lengths (`pymatgen … calculate_bv_sum`, Brown–Altermatt) minus the formal oxidation state,
    RMS over sites. Continuous (0 = perfectly self-consistent); `None` for pure intermetallics
    (no anion) — those just don't contribute to the BVS rung (15 of 24 bins qualify).
  - **e_above_hull** = single-MACE single-point vs a MACE-reevaluated MP hull (`chem.stability`),
    filtered to finite `|eah| ≤ 5.0` eV/atom (above that = an exploded/failed generation, not a
    real phase — dropping these keeps the variance budget meaningful).
- **`E_model` de-noised**: for the standalone stability run, `E_model = E_naive − T_model/M_GEN`
  removes finite-sample inflation of the between-bin variance.

### Note — why BVS needed a real metric (a fixed bug)

The first spectrum run reported `bvs_dev` (from `check_validity`'s `bvs_max_deviation`) as the BVS
signal and it came out **all zeros → r²=nan**. Cause: `bvs_max_deviation` is
`|valence − round(valence)|` computed on the **integer** oxidation states `BVAnalyzer.get_valences()`
returns — structurally ≈0, so it carries no continuous information. Replaced with the **GII**
above (real-valued bond-valence sums from geometry), which gives the genuine continuous
self-consistency signal (data oxides ≈ 0.22–0.37 v.u.; metals → `None`). The all-zero result is
preserved at `claim1_survival_spectrum.bvsdev_nan.json`.

## For framing parity — the image result this mirrors (in the paper)

Brightness, 40 classes, raw conditional: data within/between/total 0.0130 / 0.0032 / 0.0162; model
0.0177 / 0.0017 / 0.0194 — the total and the dominant within-bin part carry through, guidance
reweights the small between-bin part. Crystals show the same "carries through where the model has
converged," now resolved along a difficulty axis.

## Artifacts

| file | what |
|---|---|
| `claim1_survival_spectrum.py` | the difficulty-ladder driver (ladder + survival, this result) |
| `claim1_survival_crystals.py` | density-only fast view + the standalone filtered stability run |
| `/opt/dlami/nvme/recast/train/claim1_survival_spectrum.json` | tables (A) + (B) |
| `/opt/dlami/nvme/recast/train/claim1_survival_crystals.json` | standalone filtered stability (hard-end corroboration) |
| `…/claim1_survival_spectrum.bvsdev_nan.json` | the superseded all-zero-BVS run (kept for the record) |
| `…/claim1_survival_crystals.RAW_unfiltered.json` | the pre-filter stability run (inf-poisoned; kept for the record) |

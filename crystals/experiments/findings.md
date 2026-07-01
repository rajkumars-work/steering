# Within-distribution steering: Combined is chemistry-steerable from 0.03 → 0.82

**Date:** 2026-06-16
**Model:** `alex_nolemat_lowhull` ep120 (`/opt/dlami/nvme/recast/ckpts/ckpt_alex_nolemat_lowhull`, backup `ed/checkpoints/.../ed_ckpt_e120.pt`)
**Scorer:** single-MACE Combined (`chem.stability.compute_e_above_hull` [MACE single-point vs MACE-reevaluated MP hull] + lemat-genbench `SUNMetric` over LeMat-Bulk). Combined = SUN + MSUN, over n_total (no validity gate — validity is a separate metric).
**Pool source:** the **(E,N) distribution Counter** of the `alex_nolemat_lowhull` eval split (natoms<20) — `an_lh.dist.json`, 12,917 unique (element-set, natoms) tuples. No dataset rows used; verified earlier that drawing from the Counter reproduces the pool's Combined (Counter-draw 0.680 vs direct-draw 0.646, within noise).

## Headline result

Within the an_lh distribution (whole-pool Combined ≈ 0.65), **Combined is steerable across ~0.03 → 0.82 purely by selecting (E,N) chemistry**, and the extremes are physically interpretable:

| extreme | Combined (validated n=120) | SUN / MSUN | chemistry | breadth |
|---|---|---|---|---|
| **HIGH** | **0.824** | 0.118 / 0.706 | heavy **rare-earths** (Y, Ho, Tb, Er, Tm, Dy, Sm, Ce) | **broad**: 863 (E,N), ~60 elements |
| **LOW (broad)** | ~0.08 (n=24) | — | **H + Pt/Rh/Pd/Ni** (hydrides + platinum-group) | broad: 84 (E,N) |
| **LOW (narrow floor)** | **0.033** | 0.017 / 0.017 | H/Pt/Rh/Mn niche | narrow: 10 (E,N) |

- **HIGH is mostly metastable** (MSUN 0.71 ≫ SUN 0.12): the model makes many novel, near-hull rare-earth compounds, few strictly on-hull.
- **Breadth is lopsided**: the high extreme is broad & reusable (rare-earths are a large model-friendly family in Alexandria); broad low pools bottom at ~0.08–0.21, and reaching 0.033 requires a narrow 10-tuple niche.

## Methodology

Goal: find the highest- and lowest-Combined **sub-pools within one distribution** (not across datasets), pushing hard to the extremes. Premise: Combined is set by the pool's (E,N) chemistry, so chemistry sub-regions span a wide Combined range.

1. **Cluster** the 12,917 (E,N) tuples into **K=64 chemistry sub-pools** via KMeans (features = element-presence one-hot + natoms/40; `random_state=0, n_init=4` → deterministic/reconstructable).
2. **Scan**: score every sub-pool empirically — sample 24 prompts ∝ count, generate (top_k=10), score single-MACE Combined. → the full spread.
3. **Push to each extreme** (greedy, separate high & low): start from the **union of the top-3 / bottom-3 scan pools** (robust vs a single fluke pool), re-cluster into 6, keep the best/worst sub-bin, repeat up to 6 rounds or until the pool drops below ~24 (E,N).
4. **Validate** each final extreme at **n=120** (winner's-curse control).

**Compute:** scan 64×24 ≈ 1,536 structures; pushes + validations ≈ 1,000 more; ~4–5 GPU-hours total at ~7 s/structure (generate + single-MACE).

## Methodology caveats (so the numbers are read correctly)

- **Winner's-curse is real here.** The scan's top rare-earth pools hit 0.92–0.96 at n=24; the broad pool validates at **0.82** at n=120 (honest value). And the greedy HIGH push *over-refined* into a size-17 sub-bin that scored 1.0 @ n=24 and regressed to 0.675 @ n=120 — **deep greedy refinement HURT at the high end.** The trustworthy reads are the **scan + broad-pool validation**, not the deepest greedy pick.
- The LOW greedy push behaved well (491→79→10 (E,N), validated 0.033), but that floor is a narrow niche; the broad low anchor is ~0.08.
- **Greedy ≠ global optimum** — these are well-pushed local extremes, not proven optima.
- Single-MACE scorer (validated against the documented nxn matrix; sidesteps the orb-`_omat` hull mis-calibration — use orb `_mpa` for 3-MLIP).

## Artifacts (this directory)

| file | what |
|---|---|
| `pools/HIGH_rareearth.json` | the 863-(E,N) rare-earth HIGH pool (validated Combined 0.824) |
| `pools/LOW_broad_HPtRh.json` | the 84-(E,N) broad H/Pt/Rh LOW pool (~0.08) |
| `scan_64pools.json` | full 64-pool scan: per-pool Combined, size, top elements |
| `push_HIGH.json`, `push_LOW.json` | greedy refinement trajectories |

Distribution Counter: `dist_verify/distributions/an_lh.dist.json` (also in the verification bundle `/opt/dlami/nvme/lemat_verify/lemat/data/distributions/`).
Note: the exact 10-tuple narrow-LOW pool's (E,N) were not persisted by the run (only its chemistry signature H/Pt/Rh/Mn/In/Ho/Nd/Pr/Mg/Ni); the broad-LOW pool above is the recorded, reusable low anchor.

## How to reproduce a pool's Combined

```bash
# sample n prompts ∝ count from a pool JSON, format "elements | natoms | ", then:
python ed/nxn/steer_experiment.py --pools <dir> --ckpt <alex_nolemat_lowhull dir> \
    --version d15_binrho_k7 --n 120 --out <out>
```
Driver that produced this: `/opt/dlami/nvme/recast/train/binning_experiment.py`.

## Hitting intermediate (0.1-increment) targets — two methods

Per the steering paper (`docs/steering_paper/application.v2.md`, §"Two ways"), a target Combined
`g*` can be reached two ways:

**Method 1 — pool mixing (done).** Blend the HIGH (0.85) and broad-LOW (0.14) anchors at
`lambda = (g* - 0.14)/(0.85 - 0.14)`. Linear; the dial tracks the target. Measured ladder
(n=100, single-MACE):

| target | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|--------|-----|-----|-----|-----|-----|-----|-----|
| achieved | 0.23 | 0.34 | 0.47 | 0.57 | 0.65 | 0.73 | 0.85 |

Hit within ~±0.05; run-to-run std ~0.02–0.03 (3 reps) ≤ binomial SE. Reachable range ~0.15–0.85
(broad LOW floor). Shipped as `generate.py --target` in the repos. Cost: "aggressive" — piles
weight on the two extreme corners (higher χ²). Data: `mix_ladder/results.json`.

**Method 2 — natural pools (the gentler push; preferred per the paper's min-χ²).** Instead of
mixing extremes, pick a chemistry bin that *already* sits at the target. The 64-pool scan
contains a bin essentially on every 0.1 target (max |err| 0.033 @ n=24):

| target | bin | chemistry (top elems) | scan C (n=24) |
|--------|-----|-----------------------|---------------|
| 0.1 | 60 | H, Pt, Rh | 0.083 |
| 0.2 | 1  | Rb, O, K, S | 0.208 |
| 0.3 | 37 | U, N, Np, Nb | 0.333 |
| 0.4 | 53 | Zr, H, Nb, Rh | 0.391 |
| 0.5 | 20 | H, Sr, Tb, Fe | 0.500 |
| 0.6 | 22 | Au, Al, Pd, Mg | 0.609 |
| 0.7 | 6  | Sm, Nd, Dy, Er | 0.708 |
| 0.8 | 15 | Pm, Y, Pr, Sm | 0.792 |

Coverage is redundant (multiple bins within ±0.05 of most targets). Each bin saved as a reusable
pool: `natural_pools/pools/natural_T0X.json`. **Validated at n=100** (`natural_pools/results.json`):

| target | bin | chemistry | validated | err |
|--------|-----|-----------|-----------|-----|
| 0.1 | 60 | H,Pt,Rh | 0.14 | +0.04 |
| 0.2 | 1  | Rb,O,K,S | 0.29 | +0.09 |
| 0.3 | 37 | U,N,Np,Nb | 0.43 | **+0.13** |
| 0.4 | 53 | Zr,H,Nb,Rh | 0.38 | −0.02 |
| 0.5 | 20 | H,Sr,Tb,Fe | 0.34 | **−0.16** |
| 0.6 | 22 | Au,Al,Pd,Mg | 0.54 | −0.06 |
| 0.7 | 6  | Sm,Nd,Dy,Er | 0.67 | −0.03 |
| 0.8 | 15 | Pm,Y,Pr,Sm | 0.76 | −0.04 |

Most within ~0.06, but **two regressed (0.3→0.43, 0.5→0.34) — winner's-curse from the n=24 scan
picks**. So a natural single-bin gives a *coherent chemistry* but **coarse placement that needs
n=100 validation**; mixing (the calibrated dial) is the more reliable target-hitter.

**Correction (vs an earlier note):** a single *small* natural bin is **not** gentler than mixing —
concentrating all weight on one small bin is itself an aggressive push (see χ² below). The
genuinely gentle option is the paper's **min-χ²** recipe (weight spread over *all* near-target
bins, closed form) — **not yet implemented**.

### Method comparison at target 0.5 — MIX vs NATURAL (`method_compare/comparison.json`)

| | MIX (λ=0.51, HIGH+LOW) | NATURAL (bin 20) |
|---|---|---|
| Combined (3×n=100) | **0.49 ± 0.02** (on target) | **0.34 ± 0.04** (bin missed; scan estimate was noise) |
| chemistry (top elems) | **bimodal**: Pt,H + Y,Ho,Tb,Dy,Tm,Er | **unimodal**: H,Sr,Tb,Fe,Mn,Cr,Ni,O |
| bond-valence assign rate | **0.14** (exotic juxtaposition) | **0.50** (chemically coherent) |
| capacity (unique EN; dup@5%) | HIGH 863 (dup@83) / **LOW 84 (dup@10)** | 148 (dup@17) |
| χ² (push) | **40.9** | **87.6** |
| natoms mean | 11.9 | 10.9 |

Takeaways: (1) **mixing is visibly bimodal** (two extreme chemistries coexist) while the natural
bin is one coherent, far-more-BV-assignable family; (2) **mixing's LOW corner is thin** (84 (E,N),
duplicates by ~10 draws — the paper's thin-corner warning).
Driver: `method_compare_T05.py`. (Caveat: `mean_e_above_hull` is `inf` — one divergent structure
poisoned the mean; SUN/Combined unaffected.)

### χ² (push) done right — min-χ² is the gentle method

The single-bin and mixing χ² above are both **large** because both are concentrated pushes — a
single ~1%-of-data bin (χ² ≈ 1/J_P(bin) − 1 ≈ 86) and a mix leaning on a *thin* one-bin LOW corner.
**Neither is the gentle regime.** The paper's gentle recipe is **min-χ²** — weight spread over the
bins already near the target — with `χ²_min(g*) = (g*−ḡ)²/Var_J(g)` (natural over the 64 bins:
ḡ=0.604, Var=0.039):

| target | **min-χ² (spread)** | mixing (thin LOW) | single small bin |
|--------|--------------------|-------------------|------------------|
| 0.2 | **4.1** | 128 | ~86 |
| 0.3 | **2.3** | 92  | ~86 |
| 0.4 | **1.1** | 63  | ~86 |
| 0.5 | **0.27** | 40 | ~86 |
| 0.6 | **0.00** | 24 | ~86 |
| 0.7 | **0.24** | 15 | ~86 |
| 0.8 | **0.98** | 13 | ~86 |

So **min-χ² ≪ mixing** (≈150× smaller at 0.5), minimized at the natural mean and rising only to ~4
at the extremes — confirming the paper's claim that min-χ² is the gentlest push. **A lone small
natural bin is NOT a substitute for min-χ²** (it's a large push). And our *mixing* is extra-costly
because the LOW anchor is a thin one-bin corner (paper's §4.6); broad quantile corners would lower
it toward the paper's ~0.6. **Min-χ² generation (sampling the spread recipe) is not yet
implemented** — only its push (cost) is computed here. Driver for these χ²: inline (analytic over
`scan_64pools.json`).

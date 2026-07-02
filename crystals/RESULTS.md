# CUES crystal track — results & reproduction (review.1)

Repo-side reconciliation + reproduction guide for the crystal claims. The **authoritative running
write-up with the full per-experiment 95%-CI tables** is the coordination file
`dielectric/docs/steering_paper/crystal_experiments.results.md`; this file mirrors it and adds the
from-the-repo reproduction recipe.

## Environment
- venv: `requirements.txt` (full freeze). Key pins: torch 2.7/cu128, mace-torch 0.3.15,
  pymatgen 2026.5.4, matminer 0.10.1, xgboost 3.2.0, scikit-learn 1.8.0, SMACT 4.0, ase 3.28;
  lemat-genbench + the ed/dielectric code are bundled under `lemat/code/py/`.
- Model: CUES `alex_nolemat_lowhull` ep120, sha256 `9b8e4b2e…54177` (`ARTIFACTS.md`), fetched via
  `fetch_checkpoint.sh` from HF — **not committed**.
- Hardware: single A10G for generation + single-MACE scoring.

## Recorded vs measured (review.1)
| number | recorded | measured this round (95% CI) | status |
|---|---|---|---|
| density carry-over r² | 0.99 | **0.985 [0.906, 0.998]** (C-E2, 3 seeds, m=30) | ✅ confirmed |
| BVS-GII carry-over r² | 0.83 | **0.879 [0.035, 0.972]** (C-E2; wide CI — few ionic chem) | ✅ confirmed |
| stability carry-over r² | 0.30 | **0.488 [0.172, 0.706]** (C-E2) | ⚠ **higher than recorded — cite 0.49** |
| DPO held-out log-prob (fine-tuned) | ≈ −17 | **−15.10 [−15.20, −14.99]** /tok (n=250) | ✅ close (stand-in ckpt) |
| DPO − uniform separation | < −9.7 | **100% [100, 100]** of structures below −9.7 | ✅ |
| Claim-3 joint: telling-naive | 0% | **2.7% [1.7, 4.3]** (6 seeds, n=594) | ✅ ~recorded shape |
| Claim-3 joint: telling-strong | (new baseline) | **15.5% [12.5, 19.1]** (6 seeds, n=470) | ⚠ strong baseline |
| Claim-3 joint: showing | 36% | **13.2% [10.5, 16.5]** (6 seeds, n=484) | ⚠ **ties telling-strong (measured); below 36% (surrogate gap)** |
| Claim-1 stability r² (ensemble label) | — | **0.42 [0.23, 0.66]** (C-E9) ≈ single-MACE 0.49 | ✅ doesn't carry under cleaner label |
| Claim-3 frontier coverage (gap∧ε₀) | — | showing **0.40 [0.35,0.44]** ≫ conditioning 0.02 ≈ telling 0 | ✅ **outright win (C-E12)** |

Honest flags carried into the write-up: the DPO figure uses a **stand-in** DPO checkpoint
(IPO-novelty ckpt lost with the dead box) and base self-generations (base dist pinned ≈0); C-E3's
gap is a **composition surrogate**, not the structural MLIP. See the coordination write-up.

## Reproducing the review.1 experiments

**Common prerequisites.** (1) `pip install -r requirements.txt` and activate the venv. (2) Fetch
the checkpoint: `./fetch_checkpoint.sh` → `checkpoints/alex_nolemat_lowhull/` (hash-verify vs
`ARTIFACTS.md`). (3) `export HF_TOKEN=…` (LeMat-Bulk novelty ref) and `export MP_API_KEY=…` (MP
hull references for single-MACE stability). (4) The **DATA side** of C-E2/C-E3/C-E4 needs the
Alexandria-derived training CSV — `export CUES_TRAIN_CSV=/path/to/alex_nolemat_lowhull_dataset_train.csv`
(the subset is specified in `MODEL_CARD.md`: Alexandria, natoms<20, low-hull, LeMat excluded; it is
a derived artifact, **not** shipped in this data-light repo). All scripts resolve paths
repo-relative via `review1/_repo.py` — **no hardcoded working-dir or volatile paths**.

| exp | script | run command | needs |
|---|---|---|---|
| **C-E1** (CIs on saved numbers) | `review1/ce1_bootstrap_existing.py` | `python review1/ce1_bootstrap_existing.py` | **CPU only**; reads `results/{claim1_survival_crystals_density,dpo_logprob_fig}.json` |
| **C-E4** (pilot → ladder prediction) | `review1/ce4_pilot.py` | `bash review1/run_ce4pilot.sh` | GPU, checkpoint, `CUES_TRAIN_CSV`, MP_API_KEY |
| **C-E2** (converge ladder, 3 seeds, m-sweep) | `review1/ce2_converge.py` | `bash review1/run_ce2.sh` | GPU, checkpoint, `CUES_TRAIN_CSV`, MP_API_KEY (single-MACE hull) |
| **C-E3** (joint baselines) | `review1/ce3_joint.py` | `bash review1/run_ce3.sh` | GPU, checkpoint, `CUES_TRAIN_CSV`, MP_API_KEY |

Notes:
- **C-E3 gap = composition XGBoost DFT-band-gap surrogate** (`data/xgb_composition_dft_band_gap.json`,
  shipped), **not** the heavy structural MLIP / Dielectrics (Ray) stack — that needs relaxation +
  phonons + multiple GPU actors and is impractical to run inline. Stability is single-MACE
  e_above_hull. The dataset is all-low-hull, so the pools discriminate stability by the `hull-vlow`
  **tag** (most-stable bin), giving `joint ⊂ widegap ⊂ full`.
- **C-E4 is pre-registered**: `review1/ce4_prediction.md` was committed *before* C-E2's audit (see
  git history) — the predicted-vs-measured check is genuinely out-of-sample.
- Outputs land beside each script (`review1/*.json`, `*.log`); the C-E7 model card is `MODEL_CARD.md`.

## Reproducing the round-2 experiments
| exp | script | run command | needs |
|---|---|---|---|
| **C-E8** key sensitivity | `review1/ce8_keys.py` | `python review1/ce8_keys.py` | **CPU**; reads `results/ce2_peritem.json` |
| **C-E9** reliability | `review1/ce9_reliability.py` | `python review1/ce9_reliability.py` | **CPU**; reads `results/ce2_peritem.json` |
| **C-E9** ensemble | `review1/ce9_ensemble.py` | `CUES_TRAIN_CSV=… python review1/ce9_ensemble.py` | GPU, ckpt, train CSV, MP key; loads a multi-MACE ensemble |
| **C-E3** +seeds | `review1/ce3_seeds345.py` | `CUES_TRAIN_CSV=… python review1/ce3_seeds345.py` | GPU, ckpt, train CSV, MP key (pool with `ce3_joint.json` → 6 seeds) |
| **C-E12** frontier coverage | `review1/ce12_coverage.py` | `CUES_TRAIN_CSV=… python review1/ce12_coverage.py` | GPU (gen only, **no MACE**); gap+ε₀ = composition surrogates in `data/` |
| **C-E12** figures | `review1/ce12_figure.py` | `python review1/ce12_figure.py` | CPU; reads `results/ce12_coverage.json` |

## Reproducing the round-6 experiments
Tier-1 = data-side / re-analysis (CPU, no checkpoint); tier-2 = needs GPU + checkpoint.
| exp | script | run command | needs |
|---|---|---|---|
| **C-E16** strongest (best-of-N) knob | `review1/ce16_strongest_knob.py` | `bash review1/run_ce16_super.sh` (detached supervisor) | **tier-2**: GPU, ckpt, `CUES_TRAIN_CSV`, MP key (single-MACE); reads `review1/ce14_panel.json` |
| **C-E17** stability ratio bound | `review1/ce17_stability_bound.py` | `python review1/ce17_stability_bound.py` | **tier-1 CPU**; reads `results/ce14_panel.json` |
| **C-E17** verify | `review1/verify_ce17.py` | `python review1/verify_ce17.py` | CPU; checks the ≥24× floor + discount self-consistency |
| **C-E18** key-selection diagnostic | `review1/ce18_key_selection.py` | `python review1/ce18_key_selection.py` | **tier-1 CPU**; reads `results/ce8_keys.json` + `results/ce14_panel.json` → `figures/fig_ce18_key_selection.png` |
| **C-E18** verify | `review1/verify_ce18.py` | `python review1/verify_ce18.py` | CPU; checks E/(E+T)→ratio monotonicity + recipe argmax |

Notes:
- **C-E17/C-E18 are pure re-analysis** of C-E8/C-E14 — they ship in the data-light repo with no GPU.
- **C-E16** re-uses the C-E14 showing side (`bin_shift`); its supervisor retries on contended-GPU
  windows (exit 75) and mirrors the C-E14 stability harness.

## Status (round 1 + 2 — all complete)
R1: C-E7 ✅ · C-E2 ✅ · C-E4 ✅ (vindicated) · C-E1 ✅ (density/ladder/DPO/joint/Combined-sweep;
Claim-1 split blocked, data off-box) · C-E3 ✅.
R4: C-E12 ✅ (frontier coverage — showing 0.40 ≫ conditioning≈telling≈0; crystal Claim 3 = outright win, pre-registered f362a48). Round 3 superseded (not run).
R2: C-E8 ✅ (ordering stable across keys) · C-E9 ✅ (ensemble r²=0.42 ≈ single 0.49 → stability
genuinely doesn't carry; **Claim 1 strengthens**) · C-E4-artifact ✅ · scorer hygiene ✅ · C-E3
+seeds ✅ (6-seed tie measured). Live CI tables: the coordination write-up.
R6: C-E16 ✅ (best-of-12 steel-man: chemistry still wins 26×/14×/4.75× gap/density/stability; budget
ceiling holds at every real k; pre-registered) · C-E17 ✅ (stability floor ≥24× under weakest carry-over)
· C-E18 ✅ (E/(E+T) predicts realized ratio, Spearman=1; key-selection recipe). Verify scripts:
`verify_ce16/17/18.py`.

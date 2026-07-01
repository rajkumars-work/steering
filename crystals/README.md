# CUES steering — generate crystals to a target Combined score, and verify it

Self-contained package to **generate crystals steered to a target Combined (SUN+MSUN) score**
and **score them**, reproducing the steering result from the paper. Four pieces.

> **This is the minimal verification repo** — checkpoint + scripts + verification code only.
> A separate, comprehensive repo (full training/research source + datasets) is published elsewhere.

1. **a checkpoint** — `checkpoints/alex_nolemat_lowhull/` (the steering model + tokenizer)
2. **two distributions** — `lemat/data/distributions/HIGH_rareearth.json` (Combined ≈ 0.82) and
   `LOW_broad_HPtRh.json` (broad Combined ≈ 0.14); mixing them sets the target
3. **`generate.py --target T`** — generate crystals steered to Combined ≈ T (T in 0.1–0.8)
4. **`score.py`** — compute the Combined score of generated crystals

```bash
python generate.py --target 0.5 --n 100 --out gen.extxyz   # steer to Combined ~0.5
python score.py gen.extxyz                                  # -> COMBINED: 0.5x
```

## How steering works
Combined is set by the prompt pool's (element-set, natoms) **distribution**. We ship two
anchor distributions — a HIGH pool (rare-earth chemistry, Combined ≈ 0.82–0.85) and a LOW pool
(H/Pt-group chemistry, broad Combined ≈ 0.14). `generate.py` mixes them at
`lambda = (target - 0.14) / (0.85 - 0.14)`: it draws `lambda*n` naked prompts
(`elements | natoms |`) from HIGH and the rest from LOW, so the expected Combined ≈ target.
No dataset is needed — only the two distribution Counters (verified: drawing from the Counter
reproduces the pool's Combined).

## Setup (one environment)
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export HF_TOKEN=hf_...        # LeMat-Bulk novelty reference auto-downloads (public)
export MP_API_KEY=...         # free Materials Project key, for the stability hull references
```
Generation (EdGPT) and single-MACE scoring run in this one environment. Code is bundled under
`lemat/code/py/` (ed model, dielectric eval/chem, lemat-genbench + material_hasher) and the
scripts put it on `sys.path` automatically.

## What `score.py` computes
`Combined = SUN + MSUN` over all generated structures (no validity gate):
- **SUN** = stable (`e_above_hull ≤ 0`) ∧ novel (vs LeMat-Bulk) ∧ unique
- **MSUN** = metastable (`0 < e_above_hull ≤ 0.1`) ∧ novel ∧ unique
- **Stability**: single MACE single-point energy vs a Materials-Project reference hull
  (references re-evaluated with the same MACE → self-consistent; cached per chemical system).
- **Novelty/uniqueness**: lemat-genbench `SUNMetric` against LeMat-Bulk.

This single-MACE scorer is what produced every steering number in the paper.

> **Benign warnings.** On first run `score.py` may print that `EquiformerV2Similarity`
> (fairchem) is unavailable and that `cuequivariance` acceleration is disabled. Both are
> **optional** — a similarity feature and a speedup we don't use — and neither is on the
> SUN/Combined path. Ignore them; the Combined score is unaffected.

## Accuracy / how close targets are hit (measured)
Mixture-ladder calibration, n=100/point, single-MACE, anchors HIGH≈0.85 / broad-LOW≈0.14:

| target   | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|----------|-----|-----|-----|-----|-----|-----|-----|
| achieved |0.23 |0.34 |0.47 |0.57 |0.65 |0.73 |0.85 |

Monotonic; targets hit within ~0.05 (a uniform offset from the anchor estimate, now corrected
in the HI/LO defaults above). **Run-to-run variance (3 replicates): std ~0.02-0.03 at n=100** —
at/below the binomial SE (~0.05). Reachable range with the broad LOW is ~0.15-0.85; **target 0.1
saturates at ~0.15** (use a lower LOW pool for a true 0.1). Raw data:
`lemat/data/target_calibration.json`. Increase `--n` for tighter results.

## Reproduce the experiments (not just generate)
`experiments/` holds the runnable drivers behind the paper's steering numbers — the **discovery**
scan that finds the extremes, both target-hitting methods, the method comparison, and the Claim 1
budget-survival test. Each runs from a fresh clone (paths resolved relative to the repo by
`experiments/_repo.py`), in the same one environment as `score.py`:

```bash
python experiments/scan_extremes.py     # cluster -> scan -> push to extremes (the 0.03->0.82 result)
python experiments/natural_pools.py     # Method 2: a natural bin per 0.1 target
python experiments/method_compare.py    # MIX vs NATURAL at target 0.5
# Method 1 (mixing) is just generate.py --target T (see experiments/README.md for the ladder loop)

# Claim 1 — does the data-audit budget split survive the model? (needs the dataset)
CUES_TRAIN_CSV=/path/to/alex_nolemat_lowhull_dataset_train.csv \
    python experiments/claim1_survival_spectrum.py   # density -> BVS-GII -> e_above_hull ladder
```
See `experiments/README.md` for outputs, costs, and the determinism/winner's-curse notes;
`experiments/findings.md` for the steering write-up; and `experiments/claim1_survival_findings.md`
for the Claim 1 budget-survival result (per-chemistry agreement degrades 0.99 → 0.83 → 0.30 as the
property gets harder). The full (E,N) Counter the scan clusters ships at
`lemat/data/distributions/an_lh.dist.json` (12,917 tuples).

## Claim 1: does the budget survive the model?

The crystal mirror of the ImageNet experiment. A within/between budget split read off the **data**
should be reproduced by the **model's own outputs** — *but only to the extent the model has
converged on the property.* We walk a difficulty ladder on the same generated structures and watch
the per-chemistry agreement degrade:

| property (easy → hard) | per-chemistry mean r² (data vs model) |
|---|---|
| density (geometry) | **0.99** |
| BVS-GII (bond-valence self-consistency) | **0.83** |
| e_above_hull (stability) | **0.30** |

A graceful, *measured* claim: the audit predicts the model where it's good (geometry), partly where
it's so-so (bonding), and not where it's poor (stability) — the last also curation-confounded
(training data was selected on low hull, so its native stability budget is artificially narrow).

- **Scripts** (need the dataset for the DATA side — point `CUES_TRAIN_CSV` at the
  `alex_nolemat_lowhull` training CSV from the comprehensive repo):
  - `scripts/claim1_survival_spectrum.py` — the main run (convergence ladder + budget survival).
  - `scripts/claim1_survival_crystals.py` — density-only fast view (no MACE, minutes).
- **Results**: `results/claim1_survival_spectrum.json` (both tables) and
  `results/claim1_survival_crystals.json` (standalone filtered stability, hard-end corroboration).
- **Write-up**: `claim1_survival_findings.md` (full numbers, methodology, and the BVS-GII metric).

```bash
CUES_TRAIN_CSV=/path/to/alex_nolemat_lowhull_dataset_train.csv \
    python scripts/claim1_survival_spectrum.py
```
(Portable, repo-relative copies of these drivers also live under `experiments/`.)

## Paper figures (`figures/`)

Regenerate the crystal-domain figures from the shipped results (numpy + matplotlib only, no
GPU) — except the DPO one, which needs checkpoints:

| script | figure | shows | input |
|---|---|---|---|
| `scripts/make_figures.py` | `fig_claim1_survival_ladder.png`, `fig_claim1_convergence_ladder.png` | the Claim 1 budget-survival r² ladder (0.99→0.83→0.30) and the convergence pass-rate ladder | `results/claim1_survival_spectrum.json` |
| `scripts/plot_pred_vs_obs.py` | `fig_recast_pred_vs_obs.png` | Claim 4 plain-pretraining diagonal: per-chemistry predicted (data) vs observed (model) **density**, r²=0.99 | `results/claim1_survival_crystals_density.json` |
| `scripts/dpo_logprob_fig.py` | `fig_dpo_logprob_dist.png` | Claim 4.4: per-token log-prob of held-out base structures under base vs aggressively fine-tuned (DPO) model — fine-tuned mass (≈ −15) below the −9.7 uniform line ⇒ off the base support | base + DPO checkpoints (see below) |

```bash
python scripts/make_figures.py          # spectrum ladders
python scripts/plot_pred_vs_obs.py      # density diagonal (pure plot from results/)
# DPO figure needs the checkpoints (not in this data-light repo):
CUES_BASE_CKPT=/path/to/production_ckpt_dir CUES_DPO_CKPT=/path/to/dpo_ckpt_final.pt \
    python scripts/dpo_logprob_fig.py
```

**DPO-figure provenance caveat.** The paper's intended source is the IPO-novelty checkpoint,
which was lost with the original box (its dir is empty, its training data was on the dead
`/data/assets/atlas`). The shipped figure uses **base = production** and **DPO = a stand-in**
(`checkpoints_dpo_ox`). Held-out base structures are the base model's *own* generations (so
they're base-distribution and correctly encoded for both models) — which pins the base
distribution near 0 (the base is near-certain on its own samples); the off-support result
(DPO ≈ −15, below the −9.7 uniform line) holds regardless. For a moderate base distribution,
swap in a real held-out set in the base model's encoding. Numbers in `results/dpo_logprob_fig.json`.

## Validation provenance (how the two distributions were found)
The HIGH/LOW pools are the chemistry extremes of the model's training-distribution Counter,
found by clustering its (E,N) tuples into chemistry sub-pools, scanning each, and pushing
greedily to the extremes (validated at n=120): HIGH = rare-earths (0.82), LOW = H/Pt-group
(broad 0.08). This is exactly what `experiments/scan_extremes.py` re-runs.

## Optional: 3-MLIP "lemat-raw" scorer
The published bin-pool/matrix numbers used a 3-MLIP ensemble hull (orb + mace + uma).
`scripts/verify_combined.py` runs it; needs the heavier `requirements_3mlip_optional.txt`
(torch 2.8, orb-models, fairchem) and **access to the gated `facebook/UMA` model on HF**.
Use orb variant **`_mpa`** (not `_omat`) — `_omat` is mis-calibrated against the PBE hull.
For most verification the single-MACE `score.py` is simpler and matches.

## Layout
```
generate.py            score.py            requirements.txt   requirements_3mlip_optional.txt
checkpoints/alex_nolemat_lowhull/   (steering model + tokenizer)
lemat/data/distributions/   HIGH_rareearth.json  LOW_broad_HPtRh.json  an_lh.dist.json (full Counter)
lemat/data/structures/      example generated structures (to test score.py)
lemat/code/py/              ed/  dielectric/{eval,chem,dielectric_data}  lemat-genbench/  (+ material_hasher)
scripts/                    verify_combined.py (3-MLIP), score_singlemace.py, steer_experiment.py,
                            claim1_survival_{spectrum,crystals}.py, make_figures.py,
                            plot_pred_vs_obs.py, dpo_logprob_fig.py
results/                    claim1_survival_{spectrum,crystals,crystals_density}.json, dpo_logprob_fig.json
figures/                    fig_claim1_{survival,convergence}_ladder.png, fig_recast_pred_vs_obs.png,
                            fig_dpo_logprob_dist.png
claim1_survival_findings.md  (Claim 1 budget-survival write-up)
experiments/                _repo.py + scan_extremes/natural_pools/method_compare/claim1_survival drivers,
                            findings.md (full write-up), README.md
```

## Gotchas
- `score.py` needs `MP_API_KEY` (free) + `emmet-core` for the MP hull references; first run per
  chemical system fetches + caches them.
- LeMat-Bulk (~10 GB) auto-downloads on first scoring run (public; needs `HF_TOKEN`).
- For the 3-MLIP optional path only: genbench's multi-MLIP preprocessor must run with
  `n_jobs=1` (its process pool crashes loading CUDA in forked workers).

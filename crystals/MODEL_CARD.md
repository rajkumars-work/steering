---
license: mit
tags:
- materials
- crystal-structure-generation
- generative-model
- chemistry
- condensed-matter-physics
---

# CUES crystal generator — model card

The crystal-domain model behind the steering paper's cross-domain claims. Written to parity with
the image setup (DiT-XL/2 + ResNet-50): architecture, training data, objective, release, and the
exact metric definitions every cited number uses.

## Model

| | |
|---|---|
| Name | **CUES** (checkpoint `alex_nolemat_lowhull`, epoch 120) |
| Architecture | **EdGPT** encoder–decoder transformer |
| Size | encoder **6** layers / decoder **18** layers, model dim **768**, **≈238 M** parameters |
| Vocabulary | **16,000** (SentencePiece; `model_sp.model`, committed) |
| Generation-internal version id | `d15_binrho_k7` (target encoding scheme) |
| Checkpoint | `ed_ckpt_final.pt`, 909 MB, sha256 `9b8e4b2e…54177` (see `ARTIFACTS.md`) |
| Release | HuggingFace `rajkumars47/cues-alex-nolemat-lowhull` (fetched, **not** committed; `fetch_checkpoint.sh`) |

**Conditioning / I/O.** A prompt is `"<elements> | <natoms> | <bin-tags>"`, where the bin-tags
condition the generation on coarse property bins:
- `bg-{vlow,low,mid,high,vhigh}` — band-gap bin
- `hull-{vlow,low,mid,high}` — energy-above-hull (stability) bin
- `k-{…}` — dielectric constant bin · `rho-{…}` — density bin

The decoder emits a structure encoding (anonymized formula + concrete formula, space group, and
coordination/near-neighbor geometry) that is parsed to an ASE `Atoms` (lattice + sites). Naked
prompts (`"<elements> | <natoms> | "`, no tags) are used for the distribution-steering experiments.

## Training data

- **Source:** Alexandria-derived. The `alex_nolemat_lowhull` training set
  (`alex_nolemat_lowhull_dataset_train.csv`, ≈225 k structures), i.e. Alexandria structures,
  **`natoms < 20`**, restricted to a **low energy-above-hull** subset, with **LeMat-Bulk overlap
  excluded** (`nolemat`) so LeMat-Bulk can serve as the novelty reference at eval time.
- Each row: `source, target, id, origin, label, e_above_hull, in_lemat` — `source` is the
  prompt (elements | natoms | tags), `target` the structure encoding, `origin = alex`.

## Objective

Supervised next-token cross-entropy: encoder-decoder translation from (composition + property-bin
tags) to the structure encoding. No RL/preference objective in the base CUES model. (A separate
aggressive **DPO/IPO novelty fine-tune** is used only for the Claim 4.4 off-support demonstration;
it is not the CUES generator.)

## Metric definitions (exact, as used in every cited number)

- **stability — energy above hull (`e_above_hull`, eV/atom):** a single **MACE** single-point
  energy evaluated against a convex hull whose Materials-Project reference entries are
  **re-evaluated with the same MACE** (self-consistent; cached per chemical system). *stable* =
  `eah ≤ 0`; *metastable* = `0 < eah ≤ 0.1`. (`chem.stability.compute_e_above_hull`.) Needs
  `MP_API_KEY` + `emmet-core`. A 3-MLIP variant (orb `_mpa` + MACE + UMA, mean hull) is available
  for cross-checks; use orb `_mpa` not `_omat`.
- **Combined = SUN + MSUN**, over `n_total` (**no** validity gate — validity is a separate
  metric). **SUN** = stable ∧ novel ∧ unique; **MSUN** = metastable ∧ novel ∧ unique.
  Novelty (vs **LeMat-Bulk**) and uniqueness via lemat-genbench `SUNMetric`.
- **BVS-GII (bond-valence self-consistency, valence units):** the Global Instability Index — RMS
  over sites of (bond-valence sum from bond lengths, Brown–Altermatt `calculate_bv_sum`, minus the
  formal oxidation state). 0 = perfectly self-consistent; `None` for pure intermetallics (no anion).
- **validity:** SMACT charge-balance ∧ BVS-assignable ∧ minimum interatomic distance ≥ 0.5 Å
  (`chem.validity.check_validity`, one call).
- **density (g/cm³):** geometry only — total atomic mass / cell volume.

## Reproducing every cited number

Scripts live in `steering/crystals/` (mirrored from `dielectric/eval/within_distribution_steering/`):
- `generate.py --target T` + `score.py` — Combined steering (Claim 2).
- `experiments/scan_extremes.py`, `natural_pools.py`, `method_compare.py` — the steering range and
  the two target-hitting methods.
- `scripts/claim1_survival_spectrum.py` + `claim1_survival_crystals.py` — the Claim 1 carry-over
  ladder and density view.
- `scripts/plot_pred_vs_obs.py`, `dpo_logprob_fig.py`, `make_figures.py` — the paper figures.
- `review1/` — the reviewer-requested CIs, convergence, joint baselines, and ladder prediction.

Reference data revisions (LeMat-Bulk, LeMat-Bulk-MLIP-Hull, UMA) are pinned in `ARTIFACTS.md`.
The model is fetched from HF and hash-verified; nothing irreproducible is committed.

## Scorers & statistics — pinned (camera-ready hygiene)

All numbers in the crystal track use these exact scorers/versions (full freeze in
`requirements.txt`):

| scorer | package / model | version |
|---|---|---|
| stability (single-MACE e_above_hull) | `mace-torch`, foundation model **mace-mp-0b3-medium** | 0.3.15 |
| stability ensemble (C-E9) | MACE foundation models {mace-mp-0b3-medium, mace_mp `medium`, mace_mp `small`}, median | 0.3.15 |
| 3-MLIP ensemble (optional, `.venv_genbench`) | orb (`orb_v3_conservative_inf_mpa`) + MACE + UMA, mean hull | orb-models 0.7.0, fairchem-core 2.21.0 |
| novelty / uniqueness (SUN/MSUN) | lemat-genbench `SUNMetric` vs LeMat-Bulk | bundled (rev in `ARTIFACTS.md`) |
| BVS-GII | pymatgen `BVAnalyzer` + `calculate_bv_sum` (Brown–Altermatt) | pymatgen 2026.5.4 |
| validity (SMACT ∧ BVS ∧ min-dist) | `chem.validity` (SMACT + pymatgen) | SMACT 4.0.0, pymatgen 2026.5.4 |
| band gap (C-E3) | XGBoost composition DFT-band-gap surrogate (`data/xgb_composition_dft_band_gap.json`) + matminer features | xgboost 3.2.0, matminer 0.10.1 |
| MP hull references | Materials Project API + `emmet-core`, re-evaluated with the same MACE | emmet-core 0.87.0 |

**Sample sizes / seeds / CI method (consolidated):**
- **C-E1** density carry-over: n=30 chem, bootstrap-over-chemistries (B=10 000), 95% percentile CI.
- **C-E1** DPO gap: n=250 structures, bootstrap, 95%.
- **C-E1** Combined sweep: n=100/target, **Wilson** 95% on the proportion.
- **C-E2** ladder: 24 chem × **3 seeds**, M_data=15 / M_gen=30; bootstrap-over-chemistries (×seeds) 95%.
- **C-E3** joint: N=120/condition × **3 seeds** (→ 6 with the round-2 bump), **Wilson** 95% on pooled rate.
- **C-E8** key sensitivity: re-analysis of the C-E2 per-item data, bootstrap-over-groups 95%.
- **C-E9** ensemble: same 24 chem (seed 0), M_data=15 / M_gen=30, single-vs-ensemble r² on the
  **same** structures, bootstrap-over-chemistries 95%.
- **C-E4** pre-registration: committed at git `d96b827` before the C-E2 audit (`08981f7`);
  prior on a correct 3-rung ordering = 1/6.

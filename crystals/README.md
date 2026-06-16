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

## Validation provenance (how the two distributions were found)
The HIGH/LOW pools are the chemistry extremes of the model's training-distribution Counter,
found by clustering its (E,N) tuples into chemistry sub-pools, scanning each, and pushing
greedily to the extremes (validated at n=120): HIGH = rare-earths (0.82), LOW = H/Pt-group
(broad 0.08). Full methodology + the scan: see the paper / `within_distribution_steering`.

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
lemat/data/distributions/   HIGH_rareearth.json  LOW_broad_HPtRh.json
lemat/data/structures/      example generated structures (to test score.py)
lemat/code/py/              ed/  dielectric/{eval,chem,dielectric_data}  lemat-genbench/  (+ material_hasher)
scripts/                    verify_combined.py (3-MLIP), score_singlemace.py, steer_experiment.py
```

## Gotchas
- `score.py` needs `MP_API_KEY` (free) + `emmet-core` for the MP hull references; first run per
  chemical system fetches + caches them.
- LeMat-Bulk (~10 GB) auto-downloads on first scoring run (public; needs `HF_TOKEN`).
- For the 3-MLIP optional path only: genbench's multi-MLIP preprocessor must run with
  `n_jobs=1` (its process pool crashes loading CUDA in forked workers).

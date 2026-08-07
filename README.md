# Steering generative models — verification

Companion code and data for the steering paper. The paper's claim is that how far you can
steer a generative model is fixed, *before you train anything*, by a budget you can read off the
**training data's distribution** — and that you only ever need that distribution's *shadow* (a few
statistics per bin), not the data itself, to check it.

This repo lets you verify that, in **two unrelated domains** that share no model, data, or code:

- **`images/`** — class-conditional **DiT-XL/2-256 on ImageNet** (the *legible* track). You can
  reproduce the data-side claims from a **132 KB shadow** with nothing but `numpy`, and the
  model-side claims by generating from DiT.
- **`crystals/`** — **CUES**, a crystal-structure generator (the *consequence* track). Steer a
  materials benchmark (*Combined* = novel + unique + (meta)stable) to a target, and score it.

Each folder has its own README and runs independently.

## The one idea the repo embodies
You need **the model + the training-data distribution** — nothing else:
- `images/distributions/` — the `(P, L)` shadow (per-class statistics) of an ImageNet audit.
- `crystals/lemat/data/distributions/` — the `(elements, atom-count)` anchor pools that set the target.

From those, the data-side claims are a short calculation (no images, no GPU); the model-side
claims need only the published checkpoint.

## Conventions
- **Model weights are not committed.** Public models (DiT, CLIP) are fetched by a script;
  the CUES checkpoint is published on the HF model hub and fetched on setup. Each folder's
  README has the exact commands.
- Large reference datasets (LeMat-Bulk, hull caches) auto-download per their usual sources.

## The paper

The manuscript is in **[`paper/steering_budget.pdf`](paper/steering_budget.pdf)** (CC BY 4.0).
_arXiv: link TBD._

## The model

The crystal generator (CUES, ≈238 M params, 909 MB) is hosted on the Hugging Face model hub —
too large for git. It is fetched on setup, not committed:

```bash
bash crystals/fetch_checkpoint.sh   # -> crystals/checkpoints/alex_nolemat_lowhull/ (hash-verified)
```

The image track uses public models (DiT-XL/2, CLIP, SD-VAE), fetched by their usual `from_pretrained`
paths. All model/dataset revisions are pinned in `crystals/ARTIFACTS.md`.

## Reproducing the Evidence section

Two independent tracks that share no model, data, or code. Each folder's `RESULTS.md` ties every
cited number to the exact script that produced it — start there. The map:

| what the paper claims | domain | entry point | where the numbers live |
|---|---|---|---|
| **Claim 1** — the split is exact, and it survives the model | images | `images/src/verify_dataside.py` (data-side, no GPU); `claim1_survives.py` (model-side) | `images/RESULTS.md` |
| | crystals | `crystals/scripts/claim1_survival_crystals.py`, `claim1_survival_spectrum.py` | `crystals/results/crystal_experiments.results.md` |
| **Claim 2** — showing's reach ($\sqrt{\chi^2 E}$) | images | `images/src/claims234.py` | `images/RESULTS.md` |
| **Examples beat knobs** (band gap, density, stability) | crystals | `crystals/scripts/steer_experiment.py`, `experiments/claim1_density.py` | `crystals/results/ce*.json` |
| **Expressiveness** — coverage / joint (A ∧ B) | crystals | `crystals/experiments/` + `crystals/scripts/` (ce12/ce13/ce3) | `crystals/results/crystal_experiments.results.md` |
| | images | `images/src/claim3_images.py` (animal ∧ nature) | `images/RESULTS.md` |
| **Steer the generator to a target** Combined score | crystals | `python crystals/generate.py --target 0.5 --n 100 --out gen.extxyz && python crystals/score.py gen.extxyz` | `crystals/RESULTS.md` |

The **data-side** image claims need only the shipped 132 KB shadow (`numpy`, runs in <1 s). Everything
**model-side** needs the checkpoint above and a single GPU.

## Status
- `images/` — a **reconstruction** (the original scripts were lost); numbers are freshly
  measured and reconciled in `images/RESULTS.md`. Data-side claims reproduce exactly.
- `crystals/` — the CUES steering verification (checkpoint + scripts).

## Citation

```bibtex
@misc{rajendran2026steeringbudget,
  title  = {The Steering Budget: Examples beat Knobs},
  author = {Rajendran, Raj Kumar},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/rajkumars-work/steering},
  % eprint = {TODO(preprint): arXiv id}, archivePrefix = {arXiv},
}
```

## License

Code (everything outside `paper/`) is **MIT**. The paper PDF under `paper/` is **CC BY 4.0**.
See `LICENSE`.

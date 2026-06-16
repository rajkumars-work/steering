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
- `crystals/distributions/` — the `(elements, atom-count)` anchor pools that set the target.

From those, the data-side claims are a short calculation (no images, no GPU); the model-side
claims need only the published checkpoint.

## Conventions
- **Model weights are not committed.** Public models (DiT, CLIP) are fetched by a script;
  the CUES checkpoint is published on the HF model hub and fetched on setup. Each folder's
  README has the exact commands.
- Large reference datasets (LeMat-Bulk, hull caches) auto-download per their usual sources.

## Status
- `images/` — a **reconstruction** (the original scripts were lost); numbers are freshly
  measured and reconciled in `images/RESULTS.md`. Data-side claims reproduce exactly.
- `crystals/` — the CUES steering verification (checkpoint + scripts).

Paper: _link TBD_.

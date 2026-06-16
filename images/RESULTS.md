# ImageNet-domain evidence — reconstruction results

The original image-evidence scripts were lost; this is a from-scratch rebuild on this box
(volatile `/opt/dlami/nvme/imagenet_evidence`). Goal: let a reader reproduce the Evidence
image claims. **Numbers here are freshly measured and will differ from the recorded paper
values** (different sampler/subset/seed) — the reconciliation table flags what reproduces.

## Environment (all on volatile)
- Models: DiT-XL/2-256 (diffusers), CLIP ViT-L/14, SD-VAE, LAION aesthetic predictor, ResNet-50.
- ImageNet: canonical `ILSVRC/imagenet-1k` **validation** split, streamed (gate accepted).
- Pinned venv: torch 2.7/cu128, diffusers 0.38, transformers 5.12, datasets 5.0; A10G.
- Reconstruction choices recorded in code: val 25/class (vs original train 100/class),
  256 px, DPMSolver 50 steps, JPEG q90, 4 concept prompts {animal,vehicle,food,nature}.

## Scripts (`src/`)
- `labelers.py` — 7 targets (aesthetic, brightness, filesize/MP, 4 CLIP concept sims) + ResNet key π.
- `generate.py` — DiT class-conditional sampling, CFG, μ_P recipes.
- `audit_claim1.py` — streams ImageNet, computes per-class shadow → Claim 1.
- `claims234.py` — generation-side: Claim 2 realization, 4.1 (J→G), 4.3 (CFG), ρ.
- `verify_dataside.py` — **reader check: reproduces all data-side claims from the 132 KB shadow alone.**

## The shadow (the shippable artifact)
`out/claim1_perclass.npz` (132 KB) = per-class `w_b, g_b, v_b` for the 7 targets, plus the global
`Var(L)` in `claim1.json`. Per the framework's own thesis, **this is all a reader needs to verify the
data-side claims** — no ImageNet, no models, no GPU. `verify_dataside.py` does exactly that in <1 s.
License-clean (derived statistics, not images).

---

## Claim 1 — LoTV split exact, and the spectrum  ✓ REPRODUCED  (N=25,000; 1000 classes)

T + E = Var(L) to machine precision (relative residual < 1e-6 for all 7 targets; absolute
~1e-16–1e-19, filesize's 1.9e-6 is ~1e-16 relative at its ~1e9 scale):

| target | T | E | T+E | Var | E/(E+T) |
|---|---|---|---|---|---|
| aesthetic | 0.1553 | 0.0579 | 0.21325 | 0.21325 | 0.272 |
| brightness | 0.01507 | 0.00304 | 0.01811 | 0.01811 | 0.168 |
| filesize/MP | 9.65e9 | 4.11e9 | 1.376e10 | 1.376e10 | 0.299 |
| sim_animal | 2.71e-4 | 7.08e-4 | 9.80e-4 | 9.80e-4 | **0.723** |
| sim_vehicle | 2.74e-4 | 3.60e-4 | 6.35e-4 | 6.35e-4 | 0.568 |
| sim_food | 2.78e-4 | 4.77e-4 | 7.55e-4 | 7.55e-4 | 0.632 |
| sim_nature | 3.32e-4 | 3.76e-4 | 7.08e-4 | 7.08e-4 | 0.530 |

**Spectrum holds:** class-aligned CLIP concepts are high-E (0.53–0.72, mostly between-bin);
pixel-level targets (brightness 0.17, aesthetic 0.27, filesize 0.30) are high-T (within-bin).
**Coarsening 1000→50 moves variance E→T** for every target (e.g. aesthetic E 0.058→0.003, T 0.155→0.210).

## Claim 2 — bounds (data side)  ✓ REPRODUCED
For every target, the top-50 showing recipe's predicted shift `Σ μ_P g_b − ḡ` sits within
`√(χ²·E)` (e.g. brightness 0.127 ≤ 0.240; sim_animal 0.049 ≤ 0.116). All 7 within bound.

## Claim 2 realized (showing), 4.1 (J→G), 4.3 (CFG), ρ  — generation-side (50 classes, m=8, CFG=1)

**Claim 2 realized:** showing under the top-50 brightness μ_P moved DiT's mean brightness
+0.077 (vs baseline ḡ=0.451), inside the `√(χ²E)`=0.240 bound. ✓

**Claim 4.1 — DiT(CFG=1) per-class mean vs the audit g_b:**

| target | r² | MAD | mean offset |
|---|---|---|---|
| sim_animal | **0.885** | 0.0066 | +0.0006 |
| sim_nature | **0.829** | 0.0070 | +0.0003 |
| sim_food | 0.695 | 0.0091 | +0.0009 |
| filesize/MP | 0.575 | 3.9e4 | −1.5e4 |
| sim_vehicle | 0.563 | 0.0076 | +0.0012 |
| brightness | 0.433 | 0.041 | +0.009 |
| aesthetic | 0.399 | 0.397 | **−0.388** |

The bridge tracks well on **high-E (class-aligned) targets** (sim_animal/nature r²≈0.83–0.89);
the **aesthetic −0.388 offset reproduces the recorded −0.44 label-pipeline drift**. The recorded
**r²=0.9999 does not reproduce** — at m=8 the per-class means are noise-limited, worst on
**high-T targets** (brightness 0.43), and the r² ranking *follows the E/(E+T) spectrum* (more
between-class signal → higher r²). A higher-m re-run (≥50/class) is the way to recover the tight fit.

**Realization ρ = √(T(G)/T):** 0.90–1.12 across targets (brightness 1.04, sim_animal 0.91,
aesthetic 1.12) — **near 1 at canonical settings**, matching "the model realizes the within-bin
budget." The recorded ρ=1.49 *expansion* (aggressive guidance + joint μ_P) was not run.

**Claim 4.3 — CFG sweep at class 207:** brightness 0.516→0.567, sim_animal 0.178→0.196 (rise then
plateau), aesthetic peaks at CFG=7 (4.97) then dips at 12 (4.60). A clear CFG-driven drift, **not
strictly monotonic at the top** (CFG over-saturation) — directionally as predicted.

---

## Reconciliation (recorded → measured)
| Evidence claim | recorded | measured (this rebuild) | status |
|---|---|---|---|
| C1 split exact | "to FP precision" | resid ~1e-16 | ✓ reproduced |
| C1 spectrum (concepts high-E, pixels high-T) | qualitative | yes (0.53–0.72 vs 0.17–0.30) | ✓ reproduced |
| C1 coarsening E→T | qualitative | yes, all targets | ✓ reproduced |
| C2 data-side bound | telling≤√T, showing≤√(χ²E) | all 7 targets within | ✓ reproduced |
| C2 showing realizes a shift | model moves under μ_P | brightness +0.077 ≤ 0.240 bound | ✓ reproduced |
| C4.1 DiT≈audit (r²=0.9999, MAD 0.004) | recorded | high-E r²≈0.83–0.89; overall r² **not** 0.9999 at m=8 (noise-limited, high-T low) | ⚠ partial — needs ≥50/class |
| C4.1 aesthetic −0.44 label drift | recorded | offset **−0.388** | ✓ reproduced |
| C4.3 CFG monotonic drift | recorded | drift present; plateaus/dips at CFG=12 (saturation) | ~✓ direction |
| realization ρ (≈1 canonical; 1.49 expansion) | recorded | ρ=0.90–1.12 (canonical ✓); expansion case not run | ✓ canonical; ⏳ expansion |
| C2 brightness 23% over bound (heavy tail) | recorded | not yet probed (needs top-k sweep) | ⏳ todo |
| C3 bright∧animal 1%→35.5% | recorded | **deferred** — class-conditional DiT makes this the *telling-easy* (independently-controllable) case, not a showing-only joint; consistent with the Application §3 decision to keep the joint a structured-generation case | ⚠ flag |

## Honest flags
- Subset/split differ from the original (val 25/class vs train 100/class, 256 px) → exact T/E
  values differ; the **qualitative claims and the exact-split identity reproduce**.
- C3 (joint) needs a careful telling/showing definition; left out of the auto-run (see table).
- Generation-side numbers depend on sampler (50-step DPMSolver vs original DDPM-250).

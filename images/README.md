# imagenet-steering-verify

Companion verification repo for the **ImageNet-domain evidence** in the steering paper
(class-conditional DiT-XL/2-256 on ImageNet). It lets you check our claims yourself — and it
makes a point the paper makes: **to verify the data-side claims you need only the *shadow* of
the training data (a few numbers per bin), not the data itself.**

Two tiers, by what you need.

---

## Tier 1 — verify from the shadow alone (no data, no models, no GPU)

`distributions/claim1_perclass.npz` (**132 KB**) is the entire `(P, L)` shadow of an ImageNet
audit: per-class share `w_b`, mean `g_b`, and within-class variance `v_b` for 7 targets
(aesthetic, brightness, file-size/MP, and 4 CLIP concept similarities). With it and the global
`Var(L)` (in `distributions/claim1.json`) you can reproduce, in under a second:

```bash
pip install numpy            # the only dependency for this tier
python src/verify_dataside.py
```

This reproduces:
- **Claim 1** — `T + E = Var(L)` to machine precision; the `E/(E+T)` spectrum (class-aligned
  concepts high-E, pixel-level targets high-T); coarsening 1000→50 bins moves variance `E→T`.
- **Claim 2 (data side)** — the showing bound `|Σ μ_P g_b − ḡ| ≤ √(χ²·E)` holds for every target.

No ImageNet, no model weights, no GPU. That *is* the framework's claim: the audit is cheap and
the shadow is all you need on the data side.

Figures: `python scripts/make_figures.py` (numpy + matplotlib) redraws the paper's image
figures into `figures/` from the shadow + the shipped run results.

---

## Tier 2 — regenerate everything from scratch (GPU + models + ImageNet)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # torch 2.7/cu128, diffusers, transformers, datasets, ...
bash scripts/download_models.sh           # DiT, CLIP, SD-VAE, LAION predictor (public; ~6 GB)
export HF_TOKEN=...                        # to stream gated ILSVRC/imagenet-1k (or use a mirror; see below)
```

- **Rebuild the shadow** (data side) — stream ImageNet, compute the per-class stats:
  ```bash
  python src/audit_claim1.py --per_class 25     # writes distributions/claim1_perclass.npz, claim1.json
  ```
- **Claim 1 — does the split survive the model?** generate from DiT and recompute the
  within/between split on its own outputs:
  ```bash
  python src/claim1_survives.py                 # 40 classes x 40 imgs, raw conditional
  python src/claim1_survives_sweep.py           # same, swept over guidance (CFG 2, 4)
  ```
- **Claim 3 — the entangled joint "animal AND nature"** (telling vs showing):
  ```bash
  python src/claim3_images.py                   # writes distributions/claim3_images.json
  ```
- **Claim 2 — showing's reach grows with concentration (χ²), under the ceiling:**
  ```bash
  python src/claim2_chi2.py                     # writes distributions/claim2_chi2.json
  ```
- **Claims 2 & 4 — data-side bounds, the `J→G` bridge, CFG drift, ρ:**
  ```bash
  python src/claims234.py                       # writes distributions/claims234.json
  ```

ImageNet: `ILSVRC/imagenet-1k` is gated (accept terms on its HF page, then a read token streams
it). For an ungated path, `audit_claim1.py` can point at `evanarlian/imagenet_1k_resized_256`
(256 px, standard labels) — one-line change.

---

## What's shipped vs not
- **Shipped**: the shadow + run results (`distributions/*.npz`, `*.json`), all code, the figure
  script, and `RESULTS.md`. License-clean — derived statistics, not images.
- **Not shipped**: model weights (public — `scripts/download_models.sh` fetches them) and ImageNet
  (gated/licensed — link above). Keeps the repo tiny and redistributable.

## Results & honest notes
See **`RESULTS.md`** for the measured numbers and a recorded-vs-measured reconciliation. This is a
*reconstruction* (the original scripts were lost), so values differ from the paper's recorded ones;
reconstruction choices (validation split, 25/class, 256 px, 50-step DPMSolver, 4 concept prompts)
are recorded in the code. The data-side claims reproduce exactly (it's an identity); the model-side
`J→G` correlation needs more samples/class than the quick runs to hit the recorded tight fit.

## Layout
```
src/            verify_dataside.py  audit_claim1.py  claims234.py  claim2_chi2.py
                claim1_survives.py  claim1_survives_sweep.py  claim3_images.py
                generate.py  labelers.py
distributions/  claim1_perclass.npz (the shadow)  claim1.json  claims234.json
                claim2_chi2.json  claim3_images.json  claim1_survives.json  claim1_survives_sweep.json
scripts/        download_models.sh  make_figures.py
figures/        fig_brightness_split.png  fig_label_vs_exemplar_spread.png  fig_chi2_reach.png  fig_joint_target_bars.png
RESULTS.md      requirements.txt
```

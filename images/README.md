# imagenet-steering-verify

Companion verification repo for the **ImageNet-domain evidence** in the steering paper
(class-conditional DiT-XL/2-256 on ImageNet). It lets you check our claims yourself — and it
makes a point the paper makes: **to verify the data-side claims you need only the *shadow* of
the training data (a few numbers per bin), not the data itself.**

Two tiers, by what you need.

---

## Tier 1 — verify from the shadow alone (no data, no models, no GPU)

The file `data/claim1_perclass.npz` (**132 KB**) is the entire `(P, L)` shadow of an ImageNet
audit: per-class share `w_b`, mean `g_b`, and within-class variance `v_b` for 7 targets
(aesthetic, brightness, file-size/MP, and 4 CLIP concept similarities). With it and the global
`Var(L)` (in `data/claim1.json`) you can reproduce, in under a second:

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
  python src/audit_claim1.py --per_class 25     # writes data/claim1_perclass.npz, data/claim1.json
  ```
- **Model-side claims** — generate from DiT and label:
  ```bash
  python src/claims234.py                       # writes data/claims234.json
  ```
  Covers Claim 2 (realized shift), **Claim 4.1** (DiT per-class means vs the audit → the `J→G`
  bridge), **Claim 4.3** (CFG sweep drift), and the realization ratio **ρ**.

ImageNet: `ILSVRC/imagenet-1k` is gated (accept terms on its HF page, then a read token streams
it). For an ungated path, `audit_claim1.py` can point at `evanarlian/imagenet_1k_resized_256`
(256 px, standard labels) — one-line change.

---

## What's shipped vs not
- **Shipped**: the shadow (`data/*.npz`, `*.json`), all code, and `RESULTS.md`. License-clean —
  derived statistics, not images.
- **Not shipped**: model weights (public — `scripts/download_models.sh` fetches them) and ImageNet
  (gated/licensed — link above). Keeps the repo tiny and redistributable.

## Results & honest notes
See **`RESULTS.md`** for the measured numbers and a recorded-vs-measured reconciliation. This is a
*reconstruction* (the original scripts were lost), so values differ from the paper's recorded ones;
reconstruction choices (validation split, 25/class, 256 px, 50-step DPMSolver, 4 concept prompts)
are recorded in the code. The data-side claims reproduce exactly (it's an identity); the model-side
`J→G` correlation needs more samples/class than this quick run to hit the recorded tight fit.

## Layout
```
src/   labelers.py  generate.py  audit_claim1.py  claims234.py  verify_dataside.py
data/  claim1_perclass.npz (the shadow)  claim1.json  claims234.json
RESULTS.md   requirements.txt   scripts/download_models.sh
```

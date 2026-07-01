# Reproduce the steering experiments

These are the experiment drivers behind the paper's steering numbers, made runnable from a fresh
clone of this repo. Each resolves the checkpoint, the distribution Counter, and the bundled code
relative to the repo root (`_repo.py`) — no absolute paths to edit. They all load the model +
single-MACE scorer **once** and loop; expect multi-hour, GPU-bound runs.

Run from the repo root with the same environment as `score.py` (see the top-level README — one
venv, `HF_TOKEN` + `MP_API_KEY` set). Outputs land in `experiments/out/<name>/`.

```bash
. .venv/bin/activate
export HF_TOKEN=hf_...        # LeMat-Bulk novelty reference (public)
export MP_API_KEY=...         # Materials-Project hull references (free)
python experiments/scan_extremes.py
```

| script | what it reproduces | output | ~cost |
|---|---|---|---|
| `scan_extremes.py` | **The discovery experiment.** Clusters the (E,N) distribution into 64 chemistry sub-pools, scans each, and greedily pushes to the high/low extremes — finding the Combined ~0.03→0.82 range and the two anchor pools shipped in this repo. Validates each extreme at n=120. | `out/scan_extremes/{scan,HIGH,LOW,summary}.json` | ~4–5 GPU-h |
| `natural_pools.py` | **Method 2 (natural pools).** For each 0.1 Combined target, validates the single chemistry bin that naturally sits there (n=100), saving each as a reusable pool. | `out/natural_pools/{results.json,pools/}` | ~1 GPU-h |
| `method_compare.py` | **MIX vs NATURAL at target 0.5.** Means/variance, chemistry, natoms, element diversity, capacity (duplication onset), χ² push, bond-valence assignability — 3×n=100 each. | `out/method_compare/comparison.json` | ~1 GPU-h |
| `claim1_survival_spectrum.py` | **Claim 1 (crystals) — the main run.** Does the within/between budget split read off the DATA survive in the MODEL's outputs, *as a function of how hard the property is*? Walks a difficulty ladder (density → SMACT → BVS-GII → validity → e_above_hull) on the same generated structures and reports (A) convergence ladder + (B) budget survival (T/E + per-chemistry r²). Same `parse_target` decoder + same scorers on both sides. | `out/claim1_survival_spectrum.json` | multi-h |
| `claim1_density.py` | **Claim 1 fast view** — the easy end only (density, pure geometry; no MACE), finishes in minutes. | `out/claim1_survival_density.json` | ~min |

## The target ladder (Method 1, mixing) is already reproducible

`generate.py --target T` in the repo root *is* Method 1 — it mixes the HIGH/LOW anchors at
`lambda = (T − 0.14)/(0.85 − 0.14)`. Reproduce the calibration ladder directly:

```bash
for T in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
  python generate.py --target $T --n 100 --out out_T$T.extxyz
  python score.py out_T$T.extxyz
done
```

## Notes on faithful reproduction

- **Determinism**: the clustering is `KMeans(n_clusters=64, random_state=0, n_init=4)` on
  element-presence one-hot + `natoms/40`, so the 64 sub-pools (and the bin indices used by
  `natural_pools.py` / `method_compare.py`) reconstruct exactly. Generation/scoring carry the
  usual run-to-run noise (~±0.02–0.03 Combined at n=100; ≤ the binomial SE).
- **Winner's-curse is real**: scan picks at n=24 regress at n=120/n=100 (documented in
  `findings.md`). Trust the validated numbers, not the scan estimates.
- **The Claim 1 scripts need the dataset** (real structures for the DATA side), which is NOT in
  this data-light repo. Point `CUES_TRAIN_CSV` at the `alex_nolemat_lowhull` training CSV from the
  comprehensive repo. Both sides use the same scorer/decoder — never the CSV's stored
  `e_above_hull` (different labeller).
- **One scorer throughout**: `chem.stability.compute_e_above_hull` (single MACE single-point vs a
  MACE-reevaluated MP hull) + lemat-genbench `SUNMetric`. This is what produced every steering
  number; see the top-level README for the 3-MLIP optional variant.

## Claim 1 result (what the spectrum run found)

Per-chemistry agreement between the data audit and the model's own outputs **degrades exactly as
the property gets harder** for the model — the framework predicts the model where the model is
good, and progressively fails where it is poor:

| property (easy → hard) | per-chemistry mean r² (data vs model) |
|---|---|
| density (geometry) | **0.99** |
| BVS-GII (bond-valence self-consistency) | **0.83** |
| e_above_hull (stability) | **0.30** |

Full numbers, both tables, and the curation-confound discussion are in `claim1_survival_findings.md`.

## Write-ups

- `findings.md` — the full within-distribution steering result (0.03→0.82), both target-hitting
  methods, the ladder + natural-pool validation tables, the method comparison, and the χ²/min-χ²
  analysis.
- `claim1_survival_findings.md` — the Claim 1 budget-survival spectrum (density → BVS-GII →
  e_above_hull), methodology, and the BVS-GII metric note.
- `claim1_survival.README.md` — the Claim 1 hand-off spec (the difficulty-ladder design, the image
  result it mirrors).

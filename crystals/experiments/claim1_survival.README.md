# Hand-off: does the data-audit budget split survive the model? (crystals)

Crystal mirror of the ImageNet experiment, for the paper's Claim 1: the within/between budget
split read off the **data** should be reproduced by the **model's own outputs** — *to the extent
the model has converged on the property.* If the model can't reliably produce a property, its
statistics won't be preserved; that's a model limitation, not a hole in the framework.

So we study a **spectrum of three properties of increasing difficulty**, on the same generated
structures, and watch the statistics degrade.

## Scripts (this dir)
- **`claim1_survival_spectrum.py`** — the main run. A **difficulty ladder**, easy → hard
  (one `check_validity` call gives SMACT + BVS + min-dist; plus geometry and one MACE call):
  - **density** (geometry) — easiest, model nails it
  - **SMACT** (composition charge-balance) — easy
  - **BVS** (bond-valence self-consistency) — medium
  - **validity** (SMACT + BVS + min-dist all pass) — hard
  - **e_above_hull** (stability) — hardest

  Two views, both written to JSON:
  - **(A) convergence ladder** — pass rate, data vs model, for SMACT → BVS → validity →
    (meta)stable; the model's rate should fall as the check gets harder.
  - **(B) budget survival** — within (T) / between (E) split + per-chemistry mean r², for the
    *continuous* properties density → BVS deviation → e_above_hull; r²/E-match should fall
    from density to stability.

  Cost: MACE loads once, then ~K·(M_data+M_gen) scorings — multi-hour, like `binning_experiment.py`.
  Output → `/opt/dlami/nvme/recast/train/claim1_survival_spectrum.json`.
- `claim1_survival_crystals.py` — density-only, **fast** (no MACE); a quick first look if you
  don't want to wait for the full spectrum. Output → `..._density.json`.

`row_to_atoms` is wired (`parse_target`); density/BVS/e_above_hull are computed the same way for
data and generated structures (one measure per property, both sides).

## How to run
Use the warm crystal venv and run from this directory:
```bash
PY=/opt/dlami/nvme/recast/.venv/bin/python
export HF_HOME=/opt/dlami/nvme/hf_cache
export MP_API_KEY=...          # needed for the MACE reference hull; already set in this box's env
cd /home/ubuntu/code/py/dielectric/eval/within_distribution_steering

# 1) fast first look — density only, no MACE (minutes); validates the generate+parse pipeline
$PY claim1_survival_crystals.py
#    -> /opt/dlami/nvme/recast/train/claim1_survival_crystals_density.json

# 2) full ladder — detached, multi-hour (MACE loads once)
nohup $PY claim1_survival_spectrum.py > /opt/dlami/nvme/recast/train/spectrum.log 2>&1 &
tail -f /opt/dlami/nvme/recast/train/spectrum.log
#    -> /opt/dlami/nvme/recast/train/claim1_survival_spectrum.json   (ends with CLAIM1-SPECTRUM-DONE)
```

## Smoke-test before the long run (important)
These scripts were authored against the codebase interfaces (`eval.screening.load_model` /
`generate_one`, `chem.stability`, `chem.validity.check_validity`,
`dielectric_data.reader.parse_target`) but have **not been executed end-to-end** — shake out any
signature mismatch on a tiny run first. Edit the constants at the top of
`claim1_survival_spectrum.py`:
```python
K_BINS, MIN_MEMBERS, M_DATA, M_GEN, SEED = 2, 10, 3, 3, 0
```
run it (~1–2 min), confirm both tables print with sane numbers, then restore `24, 10, 10, 12, 0`
and launch detached. Things to verify in the smoke run:
- `parse_target(row[1], VERSION, row[3])` return type — `to_ase()` handles ASE *or* pymatgen, but
  confirm data densities come out ~2–8 g/cc for known structures.
- `generate_one(...).atoms` is the ASE structure (same usage as `binning_experiment.py`).
- `compute_e_above_hull(atoms, calc=..., timeout=120)["e_above_hull"]` returns a float.
- `check_validity(atoms)` returns `smact_pass` / `bvs_pass` / `validity_pass` / `bvs_max_deviation`.

## Tuning knobs (top of each script)
`K_BINS` (chemistry bins), `MIN_MEMBERS` (data structures required per bin), `M_DATA`, `M_GEN`,
`EAH_MAX` (drop exploded structures above this eV/atom). `VERSION = "d15_binrho_k7"`,
`CKPT = /opt/dlami/nvme/recast/train/mix_ep120_ckpt` (alex_nolemat_lowhull ep120).

## Why a spectrum (the point)
It turns the framing into a *measured* claim: the data audit predicts the model exactly where the
model is good (density), partly where it's so-so (BVS), and not where it's poor (stability) — and
the realization diagnostic (ρ/δ) is what flags the last case. The earlier stability-only run
(r² ≈ 0.07, off-hull) is the hard end of this spectrum; it's also curation-confounded (the data
was selected on low hull), which is a second reason its statistics are narrow.

## For framing parity — the image result this mirrors (in the paper)
Brightness, 40 classes, raw conditional: data within/between/total 0.0130 / 0.0032 / 0.0162;
model 0.0177 / 0.0017 / 0.0194. Total + dominant within-bin carry through; guidance reweights.

## The deliverable
`/opt/dlami/nvme/recast/train/claim1_survival_spectrum.json` is the whole output — it contains
the convergence ladder (data vs model pass rates) and the survival table (T / E / per-chemistry
r² per continuous property). The paper-side work reads that JSON file directly, so no write-up or
summary is needed — just leave it at that path (the density-only run leaves `..._density.json`).

Optional, to keep the public repo current: mirror the two scripts (and the JSON) into the repo's
crystal folder —
```bash
rsync -a --exclude '*.pt' /opt/dlami/nvme/cues_steering/ ~/code/py/steering/crystals/
cp claim1_survival_spectrum.py claim1_survival_crystals.py ~/code/py/steering/crystals/scripts/ 2>/dev/null || true
```
then commit locally (no push from this shared box; publish via `~/code/py/steering/sync_to_wip.sh`).

# validity — Physical Validity Checks

Post-relaxation screening for structures produced by generative models. Filters out
thermodynamically unstable, dynamically unstable, and out-of-distribution structures
before they enter the DFT queue.

## Files

| File | Purpose |
|---|---|
| `structural.py` | All check functions (per-structure + composition-level) + loader helpers |
| `screen_generated.py` | CLI tool for batch screening an extxyz file |

The pipeline batch orchestrator (`stage7.py`) lives in `pipeline/`.

---

## Requirements

- **Tier 0 checks** (element filter, min dist, isolated atoms, SMACT, BVS, chemistry filters) run on **CPU only** — no GPU required.
- **Full checks** (Tier 1–5: UQ, e-hull, phonons, cross-MLIP, MD) require a **CUDA-capable GPU** with MACE and PET models loaded.
- `screen_generated.py --full` and `pipeline/stage7.py` require GPU.

---

## Checks

All per-structure check functions in `structural.py` take an ASE `Atoms` object and return
a plain `dict` with a boolean `*_pass` key.

### Hard rejects — stop processing the structure if they fail

| Function | What it checks | Default threshold |
|---|---|---|
| `check_element_filter` | Radioactive / blocked elements | any blocked → reject |
| `check_min_dist` | Minimum interatomic distance | < 0.5 Å → reject |
| `check_isolated_atoms` | Disconnected atoms (no neighbour within cutoff) | any isolated → reject |
| `check_smact` | SMACT charge balance / electronegativity | no valid oxidation states → reject |
| `check_bvs` | Bond Valence Sum self-consistency | max deviation > 0.5 → reject |
| `check_uq` | MLIP energy uncertainty (PET UQ) | > 0.05 eV/atom → reject |
| `check_ehull` | Energy above convex hull (all-MLIP phase diagram) | > 0.1 eV/atom → reject |
| `check_phonon_stability` | Γ-point imaginary phonon modes (MACE finite differences) | any optic mode < −0.5 THz → reject |

### Soft flags — recorded but do not gate `overall_pass`

| Function | What it checks | Default threshold |
|---|---|---|
| `check_cross_mlip` | PET vs MACE single-point energy agreement | Δ > 0.5 eV/atom → flag |
| `check_md_stability` | Short NVT-MD (500 steps × 2 fs at 600 K, ASE Langevin) | displacement > 3 Å or volume drift > 20% → flag |

### Composition-level checks (formula string → bool)

| Function | What it checks | Default threshold |
|---|---|---|
| `is_charge_balanced` | Pymatgen oxidation-state guesses exist | no valid guess → fail |
| `passes_electronegativity_filter` | Pauling electronegativity spread | > 3.3 → fail |
| `passes_radius_filter` | Ionic radius ratio (max/min) | > 2.5 → fail |
| `passes_entropy_filter` | Stoichiometric entropy | > 1.3 → fail |
| `passes_tolerance_filter` | Goldschmidt tolerance factor (ABX₃ only) | outside 0.825–1.05 → fail |
| `passes_chemistry_filters` | All of the above combined | any fail → fail |

### `run_checks` — tiered execution

Runs checks in order (fast → slow), stopping immediately on any hard reject:

```
Tier 0  check_element_filter      ~0 s/structure  (CPU)
        check_min_dist            ~0 s/structure  (CPU)
        check_isolated_atoms      ~0 s/structure  (CPU)
        check_smact               ~0 s/structure  (CPU)
        check_bvs                 ~0 s/structure  (CPU)
Tier 1  check_uq                  ~0.1 s/structure (GPU)
Tier 2  check_ehull               ~10 s first call per chemsys, ~0 s cached (GPU)
Tier 3  check_cross_mlip          ~0.5 s/structure  (if run_cross=True) (GPU)
Tier 4  check_phonon_stability    ~2 s/structure (GPU)
Tier 5  check_md_stability        ~3 s/structure (if run_md=True) (GPU)
```

Returns a merged dict with all check results and `"overall_pass": bool`.

---

## Energy-above-hull approach

`check_ehull` builds an **all-MLIP phase diagram**: it fetches competing-phase structures
from the Materials Project via `MPRester`, re-evaluates each with the same MLIP calculator,
then constructs the convex hull from those MLIP energies. This avoids the energy-scale
mismatch between raw MLIP predictions and DFT-corrected MP entries (GGA/GGA+U mixing
corrections, oxide corrections) which otherwise causes errors of 1–5 eV/atom for
transition-metal oxide systems.

A `cache` dict keyed by `frozenset(elements)` is shared across calls so the expensive
MPRester query and bulk re-evaluation only runs once per chemical system.

---

## Standalone usage (any project)

```python
from validity.structural import load_pet, load_mace, run_checks

pet  = load_pet(device="cuda")
mace = load_mace(device="cuda")
cache = {}  # shared across multiple structures in the same chemsys

result = run_checks(atoms, pet, mace, ehull_cache=cache)
print(result["overall_pass"])       # True/False
print(result["e_above_hull"])       # eV/atom
print(result["n_imaginary_modes"])  # int
```

Run individual checks directly for more control:

```python
from validity.structural import check_ehull, check_phonon_stability

ehull = check_ehull(atoms, pet, threshold=0.15, cache=cache)
# {"e_above_hull": 0.08, "dft_e_above_hull": 0.095, "decomposition": {...}, "ehull_pass": True}

phonon = check_phonon_stability(atoms, mace, threshold_thz=-0.3)
# {"n_imaginary_modes": 0, "most_negative_freq_thz": 1.2, "phonon_pass": True}
```

Composition-level checks (CPU-only, no Atoms object needed):

```python
from validity.structural import passes_chemistry_filters, is_charge_balanced

passes_chemistry_filters("BaTiO3")  # True
is_charge_balanced("BaTiO3")        # True
```

---

## Pipeline usage (stage 7)

`pipeline/stage7.py` is the batch runner used in `pipeline/pipeline_eval.py`. It runs on
the Pareto-front structures produced by stage 6.

```python
from pipeline.stage7 import run_stage7

passing_atoms, passing_rows = run_stage7(
    atoms_list=relaxed_atoms,   # list[Atoms] from stage6_relax_and_props
    rows=metadata_rows,          # list[dict] with at least "composition_formula"
    output_dir="results/stage7",
    device="cuda",
    run_md=False,   # enable for extra soft-flag check (~3 s/structure)
    run_cross=True,
)
```

Writes:
- `results/stage7/stage7_out.csv` — all structures with all check columns
- `results/stage7/structures/*.extxyz` — passing structures only

`pipeline_eval.py` calls this automatically; the JSON output includes `"validity_pass": bool`
for each Pareto-front entry.

---

## Output columns (stage7_out.csv)

| Column | Source |
|---|---|
| `uq_eV_pa` | MLIP uncertainty (eV/atom); `None` if unsupported |
| `uq_pass` | bool |
| `e_above_hull` | MLIP hull distance (eV/atom) |
| `dft_e_above_hull` | DFT reference from `atoms.info`, if present |
| `decomposition` | Competing phases at hull |
| `ehull_pass` | bool |
| `n_imaginary_modes` | Number of optic modes below threshold |
| `most_negative_freq_thz` | Lowest optic mode frequency (THz) |
| `phonon_pass` | bool |
| `energy_pet_eV_pa` | PET single-point (eV/atom) |
| `energy_mace_eV_pa` | MACE single-point (eV/atom) |
| `cross_delta_eV_pa` | \|PET − MACE\| (eV/atom) |
| `cross_pass` | bool |
| `md_max_disp_A` | Max per-atom displacement over MD run (Å) |
| `md_volume_drift_pct` | Volume change relative to t=0 (%) |
| `md_pass` | bool |
| `overall_pass` | True iff all hard-reject checks pass |

---

## Dependencies

All already installed in the environment:
- `upet` — `UPETCalculator` (pet-oam-xl v1.0.0)
- `mace-torch` — `MACECalculator`, model at `/data/assets/checkpoints/mace/mace-mp-0b3-medium.model`
- `pymatgen` — `PhaseDiagram`, `MPRester`, `Composition`, `Element`
- `ase` — `Vibrations`, `Langevin`
- `smact` — charge-balance / electronegativity screening

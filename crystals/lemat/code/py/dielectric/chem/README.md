# chem — Materials Chemistry Utilities

Reusable utilities for dielectric materials discovery: property prediction,
structure relaxation, composition expansion, and validity screening.

## Package Layout

```
chem/
    __init__.py                   # Lightweight re-exports (no heavy deps)

    # Standalone modules (small, no internal deps, frequently imported)
    validity.py                   # SMACT + BVS + atomic-distance, with timeout
    novelty.py                    # StructureMatcher vs LeMat-Bulk (SQLite + parquet)
    stability.py                  # MACE e_above_hull with disk cache
    _timeout.py                   # SIGALRM wrapper used by the three above
    check_validity.py             # legacy is_realistic()
    pareto_front.py               # ParetoFront class
    material_classifier.py        # classify_material()
    flag_dielectric.py            # flag_dielectric()
    prototypes.py                 # choose_prototype(), is_perovskite_like(), etc.
    composition_features.py       # Magpie featurization, electronegativity, entropy

    # Subpackage: surrogates/ (ML surrogate inference, CPU)
    surrogates/
        soap.py                   # SOAP+XGBoost property surrogates
        eps0.py                   # Dedicated eps_0 surrogate
        ehull.py                  # Energy-above-hull surrogate

    # Subpackage: props/ (GPU property calculators)
    props/
        relax.py                  # MLIP relaxation (MACE, ORB, SevenNet)
        physics.py                # Phonons, dielectric tensors, mechanical
        dielectric.py             # Full dielectric property pipeline
        stage6.py                 # Stage-6 relaxation + property evaluation

    # Subpackage: gen/ (composition & structure generation)
    gen/
        sampler.py                # Composition statistics & sampling
        generator.py              # 7-step composition generator
        expander.py               # Prototype substitution expander
        structures.py             # Structure generation from compositions

    # Subpackage: analysis/ (plotting, clustering, stats)
    analysis/
        plot.py                   # Pareto plots, band gap vs eps_0
        cluster.py                # Top-quadrant clustering

    # Subpackage: pipeline/ (orchestration, stages 1-6)
    pipeline/
        orchestrator.py           # Full pipeline (stages 1-5)
        evaluator.py              # Stage-6 evaluation from CSV

    # Training scripts (not importable, no __init__.py)
    scripts/
        train_soap.py             # Train SOAP surrogate models
        train_composition.py      # Train composition-based models
        train_eps0.py             # Train eps_0 outlier-weighted model
        compute_stats.py          # Compute training dataset statistics

    data -> ../pipeline/data      # Symlink to shared data
```

## Installation

```bash
cd /data/rkumar/code/py/dielectric
pip install -e .
```

## Quick Start

### Three-axis screening (validity, novelty, stability)

These three modules are the canonical entry points for screening a single
`ase.Atoms`. Each call has a hard 30-second SIGALRM timeout; all artifact
paths are parameters with sensible defaults so the same code works on the
local box and on EC2.

```python
from chem.validity import check_validity
from chem.novelty import check_lemat_novelty
from chem.stability import compute_e_above_hull, load_stability_calc

# 1. Validity — SMACT + BVS + minimum-distance check (no MLIP, ~30 ms)
v = check_validity(atoms, timeout=30)
# {'min_dist_A': 2.82, 'volume_A3': 44.9, 'min_dist_pass': True,
#  'smact_pass': True, 'bvs_max_deviation': 0.0, 'bvs_pass': True,
#  'validity_pass': True}

# 2. Lemat-novelty — StructureMatcher vs LeMat-Bulk (SQLite + parquet)
is_novel = check_lemat_novelty(atoms, timeout=30)         # bool
# Or for a batch (amortizes the index load):
flags = check_lemat_novelty([a1, a2, a3])                 # list[bool]

# 3. Stability — e_above_hull via MACE-evaluated phase diagram
calc = load_stability_calc(device="cuda")  # share across many calls
mem_cache = {}                              # share across many calls
s = compute_e_above_hull(atoms, calc=calc, mem_cache=mem_cache, timeout=30)
# {'e_above_hull': 0.0044, 'ehull_pass': True, 'decomposition': {...}}
```

#### Overriding paths

```python
# Use a different MACE checkpoint or LeMat parquet
compute_e_above_hull(atoms, mlip_path="/path/to/other.model",
                     cache_dir="/tmp/my_ehull_cache")
check_lemat_novelty(atoms, parquet_path="/path/to/lematbulk.parquet",
                    sqlite_path="/path/to/composition_index.sqlite")
```

#### Reusable checker (batches, custom budgets)

```python
from chem.novelty import LematNoveltyChecker

ck = LematNoveltyChecker.from_assets(
    parquet_path="/data/assets/datasets/lematbulk/lematbulk_compatible_pbe.parquet",
    sqlite_path="/data/assets/datasets/lematbulk/composition_index.sqlite",  # None → pandas fallback
    ltol=0.1,
    max_candidates=50,         # cap on structural comparisons per call
    time_budget_s=25,          # optional wall-clock cap (returns True if exceeded)
)
print(type(ck.backend).__name__)   # _SQLiteBackend (preferred) or _PandasBackend
ck.is_novel(atoms)
ck.check_batch([a1, a2, a3])
```

The SQLite backend keeps peak RAM under ~500 MB regardless of dataset size
(streaming PyArrow row-group reads). The pandas fallback loads the full df
into RAM (~15 GB on LeMat-Bulk) — only use it when SQLite is unavailable.

#### Smoke + latency test

```bash
PYTHONPATH=dielectric python dielectric/chem/tests/test_three_axes.py
```

Runs all three axes on 10 representative compositions and prints per-call
timing. Useful for regression-checking after a checkpoint or dataset swap.

### Screening and classification (legacy helpers)

```python
from chem import is_realistic, ParetoFront, classify_material, flag_dielectric

ok, reasons = is_realistic(atoms)
category = classify_material("BaTiO3")
warning = flag_dielectric({"eps_0": 150, "bandgap": 2.5, "family": "oxide"})
```

### SOAP surrogate properties

```python
from chem.surrogates import predict_properties, load_structures_from_extxyz

atoms_list = load_structures_from_extxyz("/path/to/structures/*.extxyz")
preds = predict_properties(atoms_list)  # {"dft_band_gap": [...], "dft_eps_0": [...]}
```

### Dedicated eps_0 surrogate

```python
from chem.surrogates import predict_eps_0_fast

eps_0 = predict_eps_0_fast(atoms)
```

### Energy above hull (surrogate)

```python
from chem.surrogates import mlip_energy_above_hull_surrogate

ehulls = mlip_energy_above_hull_surrogate(atoms_list)
```

### Relax structures with MLIP

```python
from chem.props import get_calculator, relax_structure

calc = get_calculator("mace", model=None, device="cuda")
relaxed, converged, steps, energy = relax_structure(atoms, calc)
```

### Compute dielectric properties

```python
from chem.props import Dielectrics, compute_eps_0

# Quick: surrogate eps_0
eps_0 = compute_eps_0(atoms)

# Full: phonon-based dielectric tensor
with Dielectrics() as di:
    results = di.compute(atoms_list)
```

### Composition generation and expansion

```python
from chem.gen import CompositionGenerator, expand_many

# Generate candidates from statistics
gen = CompositionGenerator()
compositions = gen.generate(n=1000)

# Expand via prototype substitution
expanded = expand_many(["BaTiO3"], return_pareto=True)
```

### Structure generation

```python
from chem.gen import generate_structures_for_compositions_topk

structures = generate_structures_for_compositions_topk(compositions, dataset)
```

### Pareto front analysis

```python
from chem import ParetoFront

pf = ParetoFront(maximize=["eps_0"], minimize=["ehull"])
front = pf.filter(candidates)
```

### Plotting and clustering

```python
from chem.analysis import plot_pareto_points, cluster_topq
```

### Pipeline orchestration

```python
from chem.pipeline import run_pipeline, run_stage6_from_csv
```

## CLI Tools

```bash
# Relax structures
python -m chem.props.relax --input refined.extxyz --output relaxed.extxyz

# Run full pipeline
python -m chem.pipeline.orchestrator --clusters topq_clusters.csv

# Run stage-6 evaluation
python -m chem.pipeline.evaluator --stage5-csv stage5.csv

# Screen generated structures (validity package)
python -m validity.screen_generated structures.extxyz --full
```

## MACE Relaxation: e3nn/PyTorch Compatibility Fix

### Problem

MACE checkpoints (e.g. `mace-mpa-0-medium.model`) were saved with an older
PyTorch + e3nn. When loaded with PyTorch 2.10 + e3nn 0.5.7, two things break:

1. **Unpickling error** in `e3nn/util/codegen/_mixin.py`: Old checkpoints store
   compiled TorchScript as raw `bytes`, but e3nn 0.5.7 expects `(buffer_type, bytes)`
   tuples → `ValueError: too many values to unpack`

2. **Missing compiled submodules**: Even after fixing unpickling, the compiled
   TorchScript IR from older PyTorch doesn't work at runtime →
   `AttributeError: 'Linear' object has no attribute '_compiled_main'` and
   `'TensorProduct' object has no attribute '_compiled_main_left_right'`

3. **Device mismatch** (GPU only): Regenerated codegen creates constant tensors
   on CPU while model is on CUDA → `RuntimeError: tensors on different devices`

### Fix (two files)

**File 1: `e3nn/util/codegen/_mixin.py`** (in the e3nn package)

In `__setstate__`, skip restoring compiled code entirely and just record the
codegen names. In `__getstate__`, use `pop()` instead of `del` for robustness:

```python
# In __setstate__ — replace the for loop with:
if codegen_state is not None:
    # Skip restoring compiled code from incompatible PyTorch version
    self.__codegen__ = list(codegen_state.keys())

# In __getstate__ — change del to pop:
out["_modules"].pop(fname, None)  # was: del out["_modules"][fname]
```

Find the file with:
```bash
python3 -c "import e3nn.util.codegen._mixin as m; print(m.__file__)"
```

Clear pycache after editing:
```bash
find $(python3 -c "import e3nn; print(e3nn.__path__[0])") -name __pycache__ -exec rm -rf {} +
```

**File 2: `chem/props/relax.py`** — `get_calculator()` function

After loading the model, regenerate compiled code for all `Linear` and
`TensorProduct` modules, then move models to the correct device:

```python
from e3nn.o3._linear import Linear as E3nnLinear, _codegen_linear
from e3nn.o3._tensor_product import TensorProduct as E3nnTP
from e3nn.o3._tensor_product._codegen import codegen_tensor_product_left_right, codegen_tensor_product_right

# After MACECalculator(...):
for m in calc.models:
    for mod in m.modules():
        # ... existing SphericalHarmonics and Activation patches ...
        if isinstance(mod, E3nnLinear) and not hasattr(mod, "_compiled_main"):
            graphmod, _, _ = _codegen_linear(
                mod.irreps_in, mod.irreps_out, mod.instructions,
                shared_weights=mod.shared_weights,
                optimize_einsums=mod._optimize_einsums,
            )
            mod._codegen_register({"_compiled_main": graphmod})
        if isinstance(mod, E3nnTP) and not hasattr(mod, "_compiled_main_left_right"):
            code_lr = codegen_tensor_product_left_right(
                mod.irreps_in1, mod.irreps_in2, mod.irreps_out,
                mod.instructions,
                shared_weights=mod.shared_weights,
                specialized_code=mod._specialized_code,
                optimize_einsums=mod._optimize_einsums,
            )
            codegen = {"_compiled_main_left_right": code_lr}
            # ... also _compiled_main_right if applicable ...
            mod._codegen_register(codegen)
    m.to(device)  # Move regenerated constants to correct device
```

### Verifying the Fix

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from chem.props.relax import get_calculator, relax_structure
from ase import Atoms
calc = get_calculator('mace', None, 'cuda')  # or 'cpu'
atoms = Atoms('NaCl', positions=[[0,0,0],[2.82,0,0]], cell=[5.64]*3, pbc=True)
r, conv, steps, en = relax_structure(atoms, calc, fmax=0.05, max_steps=50, timeout=60)
print(f'converged={conv} steps={steps} energy={en:.4f}')
# Expected: converged=True steps=~24 energy=~-5.53
"
```

### Impact

All screening results before 2026-03-19 used **broken relaxation** (0%
convergence). Structures were screened unrelaxed, inflating pass rates. The old
baseline's reported 71% e-hull dropped to 9.6% with proper relaxation. Always
verify `Converged: N/M` in screening output — if 0%, the fix hasn't been applied.

## Training Scripts

Training scripts live in `chem/scripts/` and are meant to be run directly:

```bash
# Train SOAP surrogate models
python chem/scripts/train_soap.py

# Train composition-based models
python chem/scripts/train_composition.py

# Train eps_0 outlier-weighted model
python chem/scripts/train_eps0.py --extxyz /path/to/data.extxyz

# Compute training statistics
python chem/scripts/compute_stats.py --extxyz "/path/to/mp/*.extxyz"
```

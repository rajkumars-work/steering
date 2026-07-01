# Synthetic Source Generator

Generate realistic source prompts for the crystal structure model without
needing the training data. Only a JSON statistics file is required.

## Overview

The model takes 4-segment source prompts:
```
elements | natoms | density | prop_labels
```
For example: `Ba Mg S V | 11 | 3.9 | metal stable`

This module extracts distributional statistics from a dataset CSV, serializes
them to JSON, and generates synthetic sources that match the original
distribution. This enables:

1. **Inference without training data** — deploy the model with just a stats file
2. **Source ablation experiments** — compare model behavior with different source distributions
3. **Out-of-distribution testing** — generate prompts from a different composition space

## Quick Start

```python
from chem.source_gen import SourceStatistics, SourceGenerator

# Extract statistics from a dataset (one-time)
stats = SourceStatistics.from_csv("data/d13_mixed_lemat_ehull.csv", split="train")
stats.to_json("data/stats/inset_stats.json")

# Generate synthetic sources (only needs the JSON)
stats = SourceStatistics.from_json("data/stats/inset_stats.json")
gen = SourceGenerator(stats, seed=42)

# Get a list of source strings
sources = gen.generate(250)

# Or write directly to a file (one source per line)
gen.generate_source_file("my_sources.txt", 250)
```

## Use with Screening Pipeline

```bash
# Screen with synthetic sources
python scripts/run_screen.py \
    --ckpt ckpt_d13_mixed_lemat_ehull_250ep \
    --data data/d13_mixed_lemat_ehull.csv \
    --sources-file my_sources.txt \
    --n 250 --gpu 0
```

The `--sources-file` flag loads sources from a text file instead of the dataset CSV.

## What Statistics Are Captured

| Statistic | Purpose | Serialized as |
|-----------|---------|---------------|
| Element frequencies | Weighted element sampling | `{element: count}` |
| N-elements distribution | How many elements per source | `{n: count}` |
| Natoms histogram | Empirical (multimodal, not parametric) | `{natoms: count}` |
| Per-element density | Element-density correlation | `{element: {mean, std}}` |
| Global density | Fallback for unknown elements | `{mean, std, min, max}` |
| Tag frequencies | Property label distribution | `{tag: count}` |
| Tag co-occurrence | Valid tag combinations | `{"tag1 tag2": count}` |

### Density Conditioning

Density is sampled conditioned on the chosen elements, not independently.
For each element, we store mean/std density from the training data. At
generation time:
1. Average the per-element means for the chosen element set
2. Pool the variances and scale by 1/sqrt(n_elements)
3. Sample from Normal, clip to [min, max]

This captures the main correlation (heavy elements → high density) without
needing a full joint distribution model.

## Pre-computed Statistics

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `data/stats/d13_inset_stats.json` | d13_mixed_lemat_ehull.csv (train) | 131K | Mixed-origin structures with ehull (despite legacy "nolemat" name, ~85% lemat-bulk) |
| `data/stats/d13_outset_stats.json` | d13_outset.csv | 1.22M | LeMat-Bulk-known structures |

## Source Experiments

See `docs/BenchmarkAnalysis.md` for the source prompt ablation experiment
comparing 5 source types (train, eval, OOD, synthetic in-dist, synthetic
out-dist) using the `SourceGenerator`.

Experiment data: `experiments/source_exp/`
Runner: `scripts/run_source_experiments.sh`
Data prep: `scripts/prepare_source_experiments.py`

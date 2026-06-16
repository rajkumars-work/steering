"""Rebalance DPO dataset to emphasise stability pairs for E-hull improvement.

Current distribution (dpo_combined_nopf.csv):
  stability:          42,524 (55.5%)
  structural_quality: 33,129 (43.3%)  -- bvs, min_dist, density, isolated
  composition:           918 ( 1.2%)  -- stoich, atom_count, element mismatch

Strategy: upweight stability to ~75%, keep all composition, downsample
structural quality.  This focuses the DPO signal on thermodynamic
stability (E-hull) while retaining enough structural quality pairs
to maintain BVS/geometry.

Usage:
    python rebalance_dpo.py [--stability_frac 0.75] [--seed 42]
    # Writes data/dpo_rebalanced.csv
"""

import argparse
import csv
import random
import sys

csv.field_size_limit(sys.maxsize)

STABILITY_TAGS = {
    'stability_metastable_vs_stable', 'stability_unstable_vs_stable',
    'stability_unstable_vs_metastable', 'cross_db_stability_jarvis_vs_mp',
    'cross_db_stability_mp_vs_jarvis', 'cross_db_stability_alex_vs_mp',
    'cross_db_stability_alex_vs_jarvis',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/dpo_combined_nopf.csv')
    p.add_argument('--output', default='data/dpo_rebalanced.csv')
    p.add_argument('--stability_frac', type=float, default=0.75,
                   help='Target fraction of stability pairs (default 0.75)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    stability_rows = []
    other_rows = []

    with open(args.input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = set(t.strip() for t in row['score'].split('+'))
            if tags & STABILITY_TAGS:
                stability_rows.append(row)
            else:
                other_rows.append(row)

    n_stability = len(stability_rows)
    # Target: stability_frac of total -> solve for n_other
    # n_stability / (n_stability + n_other) = frac
    # n_other = n_stability * (1 - frac) / frac
    n_other_target = int(n_stability * (1 - args.stability_frac) / args.stability_frac)
    n_other_target = min(n_other_target, len(other_rows))

    sampled_other = random.sample(other_rows, n_other_target)
    combined = stability_rows + sampled_other
    random.shuffle(combined)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'chosen', 'rejected', 'score'])
        writer.writeheader()
        writer.writerows(combined)

    total = len(combined)
    actual_frac = n_stability / total
    print(f'Stability pairs:          {n_stability:,}')
    print(f'Other pairs (sampled):    {n_other_target:,} (from {len(other_rows):,})')
    print(f'Total:                    {total:,}')
    print(f'Stability fraction:       {actual_frac:.1%}')
    print(f'Written to:               {args.output}')


if __name__ == '__main__':
    main()

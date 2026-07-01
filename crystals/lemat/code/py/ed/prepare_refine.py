"""Prepare stage-1 output for stage-2 refine models.

Reads an extxyz file from stage-1 inference and produces a CSV ready for
ed_translate.py, formatted for the specified refine variant.

Usage:
    python prepare_refine.py --input stage1.extxyz --variant v1 --output refine_input.csv
    python prepare_refine.py --input stage1.extxyz --variant v2 --output refine_input.csv

Variants:
    v1  - Direct: prepend 'refine | ' to scratchpad target (model expects scratchpad)
    v2  - Strip scratchpad: remove CN/NN/WP section, prepend 'refine | '
    v3  - Same source as v2 (model outputs numbers only; reconstruction done post-inference)
    v4  - Same source as v2 (model outputs corrections; reconstruction done post-inference)
"""
import argparse
import csv
import re
import sys

from ase.io.extxyz import read_extxyz


def strip_scratchpad(target_str):
    """Remove the CN:... NN:... WP:... section from a scratchpad-format target.

    Input:  'Formula | SG N Sym | CN:... NN:... WP:... | a b c α β γ | atoms...'
    Output: 'Formula | SG N Sym | a b c α β γ | atoms...'
    """
    parts = target_str.split(" | ")
    filtered = []
    for p in parts:
        # Skip parts that look like scratchpad (start with CN: or contain NN: WP:)
        if p.startswith("CN:") or (p.startswith("CN:") and "NN:" in p):
            continue
        filtered.append(p)
    return " | ".join(filtered)


def validate_source(source, variant):
    """Basic format validation. Returns (ok, reason)."""
    parts = source.split(" | ")

    if not source.startswith("refine | "):
        return False, "missing 'refine | ' prefix"

    # After prefix, expect: formula | SG ... | [scratchpad |] lattice | atoms
    inner = source[len("refine | "):]
    inner_parts = inner.split(" | ")

    if len(inner_parts) < 3:
        return False, f"too few sections ({len(inner_parts)}, need ≥3)"

    # Check SG section exists
    has_sg = any(p.startswith("SG ") for p in inner_parts)
    if not has_sg:
        return False, "no SG section found"

    if variant == "v1":
        # Should have scratchpad
        has_scratch = any(p.startswith("CN:") for p in inner_parts)
        if not has_scratch:
            return False, "v1 expects scratchpad but none found"
    else:
        # Should NOT have scratchpad
        has_scratch = any(p.startswith("CN:") for p in inner_parts)
        if has_scratch:
            return False, f"{variant} should not have scratchpad"

    return True, "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Stage-1 extxyz output")
    p.add_argument("--variant", default='v1', choices=["v1", "v2", "v3", "v4"],
                   help="Refine variant")
    p.add_argument("--output", required=True, help="Output CSV for ed_translate.py")
    p.add_argument("--max", type=int, default=None, help="Max structures to process")
    args = p.parse_args()

    ok = 0
    skipped = 0
    errors = []

    with open(args.output, "w", newline="") as fout, \
         open(args.input, "r") as fin:
        writer = csv.writer(fout)
        writer.writerow(["source", "target"])

        for i, atoms in enumerate(read_extxyz(fin, index=slice(None))):
            if args.max and ok >= args.max:
                break

            target_str = atoms.info.get("target", "")
            if not target_str:
                skipped += 1
                continue

            if args.variant == "v1":
                source = f"refine | {target_str}"
            else:
                # v2, v3, v4 all use the same source format (no scratchpad)
                stripped = strip_scratchpad(target_str)
                source = f"refine | {stripped}"

            # Validate
            valid, reason = validate_source(source, args.variant)
            if not valid:
                if len(errors) < 5:
                    errors.append(f"  Structure {i}: {reason}")
                skipped += 1
                continue

            writer.writerow([source, ""])
            ok += 1

    print(f"Wrote {ok} refine inputs ({args.variant}) to {args.output}")
    if skipped:
        print(f"Skipped {skipped} structures")
    if errors:
        print("First errors:")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()

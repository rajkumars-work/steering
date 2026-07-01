"""Convert stage-1 extxyz output to refine model input CSV.

Reads the 'target' field from extxyz info and prepends 'refine | '.
No recomputation needed — uses the target string as-is from stage-1.
"""
import argparse
import csv

from ase.io.extxyz import read_extxyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input extxyz from stage-1")
    p.add_argument("--output", required=True, help="Output CSV for refine model")
    args = p.parse_args()

    ok = 0
    total = 0
    with open(args.output, "w", newline="") as fout, \
         open(args.input, "r") as fin:
        writer = csv.writer(fout)
        writer.writerow(["source", "target"])

        for atoms in read_extxyz(fin, index=slice(None)):
            total += 1
            target_str = atoms.info.get("target", "")
            if not target_str:
                print(f"  Skipped structure {total}: no target field")
                continue
            source = f"refine | {target_str}"
            writer.writerow([source, ""])
            ok += 1
            if ok % 100 == 0:
                print(f"  Processed {ok} structures...")
                fout.flush()

    print(f"Wrote {ok}/{total} refine inputs to {args.output}")


if __name__ == "__main__":
    main()

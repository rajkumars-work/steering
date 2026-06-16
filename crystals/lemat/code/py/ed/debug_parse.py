"""Quick diagnostic: why are DPO outputs failing to parse?"""
import csv
import sys

sys.path.insert(0, ".")
from ed_translate import parse_target

with open("/tmp/dpo_debug.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        tgt = row["target"]
        atoms = parse_target(tgt)
        status = "OK" if atoms else "FAIL"
        parts = tgt.split(" | ")
        print(f"[{i}] {status}  pipes={len(parts)}  len={len(tgt)}")
        if status == "FAIL":
            if len(parts) != 3:
                print(f"    Wrong pipe count: {len(parts)}")
                print(f"    Text: {tgt[:300]}...")
            else:
                print(f"    Formula: {parts[0][:60]}")
                print(f"    Lattice: {parts[1][:80]}")
                atom_toks = parts[2].split()
                print(f"    Atom tokens: {len(atom_toks)} (mod 4 = {len(atom_toks) % 4})")
                if len(atom_toks) % 4 != 0:
                    print(f"    Last 10 tokens: {atom_toks[-10:]}")

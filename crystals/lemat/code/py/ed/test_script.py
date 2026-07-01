"""Pick random rows from small.csv, run them through the trained model,
and print source, reference target, and model-predicted target.
Also reconstruct ase.Atoms objects from both reference and predicted targets.

With --n 0, enter interactive mode: type a source string, get a prediction."""

import argparse
import csv
import random

import numpy as np
import torch
from ase import Atoms
from ase.geometry import cellpar_to_cell

from configs import Config, get_device
from ed_data import load_sp
from ed_model import EdGPT
from ed_train import generate


def parse_target(target_str):
    """Reverse format_target: parse 'Formula | a b c alpha beta gamma | Elem x y z ...'
    into an ase.Atoms object. Returns None if parsing fails."""
    parts = target_str.split(" | ")
    if len(parts) != 3:
        return None
    try:
        # Lattice parameters
        lat_tokens = parts[1].split()
        a, b, c = float(lat_tokens[0]), float(lat_tokens[1]), float(lat_tokens[2])
        alpha, beta, gamma = float(lat_tokens[3]), float(lat_tokens[4]), float(lat_tokens[5])
        cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])

        # Atom list: "Elem x y z Elem x y z ..."
        atom_tokens = parts[2].split()
        symbols = []
        frac_coords = []
        for j in range(0, len(atom_tokens), 4):
            symbols.append(atom_tokens[j])
            frac_coords.append([
                float(atom_tokens[j + 1]),
                float(atom_tokens[j + 2]),
                float(atom_tokens[j + 3]),
            ])

        atoms = Atoms(
            symbols=symbols,
            scaled_positions=np.array(frac_coords),
            cell=cell,
            pbc=True,
        )
        return atoms
    except (ValueError, IndexError):
        return None

DATA_CSV = "../dielectric/data/small.csv"
CHECKPOINT = "checkpoints/ed_ckpt_final.pt"
SP_MODEL = "checkpoints/model_sp.model"


def print_result(src, ref, pred, index=None):
    """Print a single source/reference/prediction result with Atoms info."""
    ref_atoms = parse_target(ref) if ref else None
    pred_atoms = parse_target(pred)

    print(f"\n{'='*80}")
    if index is not None:
        print(f"Sample {index}")
    print(f"  Source:    {src}")
    if ref is not None:
        print(f"  Reference: {ref}")
    print(f"  Predicted: {pred}")

    if ref_atoms:
        print(f"  Ref  Atoms: {ref_atoms.get_chemical_formula()} "
              f"({len(ref_atoms)} atoms, vol={ref_atoms.get_volume():.1f} A^3)")
    elif ref is not None:
        print(f"  Ref  Atoms: PARSE FAILED")

    if pred_atoms:
        print(f"  Pred Atoms: {pred_atoms.get_chemical_formula()} "
              f"({len(pred_atoms)} atoms, vol={pred_atoms.get_volume():.1f} A^3)")
    else:
        print(f"  Pred Atoms: PARSE FAILED")


def main():
    parser = argparse.ArgumentParser(description="Test encoder-decoder model on crystal structures")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of random samples (0 = interactive mode)")
    args = parser.parse_args()

    # Load tokenizer
    sp = load_sp(SP_MODEL)

    # Load model from checkpoint
    device = torch.device(get_device())
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = EdGPT(config).to(device)
    # Strip "_orig_mod." prefix added by torch.compile
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    if args.n == 0:
        # Interactive mode
        print("Interactive mode. Enter a source string (empty to quit).")
        while True:
            try:
                src = input("\nSource> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not src:
                break
            pred = generate(model, sp, src, device, max_length=config.tdim)
            print_result(src, None, pred)
    else:
        # Random sample mode
        with open(DATA_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        samples = random.sample(rows, min(args.n, len(rows)))

        for i, row in enumerate(samples, 1):
            src = row["source"]
            ref = row["target"]
            pred = generate(model, sp, src, device, max_length=config.tdim)
            print_result(src, ref, pred, index=i)

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

"""MLIP relaxation of crystal structures.

Runs geometry optimization (atomic positions + cell) using a machine-learning
interatomic potential.  Intended as a post-processing step after stage-2 flow
matching refinement, to push structures into local energy minima before
validity screening.

Usage
-----
# Default: MACE-MP-0 medium, 200 steps max
python relax.py --input refined.extxyz --output relaxed.extxyz

# Use a different MACE model
python relax.py --input refined.extxyz --output relaxed.extxyz \
    --calculator mace --model /data/assets/checkpoints/mace/mace-mpa-0-medium.model

# Use ORB
python relax.py --input refined.extxyz --output relaxed.extxyz --calculator orb

# Filter-relax-screen pipeline
python sample.py --input rough.extxyz --checkpoint checkpoints/epoch_75.pt --output refined.extxyz
python relax.py --input refined.extxyz --output relaxed.extxyz
python -m validity.screen_generated relaxed.extxyz --full
"""
from __future__ import annotations

import argparse
import io
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read, write as ase_write
# ASE 3.24+ removed ExpCellFilter; FrechetCellFilter is the recommended
# replacement. Prefer the new API where available.
try:
    from ase.filters import FrechetCellFilter as ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from ase.optimize import FIRE, LBFGS

# ---------------------------------------------------------------------------
# e3nn compatibility shim (same as validity/structural.py) — old MACE
# checkpoints store TorchScript bytes directly as {fname: bytes}, but
# e3nn >=0.5.7 expects {fname: (buffer_type, buffer)}.
# ---------------------------------------------------------------------------
from e3nn.util.codegen._mixin import CodeGenMixin as _CodeGenMixin


def _codegen_setstate_compat(self, d: dict) -> None:
    d = d.copy()
    codegen_state = d.pop("__codegen__", None)
    if hasattr(super(_CodeGenMixin, self), "__setstate__"):
        super(_CodeGenMixin, self).__setstate__(d)
    else:
        self.__dict__.update(d)
    if codegen_state is not None:
        for fname, value in codegen_state.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                buffer_type, buffer = value
            else:
                buffer_type, buffer = "torchscript", value
            if buffer_type == "fx":
                smod = pickle.loads(buffer)
            elif buffer_type == "torchscript":
                smod = torch.jit.load(io.BytesIO(buffer))
            else:
                raise NotImplementedError(f"Unknown codegen buffer_type: {buffer_type!r}")
            setattr(self, fname, smod)
        self.__codegen__ = list(codegen_state.keys())


_CodeGenMixin.__setstate__ = _codegen_setstate_compat

# Default model paths
MACE_MODELS = {
    "mp-0b3": "/data/assets/checkpoints/mace/mace-mp-0b3-medium.model",
    "mpa-0": "/data/assets/checkpoints/mace/mace-mpa-0-medium.model",
    "omat-0": "/data/assets/checkpoints/mace/mace-omat-0-medium.model",
    "matpes-pbe": "/data/assets/checkpoints/mace/MACE-matpes-pbe-omat-ft.model",
    "matpes-r2scan": "/data/assets/checkpoints/mace/MACE-matpes-r2scan-omat-ft.model",
    "alex": "/data/assets/checkpoints/mace/2024-12-03-mace-mp-alex-0.model",
}
ORB_MODEL = "/data/assets/checkpoints/orb/orb-v3-direct-20-omat-20250404.ckpt"
SEVENNET_MODEL = "/data/assets/checkpoints/sevennet/checkpoint_sevennet_omat.pth"


def get_calculator(name: str, model: str | None, device: str):
    """Instantiate an ASE calculator by name."""
    if name == "mace":
        from mace.calculators import MACECalculator
        from e3nn import get_optimization_defaults
        from e3nn.nn._activation import Activation
        from e3nn.o3._spherical_harmonics import SphericalHarmonics, _spherical_harmonics

        from e3nn.o3._linear import Linear as E3nnLinear, _codegen_linear
        from e3nn.o3._tensor_product import TensorProduct as E3nnTP
        from e3nn.o3._tensor_product._codegen import codegen_tensor_product_left_right, codegen_tensor_product_right

        model_path = model or MACE_MODELS["mpa-0"]
        calc = MACECalculator(
            model_paths=[model_path], device=device, default_dtype="float64",
        )

        # Regenerate all compiled code — the checkpoint's serialized
        # TorchScript is incompatible with the current PyTorch version.
        jit_mode = get_optimization_defaults().get("jit_mode", "default")
        for m in calc.models:
            for mod in m.modules():
                if isinstance(mod, SphericalHarmonics) and not hasattr(mod, "sph_func"):
                    if jit_mode == "script":
                        mod.sph_func = torch.jit.script(_spherical_harmonics)
                    elif jit_mode == "compile":
                        mod.sph_func = torch.compile(_spherical_harmonics, fullgraph=True)
                    else:
                        mod.sph_func = _spherical_harmonics
                if isinstance(mod, Activation) and not hasattr(mod, "paths"):
                    mod.paths = [
                        (mul, (l, p), act)
                        for (mul, (l, p)), act in zip(mod.irreps_in, mod.acts)
                    ]
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
                    if not hasattr(mod, "_compiled_main_right"):
                        try:
                            code_r = codegen_tensor_product_right(
                                mod.irreps_in1, mod.irreps_in2, mod.irreps_out,
                                mod.instructions,
                                shared_weights=mod.shared_weights,
                                specialized_code=mod._specialized_code,
                                optimize_einsums=mod._optimize_einsums,
                            )
                            codegen["_compiled_main_right"] = code_r
                        except Exception:
                            pass
                    mod._codegen_register(codegen)
        # Move regenerated codegen modules to the correct device
        for m in calc.models:
            m.to(device)
        return calc
    elif name == "orb":
        from orb_models.forcefield.calculator import ORBCalculator
        model_path = model or ORB_MODEL
        return ORBCalculator(checkpoint_path=model_path, device=device)
    elif name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        model_path = model or SEVENNET_MODEL
        return SevenNetCalculator(model=model_path, device=device)
    else:
        raise ValueError(f"Unknown calculator: {name!r}. Choose from: mace, orb, sevennet")


class _RelaxTimeout(Exception):
    """Raised when a single relaxation exceeds the wall-clock timeout."""


def _timeout_handler(signum, frame):
    raise _RelaxTimeout


def relax_structure(
    atoms,
    calc,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: str = "FIRE",
    relax_cell: bool = True,
    timeout: int | None = None,
):
    """Relax a single structure. Returns (relaxed_atoms, converged, n_steps, energy).

    Parameters
    ----------
    timeout : int or None
        Wall-clock seconds for this relaxation.  ``None`` means no timeout.
        Uses ``signal.SIGALRM`` (Unix only).
    """
    import copy
    atoms = copy.deepcopy(atoms)
    atoms.calc = calc

    if timeout:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        # Get initial energy
        try:
            e0 = float(atoms.get_potential_energy())
        except _RelaxTimeout:
            raise
        except Exception:
            return atoms, False, 0, None

        # Set up optimizer
        if relax_cell:
            ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
            opt_target = ecf
        else:
            opt_target = atoms

        Opt = FIRE if optimizer == "FIRE" else LBFGS
        opt = Opt(opt_target, logfile=None)

        try:
            converged = opt.run(fmax=fmax, steps=max_steps)
            n_steps = opt.nsteps
            energy = float(atoms.get_potential_energy())
        except _RelaxTimeout:
            raise
        except Exception:
            return atoms, False, 0, None

        return atoms, converged, n_steps, energy

    except _RelaxTimeout:
        n_steps = opt.nsteps if 'opt' in locals() else 0
        return atoms, False, n_steps, None

    finally:
        if timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def parse_args():
    p = argparse.ArgumentParser(description="MLIP relaxation of crystal structures")
    p.add_argument("--input", "-i", required=True, help="Input extxyz file")
    p.add_argument("--output", "-o", required=True, help="Output extxyz file")
    p.add_argument("--calculator", default="mace",
                   choices=["mace", "orb", "sevennet"],
                   help="Which MLIP to use (default: mace)")
    p.add_argument("--model", default=None,
                   help="Path to model checkpoint (uses default for chosen calculator if not set)")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="Force convergence criterion in eV/A (default: 0.05)")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Maximum optimization steps per structure (default: 200)")
    p.add_argument("--optimizer", default="FIRE", choices=["FIRE", "LBFGS"],
                   help="Optimizer (default: FIRE)")
    p.add_argument("--no-cell", action="store_true",
                   help="Only relax atomic positions, keep cell fixed")
    p.add_argument("--max-structures", type=int, default=None,
                   help="Only process first N structures")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading structures from {args.input} ...")
    atoms_list = ase_read(args.input, index=":")
    if args.max_structures:
        atoms_list = atoms_list[:args.max_structures]
    print(f"  {len(atoms_list)} structures loaded")

    print(f"Loading {args.calculator} calculator ...")
    calc = get_calculator(args.calculator, args.model, args.device)
    model_name = args.model or f"{args.calculator} (default)"
    print(f"  Model: {model_name}")

    relaxed_list = []
    n_converged = 0
    total_steps = 0
    t0 = time.time()

    for i, atoms in enumerate(atoms_list):
        formula = atoms.get_chemical_formula()
        r_atoms, converged, steps, energy = relax_structure(
            atoms, calc,
            fmax=args.fmax,
            max_steps=args.max_steps,
            optimizer=args.optimizer,
            relax_cell=not args.no_cell,
        )
        status = "converged" if converged else f"not converged ({steps} steps)"
        e_str = f"{energy:.4f} eV" if energy is not None else "failed"
        print(f"  [{i+1}/{len(atoms_list)}] {formula}: {status}, E={e_str}")

        relaxed_list.append(r_atoms)
        if converged:
            n_converged += 1
        total_steps += steps

    elapsed = time.time() - t0

    print(f"\nRelaxation complete in {elapsed:.1f}s")
    print(f"  Converged: {n_converged}/{len(atoms_list)} "
          f"({100*n_converged/len(atoms_list):.1f}%)")
    print(f"  Average steps: {total_steps/len(atoms_list):.1f}")

    # Strip calculator results to avoid shape mismatches when writing
    # mixed-size structures to a single extxyz file
    for atoms in relaxed_list:
        atoms.calc = None
        for key in list(atoms.arrays.keys()):
            if key not in ("numbers", "positions"):
                del atoms.arrays[key]

    print(f"Writing {len(relaxed_list)} structures to {args.output} ...")
    ase_write(args.output, relaxed_list, format="extxyz")
    print("Done.")


if __name__ == "__main__":
    main()

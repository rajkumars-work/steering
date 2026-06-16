"""MACE relaxation with incremental output and resume support."""
import argparse
import copy
import signal
import time
import warnings

warnings.filterwarnings("ignore")

from ase.filters import ExpCellFilter
from ase.io import read as ase_read, write as ase_write
from ase.optimize import FIRE

from validity.structural import load_mace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--timeout", type=int, default=60,
                   help="Per-structure timeout in seconds (default: 60)")
    args = p.parse_args()

    atoms_list = ase_read(args.input, index=f":{args.max}")
    print(f"Loaded {len(atoms_list)} structures")

    # Resume: check how many are already in the output file
    start_idx = 0
    relaxed = []
    try:
        existing = ase_read(args.output, index=":")
        if not isinstance(existing, list):
            existing = [existing]
        relaxed = existing
        start_idx = len(existing)
        print(f"Resuming from structure {start_idx} ({start_idx} already done)")
    except Exception:
        pass

    calc = load_mace(args.device)
    n_conv = 0
    t_total = time.time()

    for i in range(start_idx, len(atoms_list)):
        atoms = atoms_list[i]
        a = copy.deepcopy(atoms)
        a.calc = calc
        t0 = time.time()

        def _timeout_handler(signum, frame):
            raise TimeoutError("Structure relaxation timed out")

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(args.timeout)
            ecf = ExpCellFilter(a, hydrostatic_strain=False)
            opt = FIRE(ecf, logfile=None)
            converged = opt.run(fmax=args.fmax, steps=args.max_steps)
            signal.alarm(0)
            e = float(a.get_potential_energy())
            steps = opt.nsteps
            if converged:
                n_conv += 1
        except (Exception, TimeoutError) as ex:
            signal.alarm(0)
            converged = False
            steps = 0
            e = None

        dt = time.time() - t0
        natoms = len(atoms)
        status = "ok" if converged else f"{steps}s"
        e_str = f"{e:.3f}" if e is not None else "FAIL"
        print(f"  [{i + 1}/{len(atoms_list)}] {atoms.get_chemical_formula()} "
              f"({natoms} atoms): {status} {dt:.1f}s E={e_str}", flush=True)

        # Clean up and append
        a.calc = None
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        relaxed.append(a)

        # Write incrementally every structure
        ase_write(args.output, relaxed, format="extxyz")

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.0f}s. Converged: {n_conv}/{len(atoms_list) - start_idx}")
    print(f"Output: {args.output} ({len(relaxed)} structures)")


if __name__ == "__main__":
    main()

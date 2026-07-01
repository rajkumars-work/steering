"""Dielectric Properties Calculator

This module calculates dielectric properties of materials using ML surrogate models
and force fields for phonon calculations.

Epsilon / Dielectric Models:
============================

**RECOMMENDED: nequip MLIPs** (April 2026 — see `scripts/validate_nequip_mlips.py`)
- nequip-eps predicts **ε₀** (full static dielectric, includes ionic contribution).
  Validated Pearson r = 0.97 vs DFT eps_0, MAE = 1.10 (n=50).
- nequip-bandgap predicts band gap. Validated r = 0.999, MAE = 0.02 eV.
- Checkpoints at /data/assets/checkpoints/nequip/nequip-eps-*.zip and nequip-bandgap-*.zip
- Use `_from_saved_model(<.zip>, chemical_species_to_atom_type_map=True)` plus
  the num_atoms-injection patch from `validate_nequip_mlips.py:load_calc()`.

Other approaches (kept for reference):

1. **Surrogate (XGBoost, CPU, instant)**:
   - `compute_eps_0(atoms)` → ε₀ static dielectric from composition features
   - Fast but composition-only — cannot distinguish polymorphs

2. **EGIP-epsilon MLIP (DEPRECATED — inaccurate)**:
   - `compute_egip_epsilon(atoms_list)` → ε∞ electronic dielectric only
   - Found to be inaccurate; superseded by nequip-eps

3. **Full Ray + phonons pipeline (~60s/structure)**:
   - `Dielectrics.compute(atoms_list, compute_phonons=True)` via scripts/di_props.py
   - Computes ε∞ (EGIP-epsilon) + ε_ionic (phonon-based) → ε₀
   - Uses Ray actors. Slow; nequip-eps gives equivalent accuracy faster.

Naming convention:
==================
- `mlip_eps_0`: nequip-eps MLIP ε₀ (recommended)
- `mlip_bandgap`: nequip bandgap MLIP (recommended)
- `surrogate_eps_0`: XGBoost composition surrogate (composition-only fallback)
- `egip_epsilon`: EGIP-epsilon MLIP ε∞ (deprecated, kept for legacy)
- `dft_eps_0`: DFT reference value (if available)
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import List, TypedDict, Dict, Any

import numpy as np
from ase.io import read, write
from chem.check_validity import is_realistic

# -----------------------------------------------------------------------------
# Statistics: Training data statistics for validation (loaded from data/training_stats.json)
# -----------------------------------------------------------------------------
def _load_training_stats() -> Dict[str, Dict[str, float]]:
    stats_path = Path(__file__).resolve().parent.parent / "data" / "training_stats.json"
    if not stats_path.exists():
        return {}
    try:
        with stats_path.open("r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return {}
    return {}


stats = _load_training_stats()

import ray
from ase import Atoms
from ase.build import bulk
from atlas.physics.actor import PhysicsActor
from atlas.physics.bandgap import calc_band_gap
from atlas.physics.config import ConfigBandGap
from atlas.test_utils.ray_test_setup import setup_ray_cluster, shutdown_ray_cluster
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element

# Local physics actor for phonons and ionic contributions
from .physics import (
    PhysicsActor as LocalPhysicsActor,
    calculate_phonons,
    calculate_dielectric_properties,
)

LEGACY_PROPS = [
    'bandgap', 'eps_0',
    'min_TO_phonon', 'max_born_charge', 'cell_volume', 'electronegativity_difference'
]
MLIP_PROPS = [
    'mlip_bandgap', 'surrogate_eps_0',
    'mlip_min_TO_phonon', 'mlip_max_born_charge', 'mlip_cell_volume',
    'mlip_electronegativity_difference'
]
PROPS = MLIP_PROPS


class DielectricProps(TypedDict):
    """Computed dielectric and material properties.

    Key dielectric properties (MLIP):
    - surrogate_eps_0: ε₀ (static dielectric constant, surrogate)
    """
    mlip_bandgap: float                      # Band gap in eV
    surrogate_eps_0: float                   # ε₀: Static dielectric constant (surrogate)
    mlip_min_TO_phonon: float                # Minimum TO phonon frequency (THz)
    mlip_max_born_charge: float              # Maximum Born effective charge
    mlip_cell_volume: float                  # Unit cell volume (Å³)
    mlip_electronegativity_difference: float # Max - Min Pauling electronegativity

def _migrate_legacy_mlip_keys(info: Dict[str, Any]) -> None:
    """Populate mlip_* keys from legacy keys if mlip_* keys are missing."""
    mapping = {
        "bandgap": "mlip_bandgap",
        "eps_0": "surrogate_eps_0",
        "min_TO_phonon": "mlip_min_TO_phonon",
        "max_born_charge": "mlip_max_born_charge",
        "cell_volume": "mlip_cell_volume",
        "electronegativity_difference": "mlip_electronegativity_difference",
    }
    for legacy_key, mlip_key in mapping.items():
        if mlip_key not in info and legacy_key in info:
            info[mlip_key] = info[legacy_key]


class Dielectrics:
    def __init__(
        self,
        name_suffix: str | None = None,
        local_gpus: float = 1.0,
        surrogate_gpus: float = 0.5,
        reuse_surrogates: bool = True,
        namespace: str = "dielectrics",
        kill_surrogates: bool = True,
        shutdown_ray: bool = True,
    ) -> None:
        """Initialize Ray cluster and load necessary physics actors.

        Sets up two key models:
        1. egip_bg: Bandgap surrogate model
        2. egip-inf: ML force field for phonons and ionic dielectric calculations
        """
        setup_ray_cluster()

        # Surrogate model for Bandgap prediction
        bg_name = "BandgapSurrogate"
        local_name = "LocalPhysicsActor"
        if name_suffix:
            local_name = f"{local_name}_{name_suffix}"
            if not reuse_surrogates:
                bg_name = f"{bg_name}_{name_suffix}"

        self.bg_surrogate = PhysicsActor.options(
            num_gpus=surrogate_gpus,
            name=bg_name,
            namespace=namespace,
            lifetime="detached",
            get_if_exists=reuse_surrogates,
        ).remote(model_name="egip_bg")

        # ML Force Field for Phonons and Ionic Dielectric Contributions
        # ML Force Field for Phonons and Ionic Dielectric Contributions
        # Used to compute phonons → ionic dielectric tensor → ε_ionic
        # This enables calculation of ε₀ = ε∞ + ε_ionic
        local_options = {
            "num_gpus": local_gpus,
            "name": local_name,
            "namespace": namespace,
            "get_if_exists": not name_suffix,
        }
        # Only keep the shared local actor detached. Per-batch actors should
        # not be detached to avoid leaked GPU reservations after worker exit.
        if not name_suffix:
            local_options["lifetime"] = "detached"

        self.local_actor = LocalPhysicsActor.options(
            **local_options
        ).remote(model_name="egip-inf")

        self._kill_surrogates = kill_surrogates
        self._shutdown_ray = shutdown_ray

        self.bg_config = ConfigBandGap()

    def _create_encoder(self, name_suffix: str):
        import uuid
        uid = str(uuid.uuid4())[:8]
        return PhysicsActor.options(
            num_gpus=0.25,
            name=f"EgipInfActor_{name_suffix}_{uid}",
            namespace="dielectrics",
        ).remote(model_name="egip-inf")

    def compute(
        self,
        atoms_list: List[Atoms],
        relax: bool = False,
        compute_phonons: bool = False,
        return_relaxed_structures: bool = False,
    ) -> List[DielectricProps] | tuple[List[DielectricProps], List[Structure]]:
        """Compute dielectric properties for a list of ASE Atoms.

        This method computes (MLIP-prefixed outputs):
        - surrogate_eps_0: Static dielectric constant via surrogate model (filled elsewhere)
        - Bandgap, phonon frequencies, Born charges, and other properties

        Args:
            atoms_list: List of ASE Atoms objects to process
            relax: If True, run MLIP relaxation before property evaluation.
            compute_phonons: If True, compute phonons + ionic contribution (slow).

        Returns:
            List of DielectricProps dictionaries with computed properties.
            Key fields:
            - surrogate_eps_0: Static dielectric constant (ε₀)
            - mlip_bandgap, mlip_min_TO_phonon, mlip_max_born_charge, etc.
        """
        # Convert ASE Atoms to Pymatgen Structures
        structures: List[Structure] = [
            AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list
        ]

        # 0. Relax structures to ensure stability (optional)
        # Use local actor for relaxation
        relaxed_structures = structures
        if relax:
            print(f"Relaxing {len(structures)} structures...")
            try:
                future_relax = self.local_actor.relax.remote(
                    structures,
                    nrelax=100,
                    fmax=0.01,
                    relax_type="atoms"
                )
                relaxation_results = ray.get(future_relax)
                relaxed_structures = relaxation_results.relaxed_structures
                print("Relaxation complete.")
            except Exception as e:
                print(f"Warning: Relaxation failed: {e}. Proceeding with unrelaxed structures.")
                relaxed_structures = structures

        # Create fresh actors for this batch
        encoder_bg = self._create_encoder("BG")

        # Execute calculations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 1. Bandgap (Atlas)
            future_bg = executor.submit(
                calc_band_gap,
                structs=relaxed_structures,
                config=self.bg_config,
                actor=encoder_bg,
                actor_surr=self.bg_surrogate,
            )
            
            # 2. Phonons and Ionic Contributions (Local Actor)
            phonon_futures = {}
            if compute_phonons:
                for i, s in enumerate(relaxed_structures):
                    # Check for physical validity (lattice vectors)
                    if min(s.lattice.abc) < 1.0:
                        print(f"Warning: Structure {i} has extremely small lattice vectors ({s.lattice.abc}). Skipping phonon calculation to prevent crash.")
                        continue

                    # Use 2x2x2 supercell for phonons
                    phonon_futures[i] = calculate_phonons.remote(
                        self.local_actor,
                        structure=s,
                        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                    )

            bandgaps = future_bg.result()
            # Robustly gather phonon results
            phonons_results = []
            for i in range(len(relaxed_structures)):
                if i in phonon_futures:
                    try:
                        res = ray.get(phonon_futures[i])
                        phonons_results.append(res)
                    except Exception as e:
                        print(f"Error calculating phonons for structure {i}: {e}")
                        phonons_results.append({})
                else:
                    phonons_results.append({})

            # Calculate dielectric properties using phonon data
            # We first compute ionic contribution with eps_inf=1 to recover A
            dielectric_results = []
            if compute_phonons:
                dielectric_futures = []
                for i, (s, p_data) in enumerate(zip(relaxed_structures, phonons_results)):
                    dielectric_futures.append(
                        calculate_dielectric_properties.remote(s, p_data, eps_inf=1.0)
                    )
                # Use strict ray.get here since it's lightweight (pure python/numpy usually)
                # But safer to handle exceptions too
                for f in dielectric_futures:
                    try:
                        dielectric_results.append(ray.get(f))
                    except Exception as e:
                        print(f"Error calculating dielectric props: {e}")
                        dielectric_results.append({})
            else:
                dielectric_results = [{} for _ in relaxed_structures]
        
        # Assemble results

        # Assemble results
        results = []
        for i, (bg, phon, diel, s) in enumerate(
            zip(bandgaps, phonons_results, dielectric_results, relaxed_structures)
        ):

            # 1. Bandgap from surrogate model
            bandgap_val = bg.band_gap

            # Min TO Phonon
            # physics.py returns "lowest_optical_frequency_THz"
            min_to = phon.get("lowest_optical_frequency_THz", 0.0)
            
            # Max Born Charge
            # physics.py returns "born_effective_charges" (list)
            born_charges = diel.get("born_effective_charges", [])
            max_born = max(np.abs(born_charges)) if born_charges else 0.0
            
            # Cell Volume
            cell_vol = s.volume
            
            # Electronegativity Difference
            # Max - Min Pauling electronegativity
            elements = [Element(sp.symbol) for sp in s.composition.elements]
            if elements:
                electronegativities = [e.X for e in elements if e.X is not None]
                if electronegativities:
                    en_diff = max(electronegativities) - min(electronegativities)
                else:
                    en_diff = 0.0
            else:
                en_diff = 0.0
                
            results.append({
                # MLIP-prefixed keys (preferred)
                "mlip_bandgap": bandgap_val,
                # Static dielectric constant (ε₀) from surrogate model (filled elsewhere)
                "surrogate_eps_0": None,
                "mlip_min_TO_phonon": min_to,
                "mlip_max_born_charge": max_born,
                "mlip_cell_volume": cell_vol,
                "mlip_electronegativity_difference": en_diff,

                # Legacy keys (deprecated, kept for backward compatibility)
                "bandgap": bandgap_val,
                "eps_0": None,
                "min_TO_phonon": min_to,
                "max_born_charge": max_born,
                "cell_volume": cell_vol,
                "electronegativity_difference": en_diff,
            })

        if return_relaxed_structures:
            return results, relaxed_structures
        return results

    def close(self) -> None:
        """Terminate actors and shutdown Ray cluster."""
        if self._kill_surrogates:
            for actor in [self.bg_surrogate]:
                if actor:
                    ray.kill(actor)
        if self.local_actor:
            ray.kill(self.local_actor)
        if self._shutdown_ray:
            shutdown_ray_cluster()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -----------------------------------------------------------------------------
# Convenience functions for computing eps_0
# -----------------------------------------------------------------------------

def compute_eps_0(atoms: Atoms, method: str = "surrogate") -> float:
    """Compute static dielectric constant (ε₀) for a single structure.

    This is the main function to use for computing the STATIC dielectric constant,
    which includes both electronic and ionic contributions.

    Args:
        atoms: ASE Atoms object representing the crystal structure
        method: Computation method:
            - "surrogate" (default, RECOMMENDED): Fast ML prediction (~instant, 2-20% error)
            - "phonon": Phonon-based calculation (slow ~60s, requires GPU, 25-260% error)

    Returns:
        eps_0: Surrogate static dielectric constant (ε₀)

    Example:
        >>> from ase.io import read
        >>> from di_props import compute_eps_0
        >>> atoms = read("structure.extxyz")
        >>> eps_0 = compute_eps_0(atoms)  # Uses surrogate (fast & accurate)
        >>> print(f"Static dielectric constant: {eps_0:.2f}")

    Note:
        - "surrogate" method: Fast composition-based ML model, 2-20% typical error
        - "phonon" method: Slow phonon calculation with ε∞ screening, 25-260% error
        - For best accuracy, use stored DFT values: atoms.info['dft_eps_0']
    """
    if method != "surrogate":
        raise ValueError("Only method='surrogate' is supported.")
    from chem.surrogates.eps0 import predict_eps_0_fast
    return predict_eps_0_fast(atoms)


def compute_egip_epsilon(atoms_list, device="cuda", n_ensemble=5):
    """Compute electronic dielectric constant (ε∞) using EGIP-epsilon MLIP ensemble.

    Uses the 5-model EGIP-epsilon ensemble at /data/assets/checkpoints/radsim/epsilon/.
    Returns the mean prediction across ensemble members for each structure.

    Args:
        atoms_list: list of ASE Atoms objects (or single Atoms)
        device: torch device string (default: "cuda")
        n_ensemble: number of ensemble models to use (1-5, default: 5)

    Returns:
        list of float: predicted ε∞ for each structure

    Example:
        >>> from chem.props.dielectric import compute_egip_epsilon
        >>> eps_values = compute_egip_epsilon([atoms1, atoms2])
    """
    import torch
    import numpy as np
    from atlas.physics.model import load_model
    from atlas import MODEL_CONFIG
    from pymatgen.io.ase import AseAtomsAdaptor
    from torch_sim.io import structures_to_state

    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    structs = [AseAtomsAdaptor.get_structure(a) for a in atoms_list]

    CKPT_DIR = "/data/assets/checkpoints/radsim/epsilon"
    all_preds = []
    for i in range(1, n_ensemble + 1):
        name = f"_egip_eps_{i}"
        if name not in MODEL_CONFIG:
            MODEL_CONFIG[name] = {
                "path": f"{CKPT_DIR}/EGIP-epsilon-{i}.pt",
                "dtype": torch.float64,
                "memory_scales_with": "n_atoms",
                "max_memory_scaler": 2500,
            }
        model, _, _, _ = load_model(name)
        state = structures_to_state(structs, device=model._device, dtype=model._dtype)
        output = model(state)
        all_preds.append(output["energy"].cpu().numpy())

    # Mean across ensemble
    ensemble = np.stack(all_preds, axis=0)  # (n_ensemble, n_structures)
    return np.mean(ensemble, axis=0).tolist()


def compute_epsilon(atoms: Atoms) -> float:
    """Compute electronic dielectric constant (ε∞) for a single structure.

    DEPRECATED: Use compute_egip_epsilon() instead, which uses the EGIP-epsilon
    MLIP ensemble at /data/assets/checkpoints/radsim/epsilon/.
    """
    raise RuntimeError(
        "Use compute_egip_epsilon() instead. "
        "EGIP-epsilon models at /data/assets/checkpoints/radsim/epsilon/"
    )


def compute_all_properties(atoms: Atoms) -> DielectricProps:
    """Compute all dielectric properties for a single structure.

    Args:
        atoms: ASE Atoms object representing the crystal structure

    Returns:
        Dictionary containing all computed MLIP properties including:
        - mlip_bandgap: Band gap in eV
        - surrogate_eps_0: ε₀ (static dielectric constant)
        - mlip_min_TO_phonon: Minimum TO phonon frequency
        - mlip_max_born_charge: Maximum Born effective charge
        - mlip_cell_volume: Unit cell volume
        - mlip_electronegativity_difference: Electronegativity range

    Example:
        >>> atoms = read("structure.extxyz")
        >>> props = compute_all_properties(atoms)
        >>> print(f"ε₀ = {props['surrogate_eps_0']:.2f}")
    """
    with Dielectrics() as di:
        results = di.compute([atoms])
    return results[0]


def get_test_atoms() -> List[Atoms]:
    """Generate sample structures for testing."""
    _chem_dir = str(Path(__file__).resolve().parent.parent)
    files = [_chem_dir + '/data_compact/mp-9996-mp.extxyz', _chem_dir + '/data/9996.extxyz']
    atoms = [read(f) for f in files]
    return atoms

def test_dielectrics():
    print("Running Dielectrics Test...")
    structures = get_test_atoms()

    with Dielectrics() as di:
        results = di.compute(structures, relax=False, compute_phonons=False)

    print("\nPredicted results:")
    for i, res in enumerate(results):
        print(f"Structure {i}:")
        for k in PROPS:
            v = res.get(k)
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Calculate dielectric properties for structures.")
    parser.add_argument("file", nargs="?", help="Path to structure file (e.g., .xyz, .extxyz)")
    parser.add_argument("--force", action="store_true", help="Force re-calculation even if properties exist.")
    parser.add_argument(
        "--phonons",
        action="store_true",
        help="Enable phonon calculations for ionic contribution (slow).",
    )
    
    args = parser.parse_args()

    if args.file:
        try:
            structures = read(args.file, index=":")
            if not isinstance(structures, list):
                structures = [structures]
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)
        
        print(f"Checking {len(structures)} structures from {args.file}...")

        to_compute_indices = []
        to_compute_atoms = []

        # Identify which structures need calculation
        for i, atoms in enumerate(structures):
            # Check validity
            valid = is_realistic(atoms, verbose=False)
            if not valid:
                print(f"Structure {i}: INVALID (Skipping)")
                continue
            
            # Check existing properties
            _migrate_legacy_mlip_keys(atoms.info)
            has_props = all(key in atoms.info for key in PROPS)
            
            if not has_props or args.force:
                to_compute_indices.append(i)
                to_compute_atoms.append(atoms)
            else:
                pass # Already computed

        if to_compute_atoms:
            print(f"Calculating properties for {len(to_compute_atoms)} structures...")
            with Dielectrics() as di:
                results = di.compute(
                    to_compute_atoms,
                    relax=False,
                    compute_phonons=args.phonons,
                )
            
            # Update atoms with results
            for idx, res in zip(to_compute_indices, results):
                atoms = structures[idx]
                # Update info dict
                atoms.info.update(res)
            
            # Save back to file
            print(f"Updating file {args.file} with new properties...")
            write(args.file, structures)
        else:
            print("No valid structures require calculation.")
        
        print("\nResults:")
        for i, atoms in enumerate(structures):
            print(f"Structure {i}:")
            # check validity again for reporting
            valid = is_realistic(atoms, verbose=False)
            if not valid:
                print("  Status: INVALID STRUCTURE")
                continue

            # Check if properties exist
            _migrate_legacy_mlip_keys(atoms.info)
            if all(key in atoms.info for key in PROPS):
                for k in PROPS:
                    v = atoms.info[k]
                    in_range_str = ""
                    if k in stats:
                        mean = stats[k]["mean"]
                        std = stats[k]["std"]
                        low = mean - 2 * std
                        high = mean + 2 * std
                        if isinstance(v, (int, float)):
                            if low <= v <= high:
                                in_range_str = " (OK)"
                            else:
                                in_range_str = f" (OUT OF RANGE: {low:.2f} - {high:.2f})"
                    
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}{in_range_str}")
                    else:
                        print(f"  {k}: {v}")

                # Print any DFT values if present
                dft_keys = [k for k in atoms.info.keys() if k.startswith("dft_")]
                for k in sorted(dft_keys):
                    v = atoms.info[k]
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print("  Status: Calculation Failed or Skipped")

    else:
        test_dielectrics()

if __name__ == "__main__":
    main()

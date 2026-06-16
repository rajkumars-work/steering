"""Dielectric Properties Calculator

This module calculates dielectric properties of materials using ML surrogate models
and force fields for phonon calculations.

IMPORTANT - Understanding eps_inf vs eps_0:
=========================================
- surrogate_eps_inf (ε∞): ELECTRONIC dielectric constant (high-frequency limit)
  * Only electronic polarization, no ionic/lattice contribution
  * Estimated from eps_0 surrogate + phonon-derived ionic contribution

- eps_0 (ε₀): STATIC dielectric constant (low-frequency limit)
  * Total dielectric response: electronic + ionic contributions
  * Directly predicted by egip_eps surrogate (note: name historically "epsilon")

Naming convention:
==================
To clearly distinguish MLIP vs DFT values, computed properties in this module
are stored with an `mlip_` prefix (e.g., mlip_eps_0). Any DFT values should be
stored with a `dft_` prefix (e.g., dft_eps_0).
Legacy keys (bandgap/eps_0/...) are still written for backward compatibility
but are considered deprecated.

Quick Usage:
============
    from di_props import compute_eps_0
    from ase.io import read

    atoms = read("structure.extxyz")
    eps_0 = compute_eps_0(atoms)
    print(f"Static dielectric constant: {eps_0}")

See test_dielectrics() for complete example.
"""

import concurrent.futures
from typing import List, TypedDict, Dict, Any
import numpy as np
import argparse
import sys
from ase.io import read, write
from check_validity import is_realistic

# -----------------------------------------------------------------------------
# Statistics: Training data statistics for validation (MLIP keys)
# -----------------------------------------------------------------------------
stats = {
    "mlip_bandgap": {"mean": 2.84, "std": 1.87},
    "surrogate_eps_inf": {"mean": 16.25, "std": 14.14},  # ε∞: Electronic dielectric constant
    "mlip_eps_0": {"mean": 600.08, "std": 3708.09},  # ε₀: Static (electronic + ionic)
    "mlip_min_TO_phonon": {"mean": 2.02, "std": 2.0},
    "mlip_max_born_charge": {"mean": 3.51, "std": 1.43},
    "mlip_cell_volume": {"mean": 274.89, "std": 190.03},
    "mlip_electronegativity_difference": {"mean": 1.98, "std": 0.73},
}

import ray
from ase import Atoms
from ase.build import bulk
from atlas.physics.actor import PhysicsActor
from atlas.physics.bandgap import calc_band_gap
from atlas.physics.config import ConfigBandGap, ConfigEpsilon
from atlas.physics.epsilon import calc_epsilon
from atlas.test_utils.ray_test_setup import setup_ray_cluster, shutdown_ray_cluster
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element

# Local physics actor for phonons and ionic contributions
from physics import (
    PhysicsActor as LocalPhysicsActor,
    calculate_phonons,
    calculate_dielectric_properties,
)

LEGACY_PROPS = [
    'bandgap', 'eps_0',
    'min_TO_phonon', 'max_born_charge', 'cell_volume', 'electronegativity_difference'
]
MLIP_PROPS = [
    'mlip_bandgap', 'surrogate_eps_inf', 'mlip_eps_0', 'mlip_eps_ionic_est',
    'mlip_min_TO_phonon', 'mlip_max_born_charge', 'mlip_cell_volume',
    'mlip_electronegativity_difference'
]
PROPS = MLIP_PROPS


class DielectricProps(TypedDict):
    """Computed dielectric and material properties.

    Key dielectric properties (MLIP):
    - surrogate_eps_inf: ε∞ (electronic dielectric constant, derived)
    - mlip_eps_0: ε₀ (static dielectric constant, surrogate)
    - mlip_eps_ionic_est: Ionic contribution estimate
    """
    mlip_bandgap: float                      # Band gap in eV
    surrogate_eps_inf: float                 # ε∞: Electronic dielectric constant (derived)
    mlip_eps_0: float                        # ε₀: Static dielectric = electronic + ionic
    mlip_eps_ionic_est: float                # Ionic contribution estimate
    mlip_min_TO_phonon: float                # Minimum TO phonon frequency (THz)
    mlip_max_born_charge: float              # Maximum Born effective charge
    mlip_cell_volume: float                  # Unit cell volume (Å³)
    mlip_electronegativity_difference: float # Max - Min Pauling electronegativity

def _migrate_legacy_mlip_keys(info: Dict[str, Any]) -> None:
    """Populate mlip_* keys from legacy keys if mlip_* keys are missing."""
    mapping = {
        "bandgap": "mlip_bandgap",
        "eps_inf": "surrogate_eps_inf",
        "eps_0": "mlip_eps_0",
        "eps_ionic_est": "mlip_eps_ionic_est",
        "min_TO_phonon": "mlip_min_TO_phonon",
        "max_born_charge": "mlip_max_born_charge",
        "cell_volume": "mlip_cell_volume",
        "electronegativity_difference": "mlip_electronegativity_difference",
    }
    for legacy_key, mlip_key in mapping.items():
        if mlip_key not in info and legacy_key in info:
            info[mlip_key] = info[legacy_key]


class Dielectrics:
    def __init__(self) -> None:
        """Initialize Ray cluster and load necessary physics actors.

        Sets up three key models:
        1. egip_bg: Bandgap surrogate model
        2. egip_eps: Static dielectric constant (ε₀) surrogate model
        3. egip-inf: ML force field for phonons and ionic dielectric calculations
        """
        setup_ray_cluster()

        # Surrogate model for Bandgap prediction
        self.bg_surrogate = PhysicsActor.options(
            num_gpus=0.5,
            name="BandgapSurrogate",
            namespace="dielectrics",
            lifetime="detached",
            get_if_exists=True,
        ).remote(model_name="egip_bg")

        # Surrogate model for Electronic Dielectric Constant (ε∞)
        # This model predicts ONLY the electronic contribution (high-frequency limit)
        self.eps_surrogate = PhysicsActor.options(
            num_gpus=0.5,
            name="EpsilonSurrogate",
            namespace="dielectrics",
            lifetime="detached",
            get_if_exists=True,
        ).remote(model_name="egip_eps")

        # ML Force Field for Phonons and Ionic Dielectric Contributions
        # Used to compute phonons → ionic dielectric tensor → ε_ionic
        # This enables calculation of ε₀ = ε∞ + ε_ionic
        self.local_actor = LocalPhysicsActor.options(
            num_gpus=1,
            name="LocalPhysicsActor",
            namespace="dielectrics",
            lifetime="detached",
            get_if_exists=True,
        ).remote(model_name="egip-inf")

        self.bg_config = ConfigBandGap()
        self.eps_config = ConfigEpsilon()

    def _create_encoder(self, name_suffix: str):
        import uuid
        uid = str(uuid.uuid4())[:8]
        return PhysicsActor.options(
            num_gpus=0.25,
            name=f"EgipInfActor_{name_suffix}_{uid}",
            namespace="dielectrics",
        ).remote(model_name="egip-inf")

    def compute(self, atoms_list: List[Atoms]) -> List[DielectricProps]:
        """Compute dielectric properties for a list of ASE Atoms.

        This method computes (MLIP-prefixed outputs):
        - mlip_eps_0: Static dielectric constant via egip_eps surrogate model
        - mlip_eps_ionic_est: Ionic contribution via phonon calculations with egip-inf
        - surrogate_eps_inf: Estimated by solving ε₀ = ε∞ + A/ε∞
        - Bandgap, phonon frequencies, Born charges, and other properties

        Args:
            atoms_list: List of ASE Atoms objects to process

        Returns:
            List of DielectricProps dictionaries with computed properties.
            Key fields:
            - surrogate_eps_inf: Electronic dielectric constant (ε∞)
            - mlip_eps_0: Static dielectric constant (ε₀ = ε∞ + ε_ionic)
            - mlip_bandgap, mlip_min_TO_phonon, mlip_max_born_charge, etc.
        """
        # Convert ASE Atoms to Pymatgen Structures
        structures: List[Structure] = [
            AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list
        ]

        # 0. Relax structures to ensure stability
        # Use local actor for relaxation
        print(f"Relaxing {len(structures)} structures...")
        try:
            future_relax = self.local_actor.relax.remote(
                structures,
                nrelax=100,
                fmax=0.01,
                relax_type="atoms"
            )
            relaxation_results = ray.get(future_relax)
            structures = relaxation_results.relaxed_structures
            print("Relaxation complete.")
        except Exception as e:
            print(f"Warning: Relaxation failed: {e}. Proceeding with unrelaxed structures.")

        # Create fresh actors for this batch
        encoder_bg = self._create_encoder("BG")

        encoder_eps = self._create_encoder("EPS")

        # Execute calculations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 1. Bandgap (Atlas)
            future_bg = executor.submit(
                calc_band_gap,
                structs=structures,
                config=self.bg_config,
                actor=encoder_bg,
                actor_surr=self.bg_surrogate,
            )
            
            # 2. Epsilon (Atlas - Actually predicts eps_0 / static dielectric)
            future_eps = executor.submit(
                calc_epsilon,
                structs=structures,
                config=self.eps_config,
                actor=encoder_eps,
                actor_surr=self.eps_surrogate,
            )
            
            # 3. Phonons and Ionic Contributions (Local Actor)
            phonon_futures = {}
            for i, s in enumerate(structures):
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
            epsilons = future_eps.result()

            # Robustly gather phonon results
            phonons_results = []
            for i in range(len(structures)):
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
            dielectric_futures = []
            for i, (s, p_data) in enumerate(zip(structures, phonons_results)):
                dielectric_futures.append(
                    calculate_dielectric_properties.remote(s, p_data, eps_inf=1.0)
                )
            # Use strict ray.get here since it's lightweight (pure python/numpy usually)
            # But safer to handle exceptions too
            dielectric_results = []
            for f in dielectric_futures:
                try:
                    dielectric_results.append(ray.get(f))
                except Exception as e:
                     print(f"Error calculating dielectric props: {e}")
                     dielectric_results.append({})
        
        # Assemble results

        # Assemble results
        results = []
        for i, (bg, eps, phon, diel, s) in enumerate(zip(bandgaps, epsilons, phonons_results, dielectric_results, structures)):

            # 1. Bandgap from surrogate model
            bandgap_val = bg.band_gap

            # 2. Static dielectric constant (ε₀) from egip_eps surrogate model
            # NOTE: despite the name, egip_eps is trained to predict eps_0.
            eps_0_pred = eps.epsilon

            # 3. Ionic dielectric contribution from phonon calculations with eps_inf=1
            #    This gives A in: eps_ionic = A / eps_inf
            e_ionic_tensor_1 = diel.get("ionic_dielectric_constant_nominal")
            if e_ionic_tensor_1 is not None:
                e_ionic_a = float(np.mean(np.diag(e_ionic_tensor_1)))
            else:
                e_ionic_a = 0.0
            # Debug: print ionic diagnostics for mass-weighting check
            diag = diel.get("ionic_diagnostics", {})
            print(
                f"[ionic diag] A={e_ionic_a:.6f} "
                f"eigvec_shape_before={diag.get('eigvecs_shape_before')} "
                f"eigvec_shape_after={diag.get('eigvecs_shape_after')} "
                f"eigvec_reshape={diag.get('eigvecs_reshape_applied')} "
                f"mode_vec_norms={diag.get('mode_vec_norms')} "
                f"mass_weighted_norms={diag.get('mass_weighted_mode_vec_norms')}"
            )

            # 4. Estimate eps_inf by solving: eps_0 = eps_inf + A / eps_inf
            eps_inf_val = None
            disc = None
            if e_ionic_a >= 0:
                disc = (eps_0_pred ** 2) - 4.0 * e_ionic_a
                if disc >= 0:
                    sqrt_disc = float(np.sqrt(disc))
                    root_small = 0.5 * (eps_0_pred - sqrt_disc)
                    root_large = 0.5 * (eps_0_pred + sqrt_disc)
                    # If ionic contribution is tiny, eps_inf ~= eps_0 (use large root)
                    if e_ionic_a <= 1e-6:
                        eps_inf_val = root_large
                    else:
                        # Prefer a positive root. Use the smaller root only if it's positive.
                        if root_small > 0:
                            eps_inf_val = root_small
                        elif root_large > 0:
                            eps_inf_val = root_large
            if eps_inf_val is None:
                print(
                    f"[eps_inf solve failed] eps_0_pred={eps_0_pred:.6f} "
                    f"A={e_ionic_a:.6f} disc={disc} "
                    f"min_freq_used_THz={diag.get('min_abs_freq_used_THz')} "
                    f"max_born_abs={diag.get('max_born_abs')} "
                    f"max_sm_norm={diag.get('max_mode_polarity_norm')}"
                )

            # 5. Compute ionic contribution using estimated eps_inf
            if eps_inf_val is not None and eps_inf_val > 0:
                e_ionic_val = e_ionic_a / eps_inf_val
            else:
                e_ionic_val = None

            # Sanity check
            if eps_0_pred < 0:
                print(f"WARNING: Negative eps_0 predicted by model ({eps_0_pred:.4f}).")

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
                # Electronic dielectric constant (ε∞) estimated from eps_0 and phonons
                "surrogate_eps_inf": eps_inf_val,
                # Static dielectric constant (ε₀) from surrogate model
                "mlip_eps_0": eps_0_pred,
                # Optional: ionic contribution estimated from A/eps_inf
                "mlip_eps_ionic_est": e_ionic_val,
                "mlip_min_TO_phonon": min_to,
                "mlip_max_born_charge": max_born,
                "mlip_cell_volume": cell_vol,
                "mlip_electronegativity_difference": en_diff,

                # Legacy keys (deprecated, kept for backward compatibility)
                "bandgap": bandgap_val,
                "eps_0": eps_0_pred,
                "eps_ionic_est": e_ionic_val,
                "min_TO_phonon": min_to,
                "max_born_charge": max_born,
                "cell_volume": cell_vol,
                "electronegativity_difference": en_diff,
            })

        return results

    def close(self) -> None:
        """Terminate actors and shutdown Ray cluster."""
        for actor in [self.bg_surrogate, self.eps_surrogate, self.local_actor]:
            if actor:
                ray.kill(actor)
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
        eps_0: MLIP static dielectric constant (ε₀)

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
    if method == "surrogate":
        # Use fast ML surrogate model
        from eps_0_surrogate import predict_eps_0_fast
        return predict_eps_0_fast(atoms)
    elif method == "phonon":
        # Use phonon-based calculation (slow, less accurate)
        with Dielectrics() as di:
            results = di.compute([atoms])
        return results[0]['mlip_eps_0']
    else:
        raise ValueError(f"Unknown method: {method}. Use 'surrogate' or 'phonon'")


def compute_epsilon(atoms: Atoms) -> float:
    """Compute electronic dielectric constant (ε∞) for a single structure.

    This computes the HIGH-FREQUENCY electronic dielectric constant,
    which does NOT include ionic contributions.

    Args:
        atoms: ASE Atoms object representing the crystal structure

    Returns:
        epsilon: Surrogate electronic dielectric constant (ε∞)

    Example:
        >>> atoms = read("structure.extxyz")
        >>> eps_inf = compute_epsilon(atoms)
        >>> print(f"Electronic dielectric constant: {eps_inf:.2f}")
    """
    with Dielectrics() as di:
        results = di.compute([atoms])
    return results[0]['surrogate_eps_inf']


def compute_all_properties(atoms: Atoms) -> DielectricProps:
    """Compute all dielectric properties for a single structure.

    Args:
        atoms: ASE Atoms object representing the crystal structure

    Returns:
        Dictionary containing all computed MLIP properties including:
        - mlip_bandgap: Band gap in eV
        - surrogate_eps_inf: ε∞ (electronic dielectric constant)
        - mlip_eps_0: ε₀ (static dielectric constant)
        - mlip_min_TO_phonon: Minimum TO phonon frequency
        - mlip_max_born_charge: Maximum Born effective charge
        - mlip_cell_volume: Unit cell volume
        - mlip_electronegativity_difference: Electronegativity range

    Example:
        >>> atoms = read("structure.extxyz")
        >>> props = compute_all_properties(atoms)
        >>> print(f"ε∞ = {props['surrogate_eps_inf']:.2f}")
        >>> print(f"ε₀ = {props['mlip_eps_0']:.2f}")
        >>> print(f"Ionic contribution = {props['mlip_eps_0'] - props['surrogate_eps_inf']:.2f}")
    """
    with Dielectrics() as di:
        results = di.compute([atoms])
    return results[0]


def get_test_atoms() -> List[Atoms]:
    """Generate sample structures for testing."""
    files = ['data_compact/mp-9996-mp.extxyz', 'data/9996.extxyz']
    atoms = [read(f) for f in files]
    return atoms

def test_dielectrics():
    print("Running Dielectrics Test...")
    structures = get_test_atoms()

    with Dielectrics() as di:
        results = di.compute(structures)

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
                results = di.compute(to_compute_atoms)
            
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

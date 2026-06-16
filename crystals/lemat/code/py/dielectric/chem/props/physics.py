"""Physics calculations: PhysicsActor, elastic property evaluation, affine shear."""

import threading
import ray
import numpy as np
import typing
import torch
from scipy.stats import skew
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pydantic import BaseModel, ConfigDict

from ase import Atoms, units
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Any, Union
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Internal Atlas imports
from atlas.physics.config import (
    DataStatic,
    ConfigMechanical,
)
# Use the library PhysicsActor
from atlas.physics.actor import PhysicsActor, RelaxationResult
from atlas.physics.mechanical import calc_mechanical
from atlas.workflows.hea.structure_generation import (
    generate_optimized_sqs_structures_from_compositions,
)

# Constants
kB = 8.617333262e-5  # eV/K - Boltzmann constant


@ray.remote
def calculate_phonons(
    actor, structure: Structure, supercell_matrix: list[list[int]] = None
) -> dict[str, Any]:
    """Calculate phonon properties using Phonopy and MLIP forces (via actor)."""
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        return {"error": "Phonopy not installed"}

    if supercell_matrix is None:
        supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # 1. Setup Phonopy
    atoms = AseAtomsAdaptor.get_atoms(structure)
    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        positions=atoms.get_positions(),
    )
    phonon = Phonopy(ph_atoms, supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.supercells_with_displacements

    # 2. Calculate Forces for Supercells
    # Convert Phonopy atoms to PMG structures for batch processing
    pmg_supercells = []
    for sc in supercells:
        # Phonopy atoms -> ASE -> PMG
        sc_ase = Atoms(
            symbols=sc.symbols,
            positions=sc.positions,
            cell=sc.cell,
            pbc=True,
        )
        pmg_supercells.append(AseAtomsAdaptor.get_structure(sc_ase))

    # Batch calculate forces via actor
    # actor.static returns list[DataStatic]
    # We batch the request to avoid OOM on large supercells
    batch_size = 1
    results = []
    for i in range(0, len(pmg_supercells), batch_size):
        batch = pmg_supercells[i : i + batch_size]
        batch_res = ray.get(actor.static.remote(batch, compute_forces=True, return_forces=True))
        results.extend(batch_res)
    
    forces_set = []
    for res in results:
        if res.forces is None:
            forces_set.append(np.zeros((len(sc.symbols), 3)))
        else:
            forces_set.append(res.forces)

    # 3. Produce Force Constants
    phonon.produce_force_constants(forces=forces_set)

    # 4. Calculate Properties at Gamma point
    # Frequencies and Eigenvectors
    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q=[0, 0, 0])
    
    # Filter acoustic modes (should be close to 0)
    optical_indices = [i for i, f in enumerate(frequencies) if abs(f) > 0.1]
    optical_freqs = frequencies[optical_indices]
    
    lowest_optical = min(np.abs(optical_freqs)) if len(optical_freqs) > 0 else 0.0
    avg_optical = np.mean(np.abs(optical_freqs)) if len(optical_freqs) > 0 else 0.0
    
    # Check for imaginary modes (instability)
    imaginary_modes = [f for f in frequencies if f < -0.01]
    has_instability = len(imaginary_modes) > 0

    # Thermal properties & DOS moments
    phonon.run_mesh([10, 10, 10])
    phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)
    tp_dict = phonon.get_thermal_properties_dict()
    
    # DOS moments from mesh frequencies
    mesh_freqs = phonon.get_mesh_dict()['frequencies'].flatten()
    # Filter out negligible/imaginary for stats
    real_mesh_freqs = mesh_freqs[mesh_freqs > 0.01]
    
    if len(real_mesh_freqs) > 0:
        dos_mean = np.mean(real_mesh_freqs)
        dos_var = np.var(real_mesh_freqs)
        dos_skew = float(skew(real_mesh_freqs))
        phonon_bandwidth = np.max(real_mesh_freqs) - np.min(real_mesh_freqs)
    else:
        dos_mean, dos_var, dos_skew, phonon_bandwidth = 0.0, 0.0, 0.0, 0.0

    # Acoustic slope near Gamma (approximate group velocity)
    # Check q-point [0.01, 0, 0]
    try:
        freqs_q_small = phonon.get_frequencies(q=[0.01, 0, 0])
        # Take average of 3 acoustic modes (lowest non-imaginary)
        # Assuming first 3 are acoustic.
        acoustic_freqs = np.sort(np.abs(freqs_q_small))[:3]
        # slope = dw/dq approx w / 0.01 (units THz * A approx)
        acoustic_slope = np.mean(acoustic_freqs) / 0.01
    except Exception:
        acoustic_slope = 0.0
        
    # Vibrational entropy proxy (S at 300K)
    # tp_dict keys: 'temperatures', 'free_energy', 'entropy', 'heat_capacity'
    temps = tp_dict['temperatures']
    entropies = tp_dict['entropy']
    # Find index for ~300K
    idx_300 = (np.abs(temps - 300)).argmin()
    vib_entropy = entropies[idx_300]

    return {
        "frequencies_gamma": frequencies.tolist(),
        "eigenvectors_gamma": eigenvectors,
        "masses": phonon.masses.tolist(),
        "lowest_optical_frequency_THz": lowest_optical,
        "avg_optical_frequency_THz": avg_optical,
        "has_instability": has_instability,
        "imaginary_modes": imaginary_modes,
        "force_constants": phonon.force_constants.tolist(),
        "thermal_properties": tp_dict,
        "dos_moments": {
            "mean": dos_mean,
            "variance": dos_var,
            "skew": dos_skew
        },
        "phonon_bandwidth": phonon_bandwidth,
        "acoustic_slope": acoustic_slope,
        "vibrational_entropy_300K": vib_entropy
    }


@ray.remote
def run_thermal_perturbations(
    actor, structure: Structure, temp: float = 300.0, steps: int = 1000
) -> dict[str, Any]:
    """Run short MD to sample thermal perturbations and anharmonicity."""
    try:
        from ase.calculators.calculator import Calculator, all_changes
        
        # Internal wrapper to use actor via remote call
        class InternalActorCalculator(Calculator):
            implemented_properties = ['energy', 'forces']
            def __init__(self, actor_handle):
                super().__init__()
                self.actor = actor_handle
            
            def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                # Convert to Structure
                s = AseAtomsAdaptor.get_structure(atoms)
                # Use actor via ray.get
                res = ray.get(self.actor.static.remote([s], compute_forces=True, return_forces=True))[0]
                self.results['energy'] = res.energy
                self.results['forces'] = res.forces

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = InternalActorCalculator(actor)
        
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=0.02)
        
        positions_traj = []
        energies_traj = []
        
        def log_step():
            positions_traj.append(atoms.get_positions())
            energies_traj.append(atoms.get_potential_energy())
            
        dyn.attach(log_step, interval=10)
        dyn.run(steps)
        
        # Analyze trajectory
        pos_array = np.array(positions_traj)
        # Variance of positions (thermal ellipsoid proxy)
        # Center the trajectory first to remove drift
        pos_centered = pos_array - pos_array.mean(axis=0)
        variance = np.var(pos_centered, axis=0).sum() # Sum of variances in x,y,z
        
        return {
            "position_variance": variance,
            "mean_energy": np.mean(energies_traj),
            "energy_std": np.std(energies_traj)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "position_variance": None,
            "mean_energy": None,
            "energy_std": None
        }


@ray.remote
def calculate_anharmonicity(actor, structure: Structure, delta: float = 0.05) -> dict[str, float]:
    """Measure deviation from Hooke's law via random displacement."""
    # 1. Generate random displacement vector
    n_atoms = len(structure)
    disp = np.random.uniform(-1, 1, (n_atoms, 3))
    # Normalize to delta length
    norms = np.linalg.norm(disp, axis=1, keepdims=True)
    disp = (disp / norms) * delta
    
    # 2. Create structures (+disp and -disp)
    coords = structure.cart_coords
    
    s_plus = Structure(
        structure.lattice,
        structure.species,
        coords + disp,
        coords_are_cartesian=True,
        site_properties=structure.site_properties
    )
    
    s_minus = Structure(
        structure.lattice,
        structure.species,
        coords - disp,
        coords_are_cartesian=True,
        site_properties=structure.site_properties
    )
    
    # 3. Calculate forces
    results = ray.get(actor.static.remote([s_plus, s_minus], compute_forces=True, return_forces=True))
    f_plus = np.array(results[0].forces)
    f_minus = np.array(results[1].forces)
    
    # 4. Measure deviation
    # Harmonic: f_plus = -f_minus => f_plus + f_minus = 0
    residual = f_plus + f_minus
    
    mean_res_norm = np.mean(np.linalg.norm(residual, axis=1))
    mean_force_norm = np.mean(np.linalg.norm(f_plus, axis=1))
    
    score = mean_res_norm / mean_force_norm if mean_force_norm > 1e-6 else 0.0
    
    return {
        "harmonicity_score": 1.0 - score, # 1 is perfect
        "anharmonicity_measure": score    # 0 is perfect
    }


@ray.remote
def calculate_internal_strain_proxy(actor, structure: Structure, strain: float = 0.01) -> float:
    """Measure magnitude of internal relaxations under strain."""
    # 1. Apply affine strain
    s_strained = structure.copy()
    s_strained.apply_strain(strain)
    
    # 2. Relax internal coordinates (fixed cell)
    # Note: relax() usually relaxes cell too unless relax_type="atoms"
    # actor.relax returns RelaxationResult
    res = ray.get(actor.relax.remote([s_strained], nrelax=50, fmax=0.01, relax_type="atoms"))
    s_relaxed = res.relaxed_structures[0]
    
    # 3. Compare positions
    # Be careful with PBC. Minimum image distance.
    # Fractional coords difference
    diff_frac = s_relaxed.frac_coords - s_strained.frac_coords
    diff_frac = diff_frac - np.round(diff_frac) # PBC
    diff_cart = s_strained.lattice.get_cartesian_coords(diff_frac)
    
    rms_displacement = np.sqrt(np.mean(np.sum(diff_cart**2, axis=1)))
    return rms_displacement


@ray.remote
def calculate_gruneisen_proxy(actor, structure: Structure) -> dict[str, float]:
    """Estimate Gruneisen parameter via mini-compression."""
    # 1. Gamma phonons at V0
    
    def get_gamma_freqs(struct):
        # Manually do finite diff for Gamma point only (supercell=[1,1,1])
        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
            atoms = AseAtomsAdaptor.get_atoms(struct)
            ph_atoms = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.get_cell(), positions=atoms.get_positions())
            phonon = Phonopy(ph_atoms, supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]])
            phonon.generate_displacements(distance=0.01)
            scs = phonon.supercells_with_displacements
            pmg_scs = [AseAtomsAdaptor.get_structure(Atoms(symbols=sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)) for sc in scs]
            
            results = ray.get(actor.static.remote(pmg_scs, compute_forces=True, return_forces=True))
            forces = [r.forces for r in results]
            phonon.produce_force_constants(forces=forces)
            freqs = phonon.get_frequencies(q=[0,0,0])
            return freqs
        except Exception:
            return np.array([])

    freqs_0 = get_gamma_freqs(structure)
    
    # 2. Compress structure
    s_comp = structure.copy()
    s_comp.scale_lattice(s_comp.volume * 0.99)
    freqs_1 = get_gamma_freqs(s_comp)
    
    if len(freqs_0) == 0 or len(freqs_1) == 0:
        return {"avg_gruneisen": None}
        
    # Filter optical
    mask = np.abs(freqs_0) > 0.1
    f0 = np.abs(freqs_0[mask])
    f1 = np.abs(freqs_1[mask])
    
    if len(f0) == 0:
        return {"avg_gruneisen": 0.0}
        
    # dV/V = -0.01
    # gamma = - (V/w) (dw/dV) = - (1/w) (dw / (dV/V))
    # dw = f1 - f0
    # dV/V = -0.01
    # gamma = - (1/f0) * (f1 - f0) / (-0.01) = 100 * (f1 - f0) / f0
    
    gammas = 100 * (f1 - f0) / f0
    avg_gamma = np.mean(gammas)
    
    return {"avg_gruneisen": avg_gamma}


@ray.remote
def calculate_dielectric_properties(structure: Structure, phonon_data: dict[str, Any] = None, eps_inf: float = None) -> dict[str, Any]:
    """Estimate dielectric properties using nominal charges and phonons.

    Args:
        structure: Pymatgen Structure object
        phonon_data: Dictionary with phonon frequencies and eigenvectors from calculate_phonons
        eps_inf: Electronic dielectric constant (ε∞) for proper screening.
                 If None, will estimate from refractive index ~4 or use 1.0 (no screening).
                 IMPORTANT: Should be provided for accurate ionic dielectric calculation.

    Returns:
        Dictionary with born_effective_charges, ionic_dielectric_constant_nominal, etc.
    """

    # 1. Estimate Oxidation States (Born Charges Proxy)
    # Try multiple methods in order of reliability
    born_charges = None
    born_method = None

    # Method 1: BVAnalyzer with more permissive settings
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        # Use more permissive distance tolerance for relaxed structures
        bva = BVAnalyzer(max_radius=4.0, symm_tol=0.2)  # Increased from defaults
        oxi_structure = bva.get_oxi_state_decorated_structure(structure)
        born_charges = [site.specie.oxi_state for site in oxi_structure]
        born_method = "Bond Valence (Permissive)"
    except Exception as e:
        pass

    # Method 2: Standard BVAnalyzer
    if born_charges is None:
        try:
            from pymatgen.analysis.bond_valence import BVAnalyzer
            bva = BVAnalyzer()
            oxi_structure = bva.get_oxi_state_decorated_structure(structure)
            born_charges = [site.specie.oxi_state for site in oxi_structure]
            born_method = "Bond Valence (Standard)"
        except Exception:
            pass

    # Method 3: Composition-based guess
    if born_charges is None:
        try:
            from pymatgen.core.composition import Composition
            comp = structure.composition
            oxi_guesses = comp.oxi_state_guesses()
            if oxi_guesses:
                oxi_map = oxi_guesses[0]
                born_charges = [oxi_map.get(str(site.specie.symbol), 0.0) for site in structure]
                born_method = "Composition Guess"
        except Exception:
            pass

    # Method 4: Simple ionic charge assignment based on common oxidation states
    if born_charges is None:
        # Define common oxidation states for elements
        common_oxidation_states = {
            'O': -2, 'F': -1, 'Cl': -1, 'S': -2, 'N': -3,
            'Li': +1, 'Na': +1, 'K': +1, 'Rb': +1, 'Cs': +1,
            'Be': +2, 'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
            'Al': +3, 'Ga': +3, 'In': +3,
            'Si': +4, 'Ge': +4, 'Sn': +4, 'Pb': +4,
            'P': +5, 'As': +5, 'Sb': +5,
            'Fe': +3, 'Co': +2, 'Ni': +2, 'Cu': +2, 'Zn': +2
        }

        try:
            born_charges = []
            for site in structure:
                symbol = site.specie.symbol
                charge = common_oxidation_states.get(symbol, 0.0)
                born_charges.append(charge)
            born_method = "Common Oxidation States"
        except Exception:
            pass

    # Method 5: Last resort - all neutral
    if born_charges is None or not born_charges:
        born_charges = [0.0] * len(structure)
        born_method = "Failed (Neutral)"

    # 2. Reconstruct Epsilon Ionic (if phonon data available)
    epsilon_ionic = None
    ionic_diagnostics: dict[str, Any] = {}
    mode_polarities = []
    mode_contributions = []
    if phonon_data and "frequencies_gamma" in phonon_data:
        try:
            freqs = np.array(phonon_data["frequencies_gamma"]) # THz
            eigvecs = np.array(phonon_data["eigenvectors_gamma"])
            masses = np.array(phonon_data["masses"]) # amu

            # Physical constants
            e = 1.60217663e-19  # Elementary charge (C)
            eps0 = 8.85418781e-12  # Vacuum permittivity (F/m)
            amu = 1.66053907e-27  # Atomic mass unit (kg)
            THz = 1e12  # THz to Hz conversion
            Ang = 1e-10  # Angstrom to meter conversion

            vol_m3 = structure.volume * (Ang**3)

            # Determine eps_inf for screening
            # CRITICAL: Ionic dielectric must be screened by electronic dielectric
            if eps_inf is None:
                # Fallback: estimate from typical refractive index n~2 → ε_∞ ~ 4
                eps_inf_value = 4.0
                print(f"Warning: eps_inf not provided, using default {eps_inf_value} for screening")
            else:
                eps_inf_value = eps_inf
            
            # Handle eigenvectors
            eigvecs = np.array(eigvecs) 
            if np.iscomplexobj(eigvecs):
                eigvecs = eigvecs.real # Gamma point should be real
            
            # Reshape if necessary to (modes, atoms, 3)
            eigvecs_shape_before = tuple(eigvecs.shape)
            reshape_applied = False
            if eigvecs.ndim == 2: # (3N, 3N)
                eigvecs = eigvecs.T # Make first dim the mode index
                eigvecs = eigvecs.reshape(len(freqs), len(structure), 3)
                reshape_applied = True
            eigvecs_shape_after = tuple(eigvecs.shape)
            # Basic sanity check on shape
            if eigvecs.ndim != 3 or eigvecs.shape[0] != len(freqs) or eigvecs.shape[1] != len(structure) or eigvecs.shape[2] != 3:
                raise ValueError(
                    f"Unexpected eigenvector shape. before={eigvecs_shape_before} after={eigvecs_shape_after} "
                    f"n_modes={len(freqs)} n_atoms={len(structure)}"
                )
            # Mode vector norms (diagnostic)
            mode_vec_norms = np.linalg.norm(eigvecs.reshape(len(freqs), -1), axis=1)
            mode_vec_norms_stats = {
                "min": float(np.min(mode_vec_norms)),
                "max": float(np.max(mode_vec_norms)),
                "mean": float(np.mean(mode_vec_norms)),
            }
            # Mass-weighted norm check: sqrt(sum_i m_i * |e_i|^2)
            try:
                masses_amu = np.array(masses, dtype=float)
                e_sq = np.sum(eigvecs**2, axis=2)  # (modes, atoms)
                mw_norms = np.sqrt(np.sum(e_sq * masses_amu[None, :], axis=1))
                mass_weighted_norms_stats = {
                    "min": float(np.min(mw_norms)),
                    "max": float(np.max(mw_norms)),
                    "mean": float(np.mean(mw_norms)),
                }
            except Exception:
                mass_weighted_norms_stats = None

            eps_ionic = np.zeros((3, 3))
            used_modes = 0
            min_abs_freq_all = float(np.min(np.abs(freqs))) if len(freqs) else None
            min_abs_freq_used = None
            max_abs_freq_used = None
            min_omega = None
            max_prefactor = None
            born_abs = [abs(z) for z in born_charges] if born_charges else []
            max_born_abs = max(born_abs) if born_abs else None
            mean_born_abs = float(np.mean(born_abs)) if born_abs else None
            max_sm_norm = 0.0
            mode_polarities = []
            mode_contributions = []
            
            # Sort modes by frequency magnitude to identify acoustic modes
            mode_indices = np.argsort(np.abs(freqs))
            
            for i, m in enumerate(mode_indices):
                # Skip the first 3 modes (acoustic)
                if i < 3:
                    mode_polarities.append(np.zeros(3).tolist())
                    mode_contributions.append(0.0)
                    continue

                freq = freqs[m]
                if abs(freq) < 0.5: # Increased threshold to avoid soft artifact modes 
                    mode_polarities.append(np.zeros(3).tolist())
                    mode_contributions.append(0.0)
                    continue 
                
                omega = freq * THz

                
                mode_vec = eigvecs[m] # (N, 3)
                
                # Compute mode polarity vector S_m
                S_m = np.zeros(3)
                # Eigenvectors appear unit-norm (not mass-weighted).
                # Use sqrt(amu) (dimensionless) instead of kg to avoid unit blow-up.
                # Try interpreting eigenvectors as fractional displacements and convert to cartesian
                # This is a diagnostic path: u_cart = u_frac @ lattice (Angstrom)
                lat = structure.lattice.matrix  # 3x3 (Angstrom)
                for k in range(len(structure)):
                    z = born_charges[k]
                    mass_amu = masses[k]
                    u_frac = mode_vec[k]  # (3,)
                    u_cart = u_frac @ lat
                    S_m += z * u_cart / np.sqrt(mass_amu)
                sm_norm = float(np.linalg.norm(S_m))
                if sm_norm > max_sm_norm:
                    max_sm_norm = sm_norm

                # CORRECTED FORMULA: Ionic dielectric with high-frequency screening
                # ε_ionic = Σ_m (e² / (ε₀ ε_∞ V ω_m²)) × |Z* · u_m|²
                #
                # Previous (WRONG): prefactor = e² / (ε₀ V ω²)
                # Corrected (RIGHT): prefactor = e² / (ε₀ ε_∞ V ω²)
                #                                           ^^^
                # The electronic dielectric ε_∞ screens the ionic contribution!
                # Without this, ionic contribution is overestimated by factor of ε_∞ (~3-20)
                prefactor = (e**2) / (eps0 * eps_inf_value * vol_m3 * (omega**2))
                if min_omega is None or omega < min_omega:
                    min_omega = float(omega)
                if max_prefactor is None or prefactor > max_prefactor:
                    max_prefactor = float(prefactor)
                used_modes += 1
                abs_freq = float(abs(freq))
                if min_abs_freq_used is None or abs_freq < min_abs_freq_used:
                    min_abs_freq_used = abs_freq
                if max_abs_freq_used is None or abs_freq > max_abs_freq_used:
                    max_abs_freq_used = abs_freq

                contribution = prefactor * np.outer(S_m, S_m)
                eps_ionic += contribution
                
                mode_polarities.append(S_m.tolist())
                mode_contributions.append(np.trace(contribution))
            
            epsilon_ionic = eps_ionic.tolist()
            try:
                a_mean_diag = float(np.mean(np.diag(eps_ionic)))
            except Exception:
                a_mean_diag = None
            ionic_diagnostics = {
                "vol_m3": float(vol_m3),
                "min_abs_freq_all_THz": min_abs_freq_all,
                "min_abs_freq_used_THz": min_abs_freq_used,
                "max_abs_freq_used_THz": max_abs_freq_used,
                "min_omega_Hz": min_omega,
                "max_prefactor": max_prefactor,
                "used_modes": used_modes,
                "max_born_abs": max_born_abs,
                "mean_born_abs": mean_born_abs,
                "max_mode_polarity_norm": max_sm_norm,
                "eigvecs_shape_before": eigvecs_shape_before,
                "eigvecs_shape_after": eigvecs_shape_after,
                "eigvecs_reshape_applied": reshape_applied,
                "mode_vec_norms": mode_vec_norms_stats,
                "mass_weighted_mode_vec_norms": mass_weighted_norms_stats,
                "ionic_A_mean_diag": a_mean_diag,
                "eps_inf_used": eps_inf_value,
                "sm_mass_weighting": "divide_by_sqrt_mass_amu",
            }
            
        except Exception as e:
            print(f"Error computing ionic dielectric: {e}")
            epsilon_ionic = None
            mode_polarities = []
            mode_contributions = []
            ionic_diagnostics = {"error": str(e)}

    return {
        "born_effective_charges": born_charges,
        "born_method": born_method,
        "electronic_dielectric_constant": None, # Missing model head
        "ionic_dielectric_constant_nominal": epsilon_ionic,
        "mode_polarities": mode_polarities,
        "mode_dielectric_contributions": mode_contributions,
        "note": "Born charges approximated from oxidation states. Electronic part not available."
        ,
        "ionic_diagnostics": ionic_diagnostics,
    }


# -----------------------------
# Structural Proxies
# -----------------------------
def get_structural_proxies(structure: Structure) -> dict[str, float]:
    """Calculate structural proxies that don't require MLIP."""
    # 1. Mass stats
    masses = [s.specie.atomic_mass for s in structure]
    mass_mean = np.mean(masses)
    mass_var = np.var(masses)
    mass_ratio = max(masses) / min(masses) if min(masses) > 0 else 0.0
    
    # 2. Packing Fraction
    # Need atomic radii. Use Pymatgen Element.atomic_radius
    total_atom_vol = 0.0
    for site in structure:
        r = site.specie.atomic_radius
        if r is None: r = 1.0 # Fallback
        total_atom_vol += (4/3) * np.pi * (r**3)
    packing_fraction = total_atom_vol / structure.volume
    
    # 3. Voronoi Volume Variance - (Placeholder as in original)
    voronoi_vol_var = 0.0 
    
    # Bond Length Variance
    # Get all neighbors up to 4A
    all_neighbors = structure.get_all_neighbors(r=4.0)
    all_dists = []
    for n_list in all_neighbors:
        if n_list:
            # Take nearest neighbor distance
            dists = [x.nn_distance for x in n_list]
            all_dists.append(min(dists))
    bond_length_var = np.var(all_dists) if all_dists else 0.0
    
    # 4. Local Chemical Variance
    # Variance of Z of neighbors
    local_chem_vars = []
    for i, n_list in enumerate(all_neighbors):
        if n_list:
            neighbor_z = [x.specie.number for x in n_list]
            local_chem_vars.append(np.var(neighbor_z))
    local_chem_var = np.mean(local_chem_vars) if local_chem_vars else 0.0
    
    # 5. Electronic Density Proxy
    # Valence electrons / Volume
    total_valence = sum([s.specie.Z for s in structure]) # Z is total electrons. Use Z for now.
    elec_density = total_valence / structure.volume

    return {
        "mass_mean": mass_mean,
        "mass_var": mass_var,
        "mass_ratio": mass_ratio,
        "packing_fraction": packing_fraction,
        "bond_length_variance": bond_length_var,
        "local_chemical_variance": local_chem_var,
        "electronic_density_proxy": elec_density
    }


# -----------------------------
# Structure conversion
# -----------------------------
def get_structures(input_data: Union[str, Atoms]):
    """Convert a composition string or ASE Atoms to a list of pymatgen Structures."""
    if isinstance(input_data, str):
        composition_strs = [input_data]
        structures = generate_optimized_sqs_structures_from_compositions(
            composition_strs
        )
    else:
        structures = [AseAtomsAdaptor.get_structure(input_data)]
    return structures


# -----------------------------
# Mechanical property helpers
# -----------------------------
def _calc_mech_props(structures, actor, config):
    """Calculate mechanical properties for a list of structures."""
    # Uses atlas.physics.mechanical.calc_mechanical which supports PhysicsActor
    mech_results = calc_mechanical(structures, config, actor)
    return [_extract_mech_props(mech_result) for mech_result in mech_results]


def _extract_mech_props(mech_result) -> dict[str, float]:
    """Extract key elastic constants and moduli from a mechanical result."""
    et = mech_result.elastic_tensor
    c11 = et[0][0]
    c12 = et[0][1]
    c44 = et[3][3]
    tetragonal_shear = c11 - c12
    zener_anisotropy = 2 * c44 / tetragonal_shear
    return {
        "C11": round(c11, 3),
        "C12": round(c12, 3),
        "C44": round(c44, 3),
        "TETRAGONAL_SHEAR": round(tetragonal_shear, 3),
        "ZENER_ANISOTROPY": round(zener_anisotropy, 3),
        "SHEAR_MODULUS": round(mech_result.shear_modulus, 3),
        "YOUNG_MODULUS": round(mech_result.young_modulus, 3),
        "BULK_MODULUS": round(mech_result.bulk_modulus, 3),
        "C": et,
    }


# -----------------------------
# Affine shear
# -----------------------------
def apply_affine_shear(atoms, gamma, plane="xy"):
    """Apply an affine shear strain to atoms along the given plane."""
    new_atoms = atoms.copy()
    cell = atoms.cell.array.copy()

    if plane == "xy":
        cell[0, 1] += gamma * cell[1, 1]
    elif plane == "xz":
        cell[0, 2] += gamma * cell[2, 2]
    elif plane == "yz":
        cell[1, 2] += gamma * cell[2, 2]
    else:
        raise ValueError("Unknown shear plane")

    new_atoms.set_cell(cell, scale_atoms=True)
    return new_atoms


def calc_affine_props_gpu(
    structures: list[Structure], actor, gamma: float = 0.01, fmax: float = 0.02
) -> list[dict[str, float]]:
    """Compute affine shear curvature and affine/relaxed ratio on GPU via actor."""
    structs_static = []
    structs_relax = []
    for s in structures:
        atoms0 = AseAtomsAdaptor.get_atoms(s)

        atoms_p = apply_affine_shear(atoms0, gamma, "xy")
        atoms_m = apply_affine_shear(atoms0, -gamma, "xy")

        s_p = AseAtomsAdaptor.get_structure(atoms_p)
        s_m = AseAtomsAdaptor.get_structure(atoms_m)
        s_0 = s

        structs_static.extend([s_p, s_m, s_0])
        structs_relax.extend([s_p, s_m])

    # actor.static and actor.relax are available on Atlas PhysicsActor
    future_static = actor.static.remote(
        structs_static, compute_forces=False, return_forces=False
    )
    future_relax = actor.relax.remote(
        structs_relax, nrelax=100, fmax=fmax, relax_type="atoms"
    )

    res_static, res_relax = ray.get([future_static, future_relax])

    results = []
    for i, s in enumerate(structures):
        Ep_static = res_static[3 * i].energy
        Em_static = res_static[3 * i + 1].energy
        E0_static = res_static[3 * i + 2].energy

        Ep_relaxed = res_relax.energies[2 * i]
        Em_relaxed = res_relax.energies[2 * i + 1]

        V = s.volume

        K_aff = (Ep_static + Em_static - 2 * E0_static) / (gamma**2 * V)
        K_rel = (Ep_relaxed + Em_relaxed - 2 * E0_static) / (gamma**2 * V)

        ratio = K_aff / K_rel if abs(K_rel) > 1e-9 else 0.0

        results.append({"AFFINE_SHEAR_CURVATURE": K_aff, "AFFINE_RELAXED_RATIO": ratio})

    return results


def get_dielectric_props(atoms: Atoms, actor=None) -> dict[str, Any]:
    """Compute dielectric, phonon, and thermal properties for an Atoms object."""
    structures = get_structures(atoms)
    structure = structures[0]  # Assume single structure for now

    # 0. Structural Proxies (Local)
    struct_proxies = get_structural_proxies(structure)

    if actor is None:
        tid = threading.get_ident()
        # Use atlas PhysicsActor
        actor = PhysicsActor.options(num_gpus=1, name=f"PhysicsActor_{tid}").remote(
            model_name="egip-inf"
        )
        created_actor = True
    else:
        created_actor = False

    # 1. Phonons (Remote task)
    phon_future = calculate_phonons.remote(
        actor, structure, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )

    # 2. Thermal Perturbations (Remote task)
    therm_future = run_thermal_perturbations.remote(actor, structure, temp=300.0)
    
    # 3. Anharmonicity & Strain Proxies (Remote tasks)
    anharm_future = calculate_anharmonicity.remote(actor, structure)
    int_strain_future = calculate_internal_strain_proxy.remote(actor, structure)
    gruneisen_future = calculate_gruneisen_proxy.remote(actor, structure)

    # Wait for phonons before dielectric (dependency)
    phon_res = ray.get(phon_future)

    # 4. Dielectric (Remote task, uses phonon data)
    # calculate_dielectric_properties does NOT need actor
    diel_future = calculate_dielectric_properties.remote(structure, phon_res)

    therm_res, diel_res, anharm_res, int_strain_res, grun_res = ray.get([
        therm_future, diel_future, anharm_future, int_strain_future, gruneisen_future
    ])

    if created_actor:
        ray.kill(actor)

    return {
        "phonons": phon_res, 
        "thermal": therm_res, 
        "dielectric": diel_res,
        "structural": struct_proxies,
        "anharmonicity": anharm_res,
        "internal_strain": int_strain_res,
        "gruneisen": grun_res
    }


# -----------------------------
# Main entry point for sampling
# -----------------------------
def get_elastic_props(atoms: Atoms, actor=None):
    """Compute elastic + affine shear properties for an Atoms object (used by sample_voxel)."""
    structures = get_structures(atoms)
    
    if actor is None:
        tid = threading.get_ident()
        actor = PhysicsActor.options(num_gpus=1, name=f"PhysicsActor_{tid}").remote(
            model_name="egip-inf"
        )
        created_actor = True
    else:
        created_actor = False
        
    mech_config = ConfigMechanical(relax_type="cell", fmax=0.05, nrelax=300)
    # _calc_mech_props uses calc_mechanical which works with PhysicsActor
    mech_props = _calc_mech_props(structures, actor, mech_config)[0]
    affine_props = calc_affine_props_gpu(structures, actor)[0]
    
    if created_actor:
        ray.kill(actor)
        
    return mech_props | affine_props


# -----------------------------
# Test dielectric atoms object
# -----------------------------
def get_test_atoms():
    import os as _os
    _data = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "mp-557309-mp.extxyz")
    return read(_data)

def get_extended_atoms():
    import os as _os
    _data = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "mp-9996-mp.extxyz")
    return read(_data)

def test():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        
    atoms = get_test_atoms()
    print("Got atoms object:", atoms.info, "Getting dielectric props")

    props = get_dielectric_props(atoms)
    print("--------------------------------------------------")
    print(f"Lowest Optical Mode (THz): {props['phonons']['lowest_optical_frequency_THz']:.4f}")
    print(f"Avg Optical Mode (THz):    {props['phonons']['avg_optical_frequency_THz']:.4f}")
    print(f"Phonon Bandwidth (THz):    {props['phonons']['phonon_bandwidth']:.4f}")
    print(f"Acoustic Slope:            {props['phonons']['acoustic_slope']:.4f}")
    print(f"Vib Entropy (300K):        {props['phonons']['vibrational_entropy_300K']:.4f}")
    print(f"Lattice Instability:       {props['phonons']['has_instability']}")
    
    print("--------------------------------------------------")
    print(f"Structural Mass Mean:      {props['structural']['mass_mean']:.4f}")
    print(f"Packing Fraction:          {props['structural']['packing_fraction']:.4f}")
    print(f"Bond Length Var:           {props['structural']['bond_length_variance']:.4f}")
    print(f"Electronic Density:        {props['structural']['electronic_density_proxy']:.4f}")
    
    print("--------------------------------------------------")
    print(f"Anharmonicity Score:       {props['anharmonicity']['harmonicity_score']:.4f}")
    print(f"Internal Strain RMS:       {props['internal_strain']:.6f}")
    if props['gruneisen']['avg_gruneisen'] is not None:
        print(f"Mini-Gruneisen:            {props['gruneisen']['avg_gruneisen']:.4f}")
    else:
        print("Mini-Gruneisen:            Failed")

    print("--------------------------------------------------")
    if props["thermal"].get("position_variance") is not None:
        print("Thermal Variance:", props["thermal"]["position_variance"])
    else:
        print("Thermal Variance: Failed (see errors)")
        
    print("Ionic Dielectric (Nominal):", props["dielectric"]["ionic_dielectric_constant_nominal"])


if __name__ == "__main__":
    test()

import os
import signal
import threading

import numpy as np
import spglib
from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import neighbor_list
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from typing import Optional

from .versions import *


def _ox_timeout_handler(signum, frame):
    raise TimeoutError("BVAnalyzer timeout")

# Auto-bin integration: when a VersionSpec (or caller) provides a binning
# sidecar, we bin the continuous properties via chem.auto_bin.label_value.
# Otherwise we fall back to the legacy hard-coded ranges below. The
# auto-bin path is the new default — see docs/AutoBinDeferredWork.md.
# Map from legacy property names to the keys used in the sidecar JSON.
_AUTOBIN_PROP_MAP = {
    "band_gap": "band_gap",
    "stability": "stability",
    "eps_0": "nequip_eps_0",
    "density": "density",
}


class FieldFormatter:
    """Computes and formats individual segments for dataset rows.

    `binnings` (optional) — dict[prop_name, chem.auto_bin.Binning]. When set,
    the affected properties are binned via the auto-bin spec instead of the
    legacy hard-coded ranges. Keys match the legacy names (`band_gap`,
    `stability`, `eps_0`, `density`). Constructed via
    `chem.auto_bin.load_binnings(path)`.
    """

    def __init__(self, binnings: Optional[dict] = None):
        self.spg_symbols = self._build_spg_tables()
        self.binnings = binnings or {}

    def _build_spg_tables(self):
        symbols = {}
        for hall in range(1, 531):
            st = spglib.get_spacegroup_type(hall)
            ita = st.number
            if ita not in symbols:
                symbols[ita] = st.international_short
        return symbols

    def get_crystal_system(self, spg_number):
        if spg_number <= 2: return "triclinic"
        if spg_number <= 15: return "monoclinic"
        if spg_number <= 74: return "orthorhombic"
        if spg_number <= 142: return "tetragonal"
        if spg_number <= 167: return "trigonal"
        if spg_number <= 194: return "hexagonal"
        return "cubic"

    def get_reduced_formula(self, numbers):
        atoms = Atoms(numbers=numbers)
        return atoms.get_chemical_formula(mode="metal")

    def get_anonymous_formula(self, numbers):
        count = Counter(numbers)
        sorted_counts = sorted(count.values(), reverse=True)
        g = reduce(gcd, sorted_counts)
        reduced_counts = sorted([c // g for c in sorted_counts], reverse=True)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        anon_parts = []
        for i, c in enumerate(reduced_counts):
            if c == 1: anon_parts.append(letters[i])
            else: anon_parts.append(f"{letters[i]}{c}")
        return "".join(anon_parts)

    def bin_property(self, prop_name, val):
        if val is None: return None
        # Auto-bin path (when a sidecar Binning was injected at construction)
        if prop_name in _AUTOBIN_PROP_MAP and self.binnings:
            spec_key = _AUTOBIN_PROP_MAP[prop_name]
            b = self.binnings.get(spec_key)
            if b is not None:
                from chem.auto_bin import label_value
                return label_value(val, b)
        if prop_name == "band_gap":
            if val < 0.1: return "metal"
            if val < 0.6: return "narrow-gap"
            if val < 1.5: return "small-gap"
            if val < 3.0: return "semiconductor"
            if val < 4.0: return "wide-gap"
            return "very-wide-gap"
        if prop_name == "stability":
            if val < 0.05: return "stable"
            if val <= 0.1: return "metastable"
            return "unstable"
        if prop_name == "eps_0":
            if val < 3:  return "very-low-k"
            if val < 5:  return "low-k"
            if val < 10: return "medium-k"
            if val < 20: return "high-k"
            return "very-high-k"
        if prop_name == "eps_inf":
            if val < 3: return "low-n"
            if val <= 6: return "medium-n"
            return "high-n"
        # IC-relevant property bins
        if prop_name == "cohesive_energy":
            aval = abs(val)
            if aval > 6.0: return "very-high-cohesive"
            if aval > 4.5: return "high-cohesive"
            if aval > 3.5: return "medium-cohesive"
            return "low-cohesive"
        if prop_name == "crystal_system":
            # val is a string here
            return str(val) if val else None
        if prop_name == "mean_coordination":
            if val > 10: return "high-CN"
            if val >= 6: return "medium-CN"
            return "low-CN"
        if prop_name == "debye_temperature":
            if val > 500: return "high-debye"
            if val > 300: return "medium-debye"
            return "low-debye"
        if prop_name == "elastic_anisotropy":
            # Zener anisotropy: 0 = isotropic, >1 = anisotropic
            if val < 0.5: return "isotropic"
            if val < 2.0: return "low-anisotropy"
            return "anisotropic"
        if prop_name == "ductility":
            # Pugh ratio B/G: >1.75 = ductile
            if val > 2.5: return "very-ductile"
            if val > 1.75: return "ductile"
            return "brittle"
        if prop_name == "oxide_resistance":
            # Oxide formation energy per metal atom (eV); more negative = easier to oxidize
            if val > -0.5: return "oxidation-resistant"
            if val > -2.0: return "moderate-oxidation"
            return "oxidation-prone"
        if prop_name == "topological_class":
            # Topological classification from literature / database lookup
            if val in ("weyl", "weyl-type-ii", "dirac", "triple-point",
                       "chiral", "nodal-line", "nodal-arc",
                       "enforced-semimetal", "enforced-semimetal-fd",
                       "topological-semimetal", "z2-semimetal"):
                return "topological-semimetal"
            if val == "topological-insulator":
                return "topological-insulator"
            if val == "potential-topological":
                return "potential-topological"
            return None  # trivial or unknown — don't add a label
        if prop_name == "max_phase":
            return "max-phase" if val else None
        if prop_name == "sg_label":
            # Explicit SG label for important/rare spacegroups
            # Enables SG-conditioned generation: prompt with sg-109 → generate I4_1md
            sg_map = {
                109: "sg-109",   # I4_1md — Weyl semimetals (NbP, TaAs)
                198: "sg-198",   # P2_13 — chiral semimetals (CoSi, RhSi)
                187: "sg-187",   # P-6m2 — triple-point (MoP)
                129: "sg-129",   # P4/nmm — nodal-line (ZrSiS)
                194: "sg-194",   # P6_3/mmc — MAX phases, hexagonal metals
                221: "sg-221",   # Pm-3m — simple cubic / perovskite
                225: "sg-225",   # Fm-3m — FCC
                229: "sg-229",   # Im-3m — BCC
                216: "sg-216",   # F-43m — half-Heusler
                139: "sg-139",   # I4/mmm — body-centered tetragonal
                166: "sg-166",   # R-3m — rhombohedral
                62: "sg-62",     # Pnma — orthorhombic (common)
                12: "sg-12",     # C2/m — monoclinic (common)
            }
            return sg_map.get(val)
        if prop_name == "nelements":
            # Number of distinct elements — useful for targeting binary/ternary
            if val == 1: return "unary"
            if val == 2: return "binary"
            if val == 3: return "ternary"
            if val >= 4: return "quaternary"
            return None
        if prop_name == "natoms_bin":
            # Atoms per cell — coarser than the explicit natoms in the source
            if val <= 2: return "small-cell"
            if val <= 8: return "medium-cell"
            if val <= 16: return "large-cell"
            return "very-large-cell"
        return None

    def get_lattice_string(self, atoms, decimals=2, angle_decimals=1):
        params = atoms.cell.cellpar()
        l_fmt = f"{{:.{decimals}f}}"
        a_fmt = f"{{:.{angle_decimals}f}}"
        return " ".join([l_fmt.format(params[i]) for i in range(3)] + 
                        [a_fmt.format(params[i]) for i in range(3, 6)])

    def get_atoms_string(self, atoms, wyckoff_letters=None, decimals=2):
        numbers = atoms.numbers
        frac = atoms.get_scaled_positions() % 1.0
        # Sort by Z, Y, X for canonical ordering
        sort_keys = list(zip(numbers, frac[:, 2], frac[:, 1], frac[:, 0]))
        order = sorted(range(len(numbers)), key=lambda i: sort_keys[i])
        
        c_fmt = f"{{:.{decimals}f}}"
        parts = []
        for i in order:
            sym = chemical_symbols[numbers[i]]
            fx, fy, fz = frac[i]
            if wyckoff_letters is not None:
                parts.append(f"{sym} {wyckoff_letters[i]} {c_fmt.format(fx)} {c_fmt.format(fy)} {c_fmt.format(fz)}")
            else:
                parts.append(f"{sym} {c_fmt.format(fx)} {c_fmt.format(fy)} {c_fmt.format(fz)}")
        return " ".join(parts)

    def get_eref_string(self, elements, elem_refs):
        if not elem_refs: return ""
        parts = []
        for elem in sorted(elements):
            if elem in elem_refs:
                parts.append(f"{elem}:{elem_refs[elem]:.2f}")
        return f"Eref {' '.join(parts)}" if parts else ""

    def get_scratchpad_fields(self, atoms, wyckoff_letters, include_ox=False, ef_per_atom=None):
        """Compute OX, CN, NN, WP fields."""
        try:
            i_idx, _, d = neighbor_list('ijd', atoms, cutoff=5.0)
        except Exception:
            return {}

        natoms = len(atoms)
        cn = np.zeros(natoms, dtype=int)
        nn_dist = np.full(natoms, np.inf)
        for ii in range(len(i_idx)):
            a = i_idx[ii]
            cn[a] += 1
            if d[ii] < nn_dist[a]: nn_dist[a] = d[ii]
        nn_dist = np.where(np.isinf(nn_dist), 0.0, nn_dist)

        ox_per_atom = None
        if include_ox:
            from pymatgen.core import Structure, Lattice
            from pymatgen.analysis.bond_valence import BVAnalyzer
            # Strict per-structure timeout on BVAnalyzer (the combinatorial blowup).
            # OX_TIMEOUT_SEC env var: 0 = off (legacy). Only arms in the main thread
            # (forked precompute workers qualify; thread pools don't). On timeout the
            # bare except leaves OX empty -> structure proceeds, revisit later.
            _ox_to = int(os.environ.get("OX_TIMEOUT_SEC", "0") or 0)
            _arm = _ox_to > 0 and threading.current_thread() is threading.main_thread()
            _old = None
            try:
                struct = Structure(Lattice(atoms.cell[:]), [chemical_symbols[n] for n in atoms.numbers], atoms.get_scaled_positions())
                if _arm:
                    _old = signal.signal(signal.SIGALRM, _ox_timeout_handler)
                    signal.alarm(_ox_to)
                ox_per_atom = [int(v) for v in BVAnalyzer().get_valences(struct)]
            except:
                pass
            finally:
                if _arm:
                    signal.alarm(0)
                    if _old is not None:
                        signal.signal(signal.SIGALRM, _old)

        site_info = defaultdict(lambda: {'cn': [], 'nn': [], 'ox': [], 'mult': 0})
        for a in range(natoms):
            key = (chemical_symbols[atoms.numbers[a]], wyckoff_letters[a] if wyckoff_letters is not None else 'a')
            site_info[key]['cn'].append(cn[a])
            site_info[key]['nn'].append(nn_dist[a])
            if ox_per_atom: site_info[key]['ox'].append(ox_per_atom[a])
            site_info[key]['mult'] += 1

        fields = {}
        if ef_per_atom is not None: fields['Ef'] = f"{ef_per_atom:.2f}"
        
        ox_list, cn_list, nn_list, wp_list = [], [], [], []
        for (sym, wl), info in sorted(site_info.items()):
            cn_list.append(str(int(round(np.mean(info['cn'])))))
            nn_list.append(f"{np.mean(info['nn']):.2f}")
            wp_list.append(f"{wl}:{info['mult']}")
            if ox_per_atom:
                most_common_ox = Counter(info['ox']).most_common(1)[0][0]
                ox_list.append(f"{most_common_ox:+d}")
        
        fields['OX'] = ";".join(ox_list)
        fields['CN'] = ";".join(cn_list)
        fields['NN'] = ";".join(nn_list)
        fields['WP'] = ";".join(wp_list)

        # NP: nearest pair distances per element pair (for chain-of-thought planning)
        # Format: NP:El1-El2:dist;El1-El3:dist;...
        try:
            from itertools import combinations_with_replacement
            symbols = [chemical_symbols[n] for n in atoms.numbers]
            unique_elements = sorted(set(symbols))
            pair_dists = {}
            for ii in range(len(i_idx)):
                sym_a = symbols[i_idx[ii]]
                j_idx_val = int(d[ii] > 0)  # skip self
                # i_idx, j_idx from neighbor_list('ijd',...) — need j_idx too
                # We already have i_idx and d, but not j_idx. Recompute with 'ij'
                pass
            # Recompute with j indices
            i_arr, j_arr, d_arr = neighbor_list('ijd', atoms, cutoff=5.0)
            for ii in range(len(i_arr)):
                if d_arr[ii] < 0.1:
                    continue  # skip near-zero distances
                sym_a = symbols[i_arr[ii]]
                sym_b = symbols[j_arr[ii]]
                pair_key = tuple(sorted([sym_a, sym_b]))
                if pair_key not in pair_dists or d_arr[ii] < pair_dists[pair_key]:
                    pair_dists[pair_key] = float(d_arr[ii])
            np_parts = []
            for pair in sorted(pair_dists.keys()):
                np_parts.append(f"{pair[0]}-{pair[1]}:{pair_dists[pair]:.2f}")
            fields['NP'] = ";".join(np_parts)
        except Exception:
            fields['NP'] = ""

        return fields

    def compute_structure_fields(self, atoms, include_ox=True):
        """Compute all version-independent fields for one structure.

        Returns a dict suitable for caching — no version-specific formatting,
        just the raw computed values that are expensive to produce (spglib,
        BVAnalyzer, neighbor_list) plus pre-formatted lattice/atoms strings.
        """
        info = {}

        dataset = spglib.get_symmetry_dataset(
            (atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=0.1
        )
        if dataset:
            info['spg_num'] = dataset.number
            info['spg_sym'] = dataset.international
            info['wyckoff'] = list(dataset.wyckoffs)
        else:
            info['spg_num'], info['spg_sym'], info['wyckoff'] = 1, "P1", ["a"] * len(atoms)

        info['crystal_sys'] = self.get_crystal_system(info['spg_num'])
        info['formula'] = self.get_reduced_formula(atoms.numbers)
        info['anon_formula'] = self.get_anonymous_formula(atoms.numbers)
        info['elements'] = sorted(set(chemical_symbols[n] for n in atoms.numbers))
        info['element_counts'] = {chemical_symbols[n]: c for n, c in Counter(atoms.numbers).items()}
        info['natoms'] = len(atoms)
        info['density'] = round(atoms.get_masses().sum() * 1.66054 / atoms.get_volume(), 4)
        info['lattice'] = self.get_lattice_string(atoms)
        info['atoms_string'] = self.get_atoms_string(atoms, info['wyckoff'])

        sp = self.get_scratchpad_fields(atoms, info['wyckoff'], include_ox=include_ox)
        info.update(sp)
        # NP is always computed alongside scratchpad (cheap, uses same neighbor list)

        return info

    def format_row(self, version_id, origin, atoms, info):
        """Main entry point for formatting a source/target pair from an Atoms object."""
        spec = get_version(version_id)

        # Precompute common info if not present
        if 'spg_num' not in info:
            dataset = spglib.get_symmetry_dataset((atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=0.1)
            if dataset:
                info['spg_num'] = dataset.number
                info['spg_sym'] = dataset.international
                info['wyckoff'] = list(dataset.wyckoffs)
            else:
                info['spg_num'], info['spg_sym'], info['wyckoff'] = 1, "P1", ["a"]*len(atoms)

        info['crystal_sys'] = self.get_crystal_system(info['spg_num'])
        info['anon_formula'] = self.get_anonymous_formula(atoms.numbers)
        info['formula'] = self.get_reduced_formula(atoms.numbers)
        info['elements'] = sorted(set(chemical_symbols[n] for n in atoms.numbers))
        info['natoms'] = len(atoms)
        info['lattice'] = self.get_lattice_string(atoms)
        info['atoms_string'] = self.get_atoms_string(atoms, info['wyckoff'])

        # Compute scratchpad fields if needed
        needs_scratchpad = any(f in spec.target_segments[origin] for f in [T_SCRATCHPAD, T_OX, T_WP, T_CN, T_NN, T_EF])
        if needs_scratchpad:
            sp_info = self.get_scratchpad_fields(atoms, info['wyckoff'],
                                                 include_ox='ox' in info or version_id in ('d2','d3','d4','d5','d6'),
                                                 ef_per_atom=info.get('ef'))
            info.update(sp_info)

        return self._format_fields(version_id, origin, info)

    def format_cached_row(self, version_id, origin, cached_fields):
        """Format source/target from a pre-computed cache dict (no Atoms object needed)."""
        info = dict(cached_fields)
        # Cache stores pre-formatted atoms string under 'atoms_string'
        return self._format_fields(version_id, origin, info)

    def _format_fields(self, version_id, origin, info):
        """Assemble source/target strings from a complete info dict."""
        spec = get_version(version_id)

        def get_field(field_id):
            if field_id == S_CRYSTAL_SYS_SG:
                return f"{info['crystal_sys']} {info['spg_num']} {info['spg_sym']}" if version_id != 'd1' else f"{info['spg_num']} {info['spg_sym']}"
            if field_id == S_ANON_FORMULA_ELEMENTS:
                return f"{info['anon_formula']} {' '.join(info['elements'])}"
            if field_id == S_ELEMENTS_ONLY:
                return ' '.join(info['elements'])
            if field_id == S_NATOMS: return str(info['natoms'])
            if field_id == S_DENSITY: return f"{info.get('density', 0.0):.1f}"
            if field_id == S_PROP_LABELS:
                labels = [self.bin_property(p, info.get(p)) for p in ['band_gap', 'stability', 'eps_0', 'eps_inf']]
                return " ".join([l for l in labels if l])
            if field_id == S_PROP_LABELS_IC:
                # Standard labels + IC-relevant labels
                # Map cache field names to property binner names
                crystal_sys = info.get('crystal_system') or info.get('crystal_sys')
                mean_cn = info.get('mean_coordination')
                # Derive mean CN from scratchpad CN field if needed
                if mean_cn is None and info.get('CN'):
                    try:
                        cn_vals = [int(x) for x in info['CN'].split(';') if x.strip()]
                        mean_cn = sum(cn_vals) / len(cn_vals) if cn_vals else None
                    except (ValueError, ZeroDivisionError):
                        mean_cn = None
                labels = [self.bin_property(p, info.get(p)) for p in [
                    'band_gap', 'stability', 'eps_0', 'eps_inf',
                ]]
                # IC-specific bins
                labels.append(self.bin_property('cohesive_energy', info.get('cohesive_energy')))
                labels.append(self.bin_property('crystal_system', crystal_sys))
                labels.append(self.bin_property('mean_coordination', mean_cn))
                labels.append(self.bin_property('debye_temperature', info.get('debye_temperature')))
                labels.append(self.bin_property('elastic_anisotropy', info.get('elastic_anisotropy')))
                # v2 labels: ductility (from Pugh ratio), oxide resistance
                labels.append(self.bin_property('ductility', info.get('pugh_ratio')))
                labels.append(self.bin_property('oxide_resistance', info.get('oxide_formation_energy')))
                # v3 labels: topological classification, MAX phase
                labels.append(self.bin_property('topological_class', info.get('topological_class')))
                labels.append(self.bin_property('max_phase', info.get('max_phase')))
                # v4 labels: explicit SG, element count, cell size
                labels.append(self.bin_property('sg_label', info.get('spg_num')))
                labels.append(self.bin_property('nelements', len(info.get('elements', [])) or None))
                labels.append(self.bin_property('natoms_bin', info.get('natoms')))
                return " ".join([l for l in labels if l])
            if field_id == S_EREF: return self.get_eref_string(info.get('elements', []), info.get('elem_refs'))
            if field_id == T_FORMULA: return info['formula']
            if field_id == T_SG_INFO: return f"SG {info['spg_num']} {info['spg_sym']}"
            if field_id == T_SG_INFO_SYS: return f"SG {info['spg_num']} {info['spg_sym']} {info['crystal_sys']}"
            if field_id == T_SCRATCHPAD:
                parts = []
                if 'Ef' in info: parts.append(f"Ef:{info['Ef']}")
                if 'OX' in info: parts.append(f"OX:{info['OX']}")
                parts.extend([f"CN:{info['CN']}", f"NN:{info['NN']}"])
                if info.get('NP'): parts.append(f"NP:{info['NP']}")
                parts.append(f"WP:{info['WP']}")
                return " ".join(parts)
            if field_id == T_LATTICE: return info['lattice']
            if field_id == T_ATOMS: return info.get('atoms_string', '')
            if field_id == T_ANON_ELEMENTS: return f"{info['anon_formula']} {' '.join(info['elements'])}"
            if field_id == T_OX: return f"OX:{info.get('OX', '')}"
            if field_id == T_WP: return f"WP:{info.get('WP', '')}"
            if field_id == T_CN: return f"CN:{info.get('CN', '')}"
            if field_id == T_NN: return f"NN:{info.get('NN', '')}"
            if field_id == T_EREF: return self.get_eref_string(info.get('elements', []), info.get('elem_refs'))
            if field_id == T_EF: return f"Ef:{info.get('Ef', '')}"
            return ""

        source = " | ".join([get_field(f) for f in spec.source_segments[origin]])
        target = " | ".join([get_field(f) for f in spec.target_segments[origin]])
        return source, target

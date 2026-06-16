from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class VersionSpec:
    version_id: str
    description: str
    # origin -> List[field_name]
    source_segments: Dict[str, List[str]]
    # origin -> List[field_name]
    target_segments: Dict[str, List[str]]
    # Whether this version uses the 3-column legacy format (source, target, src_group)
    # instead of the 4-column format (source, target, id, origin)
    legacy_columns: bool = False
    # Optional path to an auto-bin sidecar JSON (see chem/auto_bin.py).
    # When set, FieldFormatter.bin_property uses auto_bin.label_value for
    # continuous properties instead of the legacy hard-coded bins. None =
    # legacy behavior.
    binning_spec_path: Optional[str] = None

# Field definitions for reuse
# Source Fields
S_CRYSTAL_SYS_SG = "crystal_sys_sg"           # "SG# SG_sym" or "crystal_sys SG# SG_sym"
S_ANON_FORMULA_ELEMENTS = "anon_formula_elements" # "ABC2 Li Fe O"
S_NATOMS = "natoms"                           # "12"
S_DENSITY = "density"                         # "3.5" (1-dec)
S_PROP_LABELS = "prop_labels"                 # "metal high-k ..."
S_EREF = "eref"                               # "Eref Li:-1.91 O:-4.94"
S_ELEMENTS_ONLY = "elements_only"             # "Li Fe O" (no anon formula prefix)
S_PROP_LABELS_IC = "prop_labels_ic"           # "metal stable high-cohesive cubic" (extended with IC labels)

# Target Fields
T_FORMULA = "formula"                         # "LiFePO4"
T_SG_INFO = "sg_info"                         # "SG 62 Pnma"
T_SG_INFO_SYS = "sg_info_sys"                 # "SG 62 Pnma orthorhombic"
T_SCRATCHPAD = "scratchpad"                   # Legacy blob: "OX:+1;+2;+5;-2 CN:6;6;4;4 ..."
T_LATTICE = "lattice"                         # "10.3 6.0 4.7 90.0 90.0 90.0"
T_ATOMS = "atoms"                             # "Li a 0.0 0.0 0.0 Fe c 0.28 0.25 0.97 ..."
T_ANON_ELEMENTS = "anon_elements"             # "A8B2CD Ag Bi Mo O"
T_OX = "ox"                                   # "OX:+1;+3;+6;+6;-2"
T_WP = "wp"                                   # "WP:c:1;b:1;a:1;d:1;g:8"
T_CN = "cn"                                   # "CN:36;36;40;40;35"
T_NN = "nn"                                   # "NN:2.56;2.48;1.79;1.79;1.79"
T_EREF = "eref"                               # "Eref Ag:-2.82 Bi:-3.85 ..."
T_EF = "ef"                                   # "Ef:-0.77"

# --- VERSION REGISTRY ---

VERSIONS: Dict[str, VersionSpec] = {
    "d16_ic": VersionSpec(
        version_id="d16_ic",
        description="IC v4: d14 format + SG labels (sg-109, sg-198, ...) + nelements + natoms_bin",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d18_ic": VersionSpec(
        version_id="d18_ic",
        description="IC v6: d16 format + 271 clean MAX phases (JARVIS+AFLOW, traditional A-elements, band_gap/CN overrides)",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d14_ic": VersionSpec(
        version_id="d14_ic",
        description="IC v3: d13 format + topological semimetal + MAX phase labels",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d13_ic": VersionSpec(
        version_id="d13_ic",
        description="IC v2: d12 format + ductility (Pugh ratio) + oxide resistance labels",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d12_ic": VersionSpec(
        version_id="d12_ic",
        description="IC-focused: d12 format (scratchpad+NP before atoms) + IC property labels (crystal system, cohesive energy, coordination)",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d11_ic_planning": VersionSpec(
        version_id="d11_ic_planning",
        description="IC-focused: scratchpad before atoms (planning), extended prop labels (cohesive energy, crystal system, coordination)",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS_IC],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d1": VersionSpec(
        version_id="d1",
        description="Symmetry (no scratchpad)",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_LATTICE, T_ATOMS],
        },
        legacy_columns=True
    ),
    "d2": VersionSpec(
        version_id="d2",
        description="Scratchpad + OX",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
        legacy_columns=False  # was True; changed to get id+origin columns
    ),
    "d2c": VersionSpec(
        version_id="d2c",
        description="d2 cleaned: base format (SG + elements + natoms + density) extracted from d3 with properties removed, filtered to valid IDs from master_ids",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
        legacy_columns=False
    ),
    "d3": VersionSpec(
        version_id="d3",
        description="Scratchpad + OX + Ef + Eref",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d3c": VersionSpec(
        version_id="d3c",
        description="d3 cleaned to exclude master_ids rows flagged as giant/overlong structures",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4": VersionSpec(
        version_id="d4",
        description="d3 + OMAT Eref + OMAT OX",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_EREF],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4c": VersionSpec(
        version_id="d4c",
        description="d4 cleaned to exclude master_ids rows flagged as giant/overlong structures",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_EREF],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_no_sg": VersionSpec(
        version_id="d4_no_sg",
        description="d4 with crystal-system / space-group source segment removed",
        source_segments={
            "mp": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "diel": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS, S_EREF],
            "omat": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_EREF],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props": VersionSpec(
        version_id="d4_props",
        description="d4 with SG, anon elements, and Eref removed from the source",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_tgt": VersionSpec(
        version_id="d4_props_anon_tgt",
        description="d4 with properties-only source and anon stoichiometry/elements moved early in target",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_src": VersionSpec(
        version_id="d4_props_anon_src",
        description="d4 with properties-only source plus anon stoichiometry/elements restored in the source",
        source_segments={
            "mp": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_eref_tgt": VersionSpec(
        version_id="d4_props_anon_eref_tgt",
        description="d4 with properties-only source, anon stoichiometry/elements early in target, and Eref moved to target",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_sg_sys_tgt": VersionSpec(
        version_id="d4_props_anon_sg_sys_tgt",
        description="d4 with properties-only source, anon stoichiometry/elements early in target, and crystal system restored in SG target field",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_causal_tgt": VersionSpec(
        version_id="d4_props_anon_causal_tgt",
        description="d4 with properties-only source and a more causal target ordering: anon -> formula -> SG -> lattice -> atoms -> scratchpad",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    # --- d5 variants: source-richness ablations on causal_tgt target ---
    "d5_elements_causal_tgt": VersionSpec(
        version_id="d5_elements_causal_tgt",
        description="causal_tgt target with element list restored to source (tests if composition info stabilizes training)",
        source_segments={
            "mp": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d5_chem_causal_tgt": VersionSpec(
        version_id="d5_chem_causal_tgt",
        description="causal_tgt target with crystal system + SG in source instead of elements (tests if structural info stabilizes training)",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d7_chem_ef_causal_tgt": VersionSpec(
        version_id="d7_chem_ef_causal_tgt",
        description="V4 format (SG source, causal target) + Eref/Ef in target for thermodynamic grounding",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_CRYSTAL_SYS_SG, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_EREF, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d12_elements_np_think_tgt": VersionSpec(
        version_id="d12_elements_np_think_tgt",
        description="d11 reordered: scratchpad(+NP) BEFORE lattice/atoms (chain-of-thought ordering)",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d13_mixed_lemat_ehull": VersionSpec(
        version_id="d13_mixed_lemat_ehull",
        description="d12 + 4 source segments for all origins + ehull stability labels in prop_labels. Renamed 2026-05-03 from d13_nolemat_ehull (the dataset is mixed origins, lemat-bulk membership is incidental, not filtered).",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d13_premium_ehull": VersionSpec(
        version_id="d13_premium_ehull",
        description="Premium DFT-quality subset: dielectrics + MP + Alexandria (+ optional WBM). Same format as d13_mixed_lemat_ehull.",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d15_binrho_k7": VersionSpec(
        version_id="d15_binrho_k7",
        description="Same as d15_binrho but density binned into 7 tiers (ulow/vlow/low/mid/high/vhigh/uhigh) instead of 5. Tests whether finer density granularity recovers the conditioning signal lost vs decimal density.",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d15_binrho": VersionSpec(
        version_id="d15_binrho",
        description="Ablation of d13_premium_ehull where the decimal density segment is replaced by a rho-* bin tier and merged into the prop-labels segment. 3-segment source: elements_only | natoms | prop_labels (now containing bg/k/hull/rho).",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d13_diel_finetune": VersionSpec(
        version_id="d13_diel_finetune",
        description="Dielectric-only fine-tuning subset: diel origin only, same format as d13_premium_ehull for tokenizer compatibility.",
        source_segments={
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d11_elements_np_causal_tgt": VersionSpec(
        version_id="d11_elements_np_causal_tgt",
        description="d9 + NP (per-pair nearest distances from structure) in scratchpad after NN",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d10_elements_md_causal_tgt": VersionSpec(
        version_id="d10_elements_md_causal_tgt",
        description="d9 + MD (minimum pair distances from covalent radii) prepended to scratchpad",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d9_elements_4bin_ef_causal_tgt": VersionSpec(
        version_id="d9_elements_4bin_ef_causal_tgt",
        description="V3 format + 4-bin stability (on-hull/near-hull/metastable/unstable) + Ef bucket tag in source",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d8_elements_ef_causal_tgt": VersionSpec(
        version_id="d8_elements_ef_causal_tgt",
        description="V3 format (elements_only source) + Ef bucket tag in source (very-stable-ef/stable-ef/unstable-ef)",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
            "alex": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "wbm": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "alex": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "wbm": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d5_elements_only_causal_tgt": VersionSpec(
        version_id="d5_elements_only_causal_tgt",
        description="causal_tgt target with element names only (no stoichiometry) in source — model must invent ratios",
        source_segments={
            "mp": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_ELEMENTS_ONLY, S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_LATTICE, T_ATOMS, T_SCRATCHPAD],
        },
    ),
    "d4_props_anon_eref_sg_sys_tgt": VersionSpec(
        version_id="d4_props_anon_eref_sg_sys_tgt",
        description="d4 with properties-only source, anon stoichiometry/elements and Eref early in target, and crystal system restored in SG target field",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO_SYS, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_eref_ox_tgt": VersionSpec(
        version_id="d4_props_anon_eref_ox_tgt",
        description="d4 with properties-only source, anon stoichiometry/elements and Eref early in target, and oxidation state split into its own target field",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d4_props_anon_eref_ox_ef_tgt": VersionSpec(
        version_id="d4_props_anon_eref_ox_ef_tgt",
        description="d4 with properties-only source, anon stoichiometry/elements and explicit OX/Eref/Ef target fields before a reduced CN/NN/WP scratchpad",
        source_segments={
            "mp": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_NATOMS, S_DENSITY],
        },
        target_segments={
            "mp": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_EF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_EF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_ANON_ELEMENTS, T_FORMULA, T_SG_INFO, T_OX, T_EREF, T_EF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d2p": VersionSpec(
        version_id="d2p",
        description="d2 base format + computed property tags (Tier 0+1): Bravais lattice, chemical family, element class, bonding, complexity, density, mass. Derived from d3, filtered by master_ids, Eref dropped.",
        source_segments={
            "mp": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "diel": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
            "omat": [S_CRYSTAL_SYS_SG, S_ANON_FORMULA_ELEMENTS, S_NATOMS, S_DENSITY, S_PROP_LABELS],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
        legacy_columns=False
    ),
    "d5": VersionSpec(
        version_id="d5",
        description="Minimal Source / Maximal Scratchpad",
        source_segments={
            "mp": [S_DENSITY, S_NATOMS, S_PROP_LABELS],
            "diel": [S_DENSITY, S_NATOMS, S_PROP_LABELS],
            "omat": [S_DENSITY, S_NATOMS],
        },
        target_segments={
            "mp": [T_FORMULA, T_SG_INFO_SYS, T_ANON_ELEMENTS, S_NATOMS, S_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "diel": [T_FORMULA, T_SG_INFO_SYS, T_ANON_ELEMENTS, S_NATOMS, S_EREF, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
            "omat": [T_FORMULA, T_SG_INFO_SYS, T_ANON_ELEMENTS, S_NATOMS, T_SCRATCHPAD, T_LATTICE, T_ATOMS],
        },
    ),
    "d6": VersionSpec(
        version_id="d6",
        description="Causal Target Ordering (current)",
        source_segments={
            "mp": [S_DENSITY, S_NATOMS, S_PROP_LABELS],
            "diel": [S_DENSITY, S_NATOMS, S_PROP_LABELS],
            "omat": [S_DENSITY, S_NATOMS],
        },
        target_segments={
            "mp": [T_FORMULA, T_ANON_ELEMENTS, T_SG_INFO_SYS, T_OX, S_NATOMS, T_LATTICE, T_WP, T_ATOMS, T_CN, T_NN, T_EREF, T_EF],
            "diel": [T_FORMULA, T_ANON_ELEMENTS, T_SG_INFO_SYS, T_OX, S_NATOMS, T_LATTICE, T_WP, T_ATOMS, T_CN, T_NN, T_EREF, T_EF],
            "omat": [T_FORMULA, T_ANON_ELEMENTS, T_SG_INFO_SYS, T_OX, S_NATOMS, T_LATTICE, T_WP, T_ATOMS, T_CN, T_NN, T_EREF, T_EF],
        },
    ),
}

VERSION_ALIASES = {
    "d13_nolemat_ehull": "d13_mixed_lemat_ehull",
}


def get_version(version_id: str) -> VersionSpec:
    if version_id in VERSION_ALIASES:
        version_id = VERSION_ALIASES[version_id]
    if version_id not in VERSIONS:
        raise ValueError(f"Unknown data version: {version_id}")
    return VERSIONS[version_id]

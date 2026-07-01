"""
Compute human-readable property tags for crystal structures.

All functions take simple inputs (element lists, SG number, density, etc.)
that can be parsed from existing d3 source/target strings — no pymatgen
or external API calls needed.

Design principles:
  - Every tag is a single hyphenated token (no spaces): "fcc", "d-block", "high-density"
  - Tags are grouped by category but stored flat (space-separated)
  - New categories can be added later without changing the format
  - Tags that duplicate info already in source segments (crystal_sys, natoms, density)
    are still useful as searchable labels (e.g., "high-density" vs raw "7.3")
"""

from typing import List, Optional
import re

# ---------------------------------------------------------------------------
# Periodic-table data
# ---------------------------------------------------------------------------

ALKALI          = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'}
ALKALINE_EARTH  = {'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'}
TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}
LANTHANIDES = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}
ACTINIDES = {
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
}
POST_TRANSITION = {'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'}
METALLOIDS      = {'B', 'Si', 'Ge', 'As', 'Sb', 'Te'}
HALOGENS        = {'F', 'Cl', 'Br', 'I', 'At'}
NOBLE_GASES     = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'}
NONMETALS       = {'H', 'C', 'N', 'O', 'P', 'S', 'Se'} | HALOGENS

ALL_METALS = (
    ALKALI | ALKALINE_EARTH | TRANSITION_METALS | LANTHANIDES
    | ACTINIDES | POST_TRANSITION
)

# Pauling electronegativity (abridged — covers all common elements)
EN = {
    'H':2.20,'He':0,'Li':0.98,'Be':1.57,'B':2.04,'C':2.55,'N':3.04,'O':3.44,
    'F':3.98,'Ne':0,'Na':0.93,'Mg':1.31,'Al':1.61,'Si':1.90,'P':2.19,'S':2.58,
    'Cl':3.16,'Ar':0,'K':0.82,'Ca':1.00,'Sc':1.36,'Ti':1.54,'V':1.63,'Cr':1.66,
    'Mn':1.55,'Fe':1.83,'Co':1.88,'Ni':1.91,'Cu':1.90,'Zn':1.65,'Ga':1.81,
    'Ge':2.01,'As':2.18,'Se':2.55,'Br':2.96,'Kr':0,'Rb':0.82,'Sr':0.95,'Y':1.22,
    'Zr':1.33,'Nb':1.60,'Mo':2.16,'Tc':2.10,'Ru':2.20,'Rh':2.28,'Pd':2.20,
    'Ag':1.93,'Cd':1.69,'In':1.78,'Sn':1.96,'Sb':2.05,'Te':2.10,'I':2.66,
    'Xe':0,'Cs':0.79,'Ba':0.89,'La':1.10,'Ce':1.12,'Pr':1.13,'Nd':1.14,
    'Pm':1.13,'Sm':1.17,'Eu':1.20,'Gd':1.20,'Tb':1.10,'Dy':1.22,'Ho':1.23,
    'Er':1.24,'Tm':1.25,'Yb':1.10,'Lu':1.27,'Hf':1.30,'Ta':1.50,'W':2.36,
    'Re':1.90,'Os':2.20,'Ir':2.20,'Pt':2.28,'Au':2.54,'Hg':2.00,'Tl':1.62,
    'Pb':2.33,'Bi':2.02,'Po':2.00,'At':2.20,'Rn':0,'Fr':0.70,'Ra':0.90,
    'Ac':1.10,'Th':1.30,'Pa':1.50,'U':1.38,'Np':1.36,'Pu':1.28,
}

# Standard atomic weights (rounded)
ATOMIC_MASS = {
    'H':1,'He':4,'Li':7,'Be':9,'B':11,'C':12,'N':14,'O':16,'F':19,'Ne':20,
    'Na':23,'Mg':24,'Al':27,'Si':28,'P':31,'S':32,'Cl':35,'Ar':40,'K':39,
    'Ca':40,'Sc':45,'Ti':48,'V':51,'Cr':52,'Mn':55,'Fe':56,'Co':59,'Ni':59,
    'Cu':64,'Zn':65,'Ga':70,'Ge':73,'As':75,'Se':79,'Br':80,'Kr':84,'Rb':86,
    'Sr':88,'Y':89,'Zr':91,'Nb':93,'Mo':96,'Tc':98,'Ru':101,'Rh':103,'Pd':106,
    'Ag':108,'Cd':112,'In':115,'Sn':119,'Sb':122,'Te':128,'I':127,'Xe':131,
    'Cs':133,'Ba':137,'La':139,'Ce':140,'Pr':141,'Nd':144,'Pm':145,'Sm':150,
    'Eu':152,'Gd':157,'Tb':159,'Dy':163,'Ho':165,'Er':167,'Tm':169,'Yb':173,
    'Lu':175,'Hf':178,'Ta':181,'W':184,'Re':186,'Os':190,'Ir':192,'Pt':195,
    'Au':197,'Hg':201,'Tl':204,'Pb':207,'Bi':209,'Po':209,'At':210,'Rn':222,
    'Fr':223,'Ra':226,'Ac':227,'Th':232,'Pa':231,'U':238,'Np':237,'Pu':244,
}

# ---------------------------------------------------------------------------
# Bravais lattice  (coarse tags people actually search for)
# ---------------------------------------------------------------------------

def get_bravais_lattice(sg_number: int, sg_symbol: str) -> str:
    """
    Return a coarse Bravais-lattice tag from SG number + Hermann-Mauguin symbol.

    The centering letter is the first character of the SG symbol:
      P → primitive, I → body-centered, F → face-centered,
      C/A/B → base-centered, R → rhombohedral

    Coarse tags:
      fcc           cubic + F
      bcc           cubic + I
      simple-cubic  cubic + P
      hexagonal     hexagonal-P  (we can't distinguish hcp from other hex-P)
      rhombohedral  trigonal + R
      other centering combos → just crystal system name
    """
    centering = sg_symbol[0].upper() if sg_symbol else 'P'

    # Crystal system from SG number
    if 195 <= sg_number <= 230:          # cubic
        if centering == 'F':
            return 'fcc'
        elif centering == 'I':
            return 'bcc'
        else:
            return 'simple-cubic'
    elif 168 <= sg_number <= 194:        # hexagonal
        return 'hexagonal'
    elif 143 <= sg_number <= 167:        # trigonal
        return 'rhombohedral' if centering == 'R' else 'hexagonal'
    else:
        return ''  # not interesting enough to tag


# ---------------------------------------------------------------------------
# Chemical family  (anion-based)
# ---------------------------------------------------------------------------

# Priority order: if O is present, "oxide" wins even if S is also there.
# Multiple families allowed (e.g., oxyfluoride → oxide fluoride).
_ANION_MAP = [
    ('O',  'oxide'),
    ('S',  'sulfide'),
    ('Se', 'selenide'),
    ('Te', 'telluride'),
    ('N',  'nitride'),
    ('P',  'phosphide'),
    ('As', 'arsenide'),
    ('C',  'carbide'),
    ('B',  'boride'),
    ('Si', 'silicide'),
    ('F',  'fluoride'),
    ('Cl', 'chloride'),
    ('Br', 'bromide'),
    ('I',  'iodide'),
    ('H',  'hydride'),
]

def get_chemical_families(elements: List[str]) -> List[str]:
    """Return anion-based family tags.  'intermetallic' if no anions detected."""
    elem_set = set(elements)
    families = []
    for anion, tag in _ANION_MAP:
        if anion in elem_set:
            # Don't tag pure-element structures (e.g., pure C as "carbide")
            if len(elem_set) > 1 or anion in ('O', 'S', 'Se', 'Te', 'N'):
                families.append(tag)
    if not families and len(elements) >= 2 and all(e in ALL_METALS for e in elements):
        families.append('intermetallic')
    elif not families and len(elements) == 1 and elements[0] in ALL_METALS:
        families.append('elemental-metal')
    elif not families and len(elements) == 1 and elements[0] in (NONMETALS | METALLOIDS):
        families.append('elemental-nonmetal')
    return families


# ---------------------------------------------------------------------------
# Element block / family tags
# ---------------------------------------------------------------------------

def get_element_tags(elements: List[str]) -> List[str]:
    """
    Classify the element mixture.  Returns tags like:
      d-block, f-block, s-block, alkali, alkaline-earth, rare-earth, actinide,
      pnictide, chalcogenide
    Only emits the "interesting" ones — avoids noisy tags.
    """
    tags = []
    es = set(elements)

    # Block classification (pick the most exotic block present among cations)
    if es & LANTHANIDES:
        tags.append('rare-earth')
    if es & ACTINIDES:
        tags.append('actinide')
    # d-block omitted: too common (72%) to be a useful filter
    if es & (ALKALI | ALKALINE_EARTH):
        # only tag if it's the *only* metal class (otherwise d-block dominates)
        if not (es & TRANSITION_METALS) and not (es & LANTHANIDES):
            if es & ALKALI:
                tags.append('alkali')
            if es & ALKALINE_EARTH:
                tags.append('alkaline-earth')

    # Metalloid presence
    if es & METALLOIDS:
        tags.append('metalloid')

    return tags


# ---------------------------------------------------------------------------
# Composition complexity
# ---------------------------------------------------------------------------

def get_complexity_tag(n_elements: int) -> str:
    return {1: 'unary', 2: 'binary', 3: 'ternary', 4: 'quaternary'}.get(
        n_elements, 'quinary'
    )


# ---------------------------------------------------------------------------
# Bonding character  (electronegativity spread)
# ---------------------------------------------------------------------------

def get_bonding_character(elements: List[str]) -> str:
    """
    Classify bonding from max electronegativity difference among elements.
      ΔEN < 0.5   → metallic
      > 1.7       → ionic
      (covalent omitted — too common at 65% to be a useful filter)
    """
    vals = [EN[e] for e in elements if e in EN and EN[e] > 0]
    if len(vals) < 2:
        return ''  # can't classify unary
    spread = max(vals) - min(vals)
    if spread < 0.5:
        return 'metallic'
    elif spread > 1.7:
        return 'ionic'
    else:
        return ''  # covalent omitted


# ---------------------------------------------------------------------------
# Density class
# ---------------------------------------------------------------------------

def get_density_class(density: float) -> str:
    if density < 2.0:
        return 'very-low-density'
    elif density < 4.0:
        return 'low-density'
    elif density < 7.0:
        return 'medium-density'
    elif density < 12.0:
        return 'high-density'
    else:
        return 'very-high-density'


# ---------------------------------------------------------------------------
# Average atomic mass class
# ---------------------------------------------------------------------------

def get_mass_class(elements: List[str]) -> str:
    masses = [ATOMIC_MASS.get(e, 0) for e in elements if e in ATOMIC_MASS]
    if not masses:
        return ''
    avg = sum(masses) / len(masses)
    if avg < 20:
        return 'ultralight'
    elif avg < 45:
        return 'light'
    elif avg < 90:
        return 'medium-weight'
    elif avg < 160:
        return 'heavy'
    else:
        return 'very-heavy'


# ===================================================================
# Main entry point
# ===================================================================

def compute_tags(
    sg_number: int,
    sg_symbol: str,
    elements: List[str],
    natoms: int,
    density: float,
) -> List[str]:
    """
    Compute all Tier-0 and Tier-1 property tags for one crystal.

    Args:
        sg_number:  Space-group number (1-230)
        sg_symbol:  Hermann-Mauguin symbol, e.g. "Pm-3m", "I4/mmm"
        elements:   List of unique element symbols, e.g. ["Sr", "Ti", "O"]
        natoms:     Number of atoms in the unit cell
        density:    Density in g/cm³

    Returns:
        Ordered list of tags, e.g.
        ['fcc', 'oxide', 'd-block', 'ionic', 'ternary', 'medium-density', 'medium-weight']
    """
    tags: List[str] = []

    # 1. Bravais lattice (coarse)
    brav = get_bravais_lattice(sg_number, sg_symbol)
    if brav:
        tags.append(brav)

    # 2. Chemical family
    tags.extend(get_chemical_families(elements))

    # 3. Element block / family
    tags.extend(get_element_tags(elements))

    # 4. Bonding character
    bond = get_bonding_character(elements)
    if bond:
        tags.append(bond)

    # 5. Composition complexity
    tags.append(get_complexity_tag(len(elements)))

    # 6. Density class
    tags.append(get_density_class(density))

    # 7. Average atomic mass class
    mc = get_mass_class(elements)
    if mc:
        tags.append(mc)

    # Deduplicate preserving order
    seen = set()
    result = []
    for t in tags:
        if t and t not in seen:
            result.append(t)
            seen.add(t)
    return result


# ===================================================================
# Parsing helpers  (extract inputs from d3 source/target strings)
# ===================================================================

def parse_d3_row(source: str, target: str):
    """
    Parse a d3-format row and return the inputs needed for compute_tags().

    Returns dict with keys: sg_number, sg_symbol, elements, natoms, density,
                            existing_props (list of existing prop_labels, may be empty)
    """
    src_parts = [s.strip() for s in source.split(' | ')]
    tgt_parts = [s.strip() for s in target.split(' | ')]

    # --- From source ---
    # Segment 0: "cubic Pm-3m"  →  sg_symbol
    seg0_tokens = src_parts[0].split()
    sg_symbol = seg0_tokens[-1] if len(seg0_tokens) >= 2 else seg0_tokens[0]

    # Segment 1: "A3B Ac Cr"  →  elements (skip the anonymous formula token)
    seg1_tokens = src_parts[1].split()
    elements = [t for t in seg1_tokens[1:] if re.match(r'^[A-Z][a-z]?$', t)]

    # Segment 2: natoms
    natoms = int(src_parts[2])

    # Segment 3: density
    density = float(src_parts[3])

    # Segments 4+: existing prop_labels (if present)
    existing_props = []
    if len(src_parts) >= 5 and src_parts[4]:
        # Filter out Eref segment
        if not src_parts[4].startswith('Eref'):
            existing_props = src_parts[4].split()

    # --- From target ---
    # Segment 1: "SG 221 Pm-3m"  →  sg_number
    sg_number = None
    if len(tgt_parts) >= 2:
        m = re.search(r'SG\s+(\d+)', tgt_parts[1])
        if m:
            sg_number = int(m.group(1))

    return {
        'sg_number': sg_number,
        'sg_symbol': sg_symbol,
        'elements': elements,
        'natoms': natoms,
        'density': density,
        'existing_props': existing_props,
    }


# ===================================================================
# Self-test
# ===================================================================

if __name__ == '__main__':
    # Quick smoke tests
    tests = [
        (225, 'Fm-3m', ['Na', 'Cl'], 8, 2.16),    # rock salt
        (227, 'Fd-3m', ['Fe', 'O'],  56, 5.18),    # magnetite
        (194, 'P6_3/mmc', ['Ti'],    4, 4.51),     # hcp titanium
        (229, 'Im-3m', ['Fe'],       2, 7.87),     # bcc iron
        (225, 'Fm-3m', ['Cu'],       4, 8.96),     # fcc copper
        (62,  'Pnma',  ['Li','Fe','P','O'], 28, 3.6), # LiFePO4
        (166, 'R-3m',  ['Bi','Te'],  5, 7.7),      # Bi2Te3
        (221, 'Pm-3m', ['Sr','Ti','O'], 5, 5.1),   # SrTiO3
        (12,  'C2/m',  ['Rb','S','Si'], 12, 2.6),  # from OMAT
    ]

    for sg, sym, elems, na, dens in tests:
        tags = compute_tags(sg, sym, elems, na, dens)
        formula = ''.join(elems)
        print(f'{formula:12s} SG{sg:3d} {sym:12s} → {" ".join(tags)}')

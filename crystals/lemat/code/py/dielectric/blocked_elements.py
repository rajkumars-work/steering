"""Single canonical 'bad' element list: elements with no stable isotopes,
or Z >= 84 (radioactive/superheavy). Zero non-stdlib imports on purpose, so
any script (including the lightweight CSV-only steering-pool builders under
cues/nxn/) can import this without pulling in ase/pymatgen/smact/torch.

This is the ONE source of truth for RADIOACTIVE_ELEMENTS, used by:
- validity/structural.py's check_element_filter() (tier0's element_filter_pass)
- dielectric_data/validation.py (training-data cleaning; Alexandria/WBM rows
  with a radioactive element are dropped there, see data/data.md's
  "invalid:radioactive_elements" stat)
- cues/nxn/build_steer_pools_v2.py (steering-pool construction; skips any
  candidate composition that intersects this list, so the model is never
  prompted with something tier0 will unconditionally reject)

History: this list used to exist as up to three independently hand-maintained
copies (validity/structural.py, dielectric_data/validation.py, and a
steering/crystals/lemat bundle copy), which silently drifted from each other
more than once (see GPU_TRAINING_RUNBOOK.md, "Steering pools were requesting
blocked elements" 2026-09-19 and its 2026-09-24 addendum). This module is now
the one hand-edited copy in the private wip repo; the public steering bundle's
copy at steering/crystals/lemat/code/py/dielectric/blocked_elements.py is a
mechanically synced copy of this file (see
dielectric/scripts/sync_steering_bundle.sh) and must never be hand-edited
directly.
"""

RADIOACTIVE_ELEMENTS: frozenset[str] = frozenset([
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",                       # no stable isotopes / Z 84-88
    "Ac", "Th", "Pa", "U", "Np", "Pu",                               # actinides
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",            # transplutonics
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",            # superheavy
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
])

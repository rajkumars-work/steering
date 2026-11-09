"""Screening protocol version — defines the exact checks, thresholds, and defaults.

The screening version is a structured record, not just a string. It captures
every parameter that affects screening results so that:
  1. Two screens with the same version are directly comparable
  2. Any change to checks/thresholds forces a new version
  3. The full configuration is recorded in every screen record

HOW TO UPDATE THE SCREENING VERSION:

  1. Add a new entry to VERSION_HISTORY at the bottom (copy the latest, bump number)
  2. Update the thresholds/checks/defaults that changed
  3. Add a 'changes' list describing what's different
  4. Update SCREENING_VERSION to the new version string
  5. If you changed a threshold in structural.py, make sure the value here matches

  The checks/thresholds in this file are the SOURCE OF TRUTH.
  structural.py functions should read from here when called via the screening
  pipeline (they also have their own defaults for standalone use).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

from blocked_elements import RADIOACTIVE_ELEMENTS


# ---------------------------------------------------------------------------
# Current version
# ---------------------------------------------------------------------------

SCREENING_VERSION = "v7"

# ---------------------------------------------------------------------------
# Check thresholds — the values that affect pass/fail decisions
# ---------------------------------------------------------------------------

CHECK_THRESHOLDS = {
    # Tier 0 (fast, no MLIP)
    "element_filter": {
        # Sourced from blocked_elements.py, not hand-copied (see that
        # module's docstring for the drift history this replaced).
        "blocked_prefixes": sorted(RADIOACTIVE_ELEMENTS),
    },
    "min_dist": {
        "cutoff_A": 0.5,
        "min_volume_A3": 0.1,
    },
    "isolated_atoms": {
        "cutoff_A": 4.0,
    },
    "plausibility": {
        "min_mass_density": 0.01,       # g/cm3
        "max_mass_density": 25.0,       # g/cm3
        "min_atomic_density": 1e-5,     # atoms/A3
        "max_atomic_density": 0.5,      # atoms/A3
        "min_volume": 1.0,              # A3
        "min_lattice": 1.0,             # A (a, b, c)
        "max_lattice": 100.0,           # A
    },
    "smact": {
        "use_pauling_test": True,
        "include_alloys": True,
    },
    "bvs": {
        "max_deviation": 0.5,
    },
    "spacegroup": {
        "symprec": 0.1,
    },
    "composition_fidelity": {
        # Element-set / atom-count vs. the requested prompt. The prompt is a
        # guide, not a hard boundary -- default is NOT strict: a mismatch
        # only sets composition_fidelity_warning=True, it does not block
        # relaxation. Set strict=True to hard-reject on mismatch instead.
        "strict": False,
    },
    # Tier 1 (MLIP uncertainty)
    "uq": {
        "threshold_eV_pa": 0.05,
    },
    # Tier 2 (e-hull)
    "ehull": {
        "threshold_eV_pa": 0.1,
    },
    # Tier 3 (cross-MLIP, soft flag)
    "cross_mlip": {
        "threshold_eV_pa": 0.5,
    },
    # Tier 4 (phonons)
    "phonon": {
        "threshold_thz": -0.5,
    },
    # Tier 5 (MD stability, soft flag)
    "md": {
        "disp_threshold_A": 3.0,
        "vol_drift_pct": 20.0,
        "n_steps": 500,
        "timestep_fs": 2.0,
        "temperature_K": 600,
    },
    # Synthesizability (composite 0-100, run on full screens)
    "synthesizability": {
        "e_hull_hard_limit_eV_pa": 0.15,
        "weights": {
            "Thermodynamic": 0.22,
            "Chemical Plausibility": 0.13,
            "Structural Prototype": 0.25,
            "Kinetic/Precursor": 0.13,
            "ML Synthesizability": 0.12,
            "Evidence": 0.15,
        },
        "uses_precomputed": ["e_above_hull", "smact_pass", "bvs_pass", "bvs_max_deviation"],
    },
    # Evidence layer scoring rubric
    "evidence": {
        "exact_aflow_points": 1.5,
        "literature_synthesis_points": 2.0,
        "literature_no_synthesis_points": 0.3,
        "subsystem_points_each": 0.3,
        "subsystem_max_hits": 3,
        "baseline_points": 0.2,
        "max_raw": 4.0,
    },
    # Set-level
    "uniqueness": {
        "ltol": 0.2,
        "stol": 0.3,
        "angle_tol": 5.0,
    },
    "novelty": {
        "ltol": 0.2,
        "stol": 0.3,
        "angle_tol": 5.0,
    },
    # Per-checker timeouts (seconds)
    "timeouts": {
        "atomic_distance": 5,
        "chemistry": 10,
        "composition_fidelity": 5,
        "novelty": 5,
        "stability_uq": 10,
        "stability_ehull": 60,
        "stability_phonon": 30,
        "stability_cross": 15,
        "stability_md": 30,
        "synthesizability": 30,
    },
}

# ---------------------------------------------------------------------------
# Official screening defaults (generation, relaxation, pipeline)
# ---------------------------------------------------------------------------

OFFICIAL_DEFAULTS = {
    # Generation
    "n": 1000,
    "seed": 42,
    "max_length": 512,
    "top_k": 10,
    "gen_timeout": 20,

    # Relaxation
    "fmax": 0.05,
    "relax_max_steps": 200,
    "relax_timeout": 60,
    "relax_model": None,  # MACE default (mpa-0)

    # Screening
    "full": True,
    "funnel": True,
    "screen_timeout": 120,

    # Set-level
    "compute_uniqueness": True,
    "compute_novelty": True,
}

# ---------------------------------------------------------------------------
# Check lists
# ---------------------------------------------------------------------------

TIER0_CHECKS = [
    "element_filter_pass",
    "min_dist_pass",
    "isolated_atoms_pass",
    "plausibility_pass",
    "smact_pass",
    "bvs_pass",
    "spacegroup_pass",
    "composition_fidelity_pass",
]

MLIP_CHECKS = [
    "uq_pass",
    "ehull_pass",
    "phonon_pass",
    "cross_pass",
]

SUMMARY_METRICS = [
    "generation_success_rate",
    "syntactic_correctness",
    "relaxation_convergence_rate",
    *TIER0_CHECKS,
    *MLIP_CHECKS,
    "overall_pass_rate",
    "uniqueness",
    "novelty",
    "sun_score",
    "mean_synth_score",
]

# ---------------------------------------------------------------------------
# Version fingerprint — deterministic hash of the full config
# ---------------------------------------------------------------------------

def compute_fingerprint() -> str:
    """SHA256 hash of thresholds + defaults. Changes when any parameter changes."""
    blob = json.dumps({
        "thresholds": CHECK_THRESHOLDS,
        "defaults": OFFICIAL_DEFAULTS,
        "tier0": TIER0_CHECKS,
        "mlip": MLIP_CHECKS,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def get_version_record() -> dict:
    """Return the full version record for embedding in screen results."""
    return {
        "version": SCREENING_VERSION,
        "fingerprint": compute_fingerprint(),
        "thresholds": CHECK_THRESHOLDS,
        "defaults": OFFICIAL_DEFAULTS,
        "tier0_checks": TIER0_CHECKS,
        "mlip_checks": MLIP_CHECKS,
    }


# ---------------------------------------------------------------------------
# Version history — append-only log of all versions
# ---------------------------------------------------------------------------

VERSION_HISTORY: List[Dict] = [
    {
        "version": "v1",
        "date": "2026-03-20",
        "description": "Initial frozen protocol",
        "changes": ["Initial version — all thresholds from structural.py defaults"],
        "fingerprint": "9dd24f39bfd7",
    },
    {
        "version": "v2",
        "date": "2026-03-21",
        "description": "Added synthesizability score to full screening",
        "changes": [
            "Added SynthesizabilityScorer (5-layer composite: thermo, chemical, structural, kinetic, ML)",
            "Full screens now compute per-structure synth_score (0-100) and synth_verdict",
            "Mean synth score aggregated in screening summary",
            "Added synthesizability thresholds to CHECK_THRESHOLDS",
            "Added mean_synth_score to SUMMARY_METRICS",
        ],
        "fingerprint": "b290f13d17f4",
    },
    {
        "version": "v3",
        "date": "2026-03-21",
        "description": "Added Evidence layer (literature + database search) to synthesizability",
        "changes": [
            "Added Layer 6 (Evidence): searches Semantic Scholar, CrossRef, AFLOW for exact/similar compositions",
            "Classifies papers as experimental vs theoretical using synthesis keyword matching",
            "Checks subsystem existence in AFLOW for novel compositions",
            "Redistributed layer weights: Prototype 0.25, Evidence 0.15, Thermo 0.22, Chemical 0.13, Kinetic 0.13, ML 0.12",
            "Added evidence scoring rubric to CHECK_THRESHOLDS",
            "Added --skip-evidence CLI flag for offline use",
            "Added evidence_check() standalone function",
            "Results cached per formula to avoid repeated API calls",
        ],
        "fingerprint": None,
    },
    {
        "version": "v4",
        "date": "2026-03-23",
        "description": "Added --no-funnel, tiered novelty, per-check error isolation; phonon/cross disabled",
        "changes": [
            "Added --no-funnel flag: runs ALL checks on every structure (no early termination)",
            "Added tiered novelty classification: copy/polymorph/variant/novel per structure",
            "Per-check error isolation in run_checks(): MLIP check failure no longer wipes tier-0 results",
            "KNOWN ISSUE: phonon_pass and cross_pass fail with torch.compile error "
            "('Linear' object has no attribute '_compiled_main') on MACE+ASE Vibrations. "
            "These checks are excluded from overall_pass until the torch.compile/MACE "
            "interaction is resolved. Tracked in BUGS.md.",
        ],
        "fingerprint": None,
    },
    {
        "version": "v5",
        "date": "2026-03-23",
        "description": "Refactored screening into composable checker classes with per-check timeouts",
        "changes": [
            "New validity/checkers.py: AtomicDistanceChecker, ChemistryChecker, StabilityChecker, "
            "NoveltyChecker, SynthesizabilityChecker, ScreeningPipeline, NoveltyIndex",
            "Per-checker and per-sub-check timeouts via ThreadPoolExecutor (replaces global SIGALRM)",
            "NoveltyIndex cached to disk (data/.novelty_cache/) — rebuild only when CSV changes",
            "run_checks() delegates to ScreeningPipeline internally (backward compatible)",
            "Added per-checker timeout configuration to CHECK_THRESHOLDS",
            "Timing instrumentation: {checker_name}_time_ms in results",
        ],
        "fingerprint": None,
    },
    {
        "version": "v6",
        "date": "2026-08-06",
        "description": "Ported genbench benchmark pipeline + validity package; added plausibility checks",
        "changes": [
            "Added check_plausibility (density/volume/lattice bounds) to AtomicDistanceChecker",
            "Added 'plausibility' thresholds to CHECK_THRESHOLDS (mass/atomic density bounds, "
            "volume/lattice bounds)",
            "Added 'plausibility_pass' to TIER0_CHECKS",
            "(Backfilled retroactively on 2026-08-28 -- this entry was missing even though "
            "SCREENING_VERSION had already been bumped to v6; fingerprint not recoverable "
            "for a past version, left as None like v3/v4.)",
        ],
        "fingerprint": None,
    },
    {
        "version": "v7",
        "date": "2026-08-28",
        "description": "Added CompositionFidelityChecker: strict/not-strict element-set + atom-count fidelity check",
        "changes": [
            "New CompositionFidelityChecker in validity/checkers.py: reuses composition_fidelity() "
            "from eval/screening.py to compare a generated structure's actual element set / atom "
            "count against its prompt",
            "Default is NOT strict: a mismatch sets composition_fidelity_warning=True but "
            "composition_fidelity_pass=True (does not block relaxation) -- the prompt is a guide, "
            "not a hard boundary, and generating novel compositions/counts is the point",
            "Set CHECK_THRESHOLDS['composition_fidelity']['strict']=True to instead hard-reject on "
            "mismatch (added to hard_reject_keys, gates overall_pass/tier0_pass like smact_pass/bvs_pass)",
            "Added 'composition_fidelity' threshold entry (strict flag) and its timeout to CHECK_THRESHOLDS",
            "Added 'composition_fidelity_pass' to TIER0_CHECKS (always True when not strict, so it "
            "never blocks existing non-strict pipelines/summary stats)",
            "tier0_one() now includes CompositionFidelityChecker in its pipeline",
            "Wired cues/nxn/stability_relax_compare.py and stability_relax_compare_resume.py to call "
            "tier0_one() before relax_structure(), skipping the expensive pre-ehull/relax steps on "
            "tier0 failure (previously these ad-hoc scripts bypassed tier0 screening entirely)",
            "(Ported into dielectric/'s own copy of this file on 2026-09-24, consolidating what had "
            "been a bundle-only v4-v7 -- see blocked_elements.py and GPU_TRAINING_RUNBOOK.md's "
            "2026-09-24 entries for the broader single-source-of-truth cleanup this was part of.)",
        ],
        "fingerprint": None,  # populated below
    },
]

# Populate fingerprint for current version
VERSION_HISTORY[-1]["fingerprint"] = compute_fingerprint()


def validate_version() -> bool:
    """Check that the current fingerprint matches the declared version.

    Returns True if consistent. If False, someone changed a threshold
    without bumping the version — this is a bug.
    """
    current_fp = compute_fingerprint()
    latest_history = VERSION_HISTORY[-1]
    if latest_history["version"] != SCREENING_VERSION:
        return False
    if latest_history["fingerprint"] != current_fp:
        return False
    return True

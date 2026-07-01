"""Conditional-generation probes + prompt-format sanity checks.

Two purposes:

1. **Prompt-format sanity** — `validate_prompt` and `build_prompt` are the
   canonical way to construct eval prompts. Historically we had bugs where
   hand-built prompts went out-of-distribution (raw floats for density, wrong
   segment order, missing pipes). Always go through `build_prompt` and assert
   `validate_prompt` returns True.

2. **Conditional-generation probe** — `run_cond_probe` generates a small fixed
   set of prompts across the (bandgap × eps_0) bin grid, predicts their bins
   with nequip MLIPs, and returns per-bin compliance fractions. Designed to
   be called periodically during training to catch silent regressions in
   conditional generation.

Usage standalone:
    from cond_probe import run_cond_probe
    result = run_cond_probe(model, sp, device, version_id="d16_ic")

Usage from training:
    See ed_train.py — called every `cond_probe_interval` epochs when
    `--cond_probe_interval > 0`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Auto-bin vocabulary — loaded lazily from the persisted sidecar (see
# chem.auto_bin.DEFAULT_BINNING_SIDECAR). The legacy bin tuples below are
# kept only as a fallback for old models trained on the pre-autobin
# vocabulary — they match `FieldFormatter.bin_property` for backward compat.
_LEGACY_EPS0_BINS = [
    ("very-low-k", 0.0, 3.0),
    ("low-k",      3.0, 5.0),
    ("medium-k",   5.0, 10.0),
    ("high-k",     10.0, 20.0),
    ("very-high-k", 20.0, 1e6),
]
_LEGACY_BG_BINS = [
    ("metal",         0.0,  0.1),
    ("narrow-gap",    0.1,  0.6),
    ("small-gap",     0.6,  1.5),
    ("semiconductor", 1.5,  3.0),
    ("wide-gap",      3.0,  4.0),
    ("very-wide-gap", 4.0,  1e6),
]


_binnings_cache: dict | None = None


def _get_binnings():
    """Lazy-load the auto-bin sidecar. Returns None if unavailable (then
    callers fall back to legacy bins)."""
    global _binnings_cache
    if _binnings_cache is None:
        try:
            import sys as _sys
            _sys.path.insert(0, "/data/rkumar/code/py/dielectric")
            from chem.auto_bin import load_binnings
            _binnings_cache = load_binnings()
        except Exception:
            _binnings_cache = {}
    return _binnings_cache


def _known_label_tokens() -> set[str]:
    """Return the full set of bin-token strings the model may have been
    trained on. Pulled from the auto-bin sidecar when available; otherwise
    falls back to the legacy hard-coded vocabulary.
    """
    toks: set[str] = set()
    binnings = _get_binnings()
    if binnings:
        for b in binnings.values():
            toks.update(b.labels)
    else:
        toks.update(name for name, _, _ in _LEGACY_EPS0_BINS)
        toks.update(name for name, _, _ in _LEGACY_BG_BINS)
        toks.update({"stable", "metastable", "unstable"})
    # Crystal system + interconnect labels are dataset-wide constants
    toks.update({
        "cubic", "tetragonal", "orthorhombic", "hexagonal",
        "trigonal", "monoclinic", "triclinic",
        "ductile", "brittle", "high-debye", "medium-debye", "low-debye",
        "topological-semimetal", "max-phase",
        "sg-109", "sg-129", "sg-187", "sg-194", "sg-198",
    })
    return toks

# ---------------------------------------------------------------------------
# Prompt format: canonical builder + validator
# ---------------------------------------------------------------------------

# Canonical prompt shape: "<elements> | <natoms> | <density> | <labels>"
# Where:
#   - elements: space-separated element symbols, sorted
#   - natoms:   integer
#   - density:  one-decimal string from f"{density:.1f}"  (see feedback_density_formatting)
#   - labels:   space-separated label tokens from the known bin/category lists
_DENSITY_RE = re.compile(r"^\d+\.\d$")  # strict "X.Y" one decimal
_NATOMS_RE = re.compile(r"^\d+$")
_ELEMENTS_RE = re.compile(r"^([A-Z][a-z]?)(?:\s+[A-Z][a-z]?)*$")


def build_prompt(elements, natoms: int, density: float,
                 labels: list[str] | None = None) -> str:
    """Canonical prompt builder. Use this everywhere we feed RECAST."""
    if isinstance(elements, str):
        elems_str = " ".join(sorted(elements.split()))
    else:
        elems_str = " ".join(sorted(elements))
    natoms_str = str(int(natoms))
    density_str = f"{float(density):.1f}"
    labels_str = " ".join(labels) if labels else ""
    return f"{elems_str} | {natoms_str} | {density_str} | {labels_str}"


def validate_prompt(prompt: str, require_labels: bool = False) -> tuple[bool, str]:
    """Return (ok, reason). Defensive format check for eval prompts.

    Catches: wrong segment count, density without .1 decimal (historical bug),
    missing natoms, bad element casing, unknown label tokens.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return False, "empty"
    parts = [s.strip() for s in prompt.split("|")]
    if len(parts) != 4:
        return False, f"expected 4 pipe-separated segments, got {len(parts)}"
    elems, natoms, density, labels = parts

    if not _ELEMENTS_RE.match(elems):
        return False, f"elements segment malformed: {elems!r}"

    if not _NATOMS_RE.match(natoms):
        return False, f"natoms segment not an integer: {natoms!r}"

    if not _DENSITY_RE.match(density):
        return False, (
            f"density must be single-decimal 'X.Y' (e.g., '3.2'); "
            f"got {density!r} — raw floats break the model "
            "(see feedback_density_formatting)"
        )

    if require_labels and not labels:
        return False, "labels segment is empty but labels required"

    if labels:
        known = _known_label_tokens()
        unknown = [t for t in labels.split() if t not in known]
        if unknown:
            return False, f"unknown label tokens: {unknown}"

    return True, "ok"


def bin_of(value: float, bins) -> Optional[str]:
    """Legacy helper: bin a value using (name, lo, hi) tuples. Still used by
    callers that hand-roll bins. New callers should prefer
    chem.auto_bin.label_value(value, Binning)."""
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return None


# ---------------------------------------------------------------------------
# Probe set — kept tiny for in-training use
# ---------------------------------------------------------------------------

# 15 prompts: 3 per bg bin spanning the eps_0 range. Uses light-element
# compositions that are physically compatible with the target bins so the
# probe tests conditioning (not composition-override — see
# docs/AutoBinCompliance.md). Labels follow the auto-bin vocabulary.
# Stability label defaults to "hull-low" (≈ on-hull / stable).
_PROBE_PROMPTS: list[tuple[str, str, str]] = [
    # (src_prompt, prompted_bg_bin, prompted_eps_bin)
    # bg-vhigh (>2 eV): oxides + light-element covalents
    ("Si O    | 12 | 2.6 | bg-vhigh hull-low k-vlow",   "bg-vhigh", "k-vlow"),
    ("Al O    | 10 | 3.9 | bg-vhigh hull-low k-low",    "bg-vhigh", "k-low"),
    ("B N     |  4 | 3.5 | bg-vhigh hull-low k-vlow",   "bg-vhigh", "k-vlow"),

    # bg-high (1-2 eV): semiconductors
    ("Ga As   |  8 | 5.3 | bg-high hull-low k-mid",     "bg-high",  "k-mid"),
    ("In P    |  8 | 4.8 | bg-high hull-low k-mid",     "bg-high",  "k-mid"),
    ("Cd Te   |  8 | 6.0 | bg-high hull-low k-mid",     "bg-high",  "k-mid"),

    # bg-mid (0.5-1 eV): narrow-gap semiconductors
    ("Ge Si   |  8 | 4.7 | bg-mid hull-low k-high",     "bg-mid",   "k-high"),
    ("Pb S    |  8 | 7.6 | bg-mid hull-low k-high",     "bg-mid",   "k-high"),
    ("In Sb   |  8 | 5.8 | bg-mid hull-low k-high",     "bg-mid",   "k-high"),

    # bg-low + bg-vlow: dielectric perovskites and metals
    ("Ba Ti O |  5 | 5.9 | bg-low hull-low k-vhigh",    "bg-low",   "k-vhigh"),
    ("Pb Zr O |  5 | 7.9 | bg-low hull-low k-vhigh",    "bg-low",   "k-vhigh"),
    ("Sr Ti O |  5 | 5.1 | bg-low hull-low k-high",     "bg-low",   "k-high"),

    ("Fe As   |  4 | 7.4 | bg-vlow hull-low k-high",    "bg-vlow",  "k-high"),
    ("Na W    |  4 | 6.6 | bg-vlow hull-low k-mid",     "bg-vlow",  "k-mid"),
    ("Mo S    |  6 | 5.1 | bg-vlow hull-low k-mid",     "bg-vlow",  "k-mid"),
]


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    n_total: int
    n_parsed: int
    n_eps_match: int
    n_bg_match: int
    per_prompt: list[dict]

    def summary(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_parsed": self.n_parsed,
            "parse_rate": round(self.n_parsed / max(self.n_total, 1), 3),
            "eps_match_rate": round(self.n_eps_match / max(self.n_parsed, 1), 3),
            "bg_match_rate": round(self.n_bg_match / max(self.n_parsed, 1), 3),
        }


def run_cond_probe(
    model,
    sp,
    device,
    version_id: str = "d16_ic",
    origin: str = "mp",
    max_length: int = 768,
    top_k: int = 10,
    prompts: list[tuple[str, str, str]] | None = None,
    eps_calc=None,
    bg_calc=None,
) -> ProbeResult:
    """Generate prompt → parse to Atoms → predict eps/bg → compare to prompt bin.

    If `eps_calc` / `bg_calc` are None, loads the nequip MLIPs (adds ~3s
    cold-start). Pass them in to keep the probe fast when running repeatedly.
    """
    import torch  # local import so test file can import module without torch
    from ed_train import generate
    from dielectric_data.reader import parse_target, robust_fuzzy_parse

    if prompts is None:
        prompts = _PROBE_PROMPTS

    # Sanity-check every prompt up front
    for src, _, _ in prompts:
        ok, why = validate_prompt(src, require_labels=True)
        assert ok, f"probe prompt malformed: {src!r} — {why}"

    # Resolve Binning objects for compliance scoring. When the sidecar is
    # available we use the auto-bin labels; otherwise we fall back to legacy
    # (name, lo, hi) tuples.
    binnings = _get_binnings()
    bg_binning = binnings.get("band_gap") if binnings else None
    eps_binning = binnings.get("nequip_eps_0") if binnings else None

    # Load MLIPs once if not supplied
    if eps_calc is None or bg_calc is None:
        import sys as _sys
        _sys.path.insert(0, "/data/rkumar/code/py/dielectric/scripts")
        from validate_nequip_mlips import load_calc
        EPS_CKPT = "/data/assets/checkpoints/nequip/nequip-eps-log-lr1e3-bs8-plateau-last.nequip.zip"
        BG_CKPT = "/data/assets/checkpoints/nequip/nequip-bandgap-softplus-lr1e3-bs5-last-best-FINAL.nequip.zip"
        device_str = "cuda" if str(device).startswith("cuda") else str(device)
        if eps_calc is None:
            eps_calc = load_calc(EPS_CKPT, device=device_str)
        if bg_calc is None:
            bg_calc = load_calc(BG_CKPT, device=device_str)

    per_prompt = []
    n_parsed = 0; n_eps = 0; n_bg = 0

    was_training = model.training
    model.eval()
    try:
        for src, bg_target, eps_target in prompts:
            entry = {"source": src, "prompted_bg_bin": bg_target,
                     "prompted_eps_bin": eps_target,
                     "parse_ok": False}
            try:
                tgt_text = generate(model, sp, src, device,
                                    max_length=max_length, top_k=top_k)
                atoms = parse_target(tgt_text, version_id, origin)
                if atoms is None:
                    atoms = robust_fuzzy_parse(tgt_text)
                if atoms is None or len(atoms) == 0:
                    per_prompt.append(entry); continue

                n_parsed += 1
                entry["parse_ok"] = True
                entry["formula"] = atoms.get_chemical_formula(mode="reduce")
                entry["natoms"] = len(atoms)

                # MLIP predictions
                a2 = atoms.copy(); a2.calc = eps_calc
                eps_pred = float(a2.get_potential_energy())
                a3 = atoms.copy(); a3.calc = bg_calc
                bg_pred = float(a3.get_potential_energy())

                entry["eps_pred"] = eps_pred
                entry["bg_pred"] = bg_pred
                if eps_binning is not None:
                    from chem.auto_bin import label_value
                    entry["pred_eps_bin"] = label_value(eps_pred, eps_binning)
                    entry["pred_bg_bin"] = label_value(bg_pred, bg_binning)
                else:
                    entry["pred_eps_bin"] = bin_of(eps_pred, _LEGACY_EPS0_BINS)
                    entry["pred_bg_bin"] = bin_of(bg_pred, _LEGACY_BG_BINS)
                if entry["pred_eps_bin"] == eps_target: n_eps += 1
                if entry["pred_bg_bin"]  == bg_target:  n_bg += 1

            except Exception as e:
                entry["error"] = str(e)[:120]

            per_prompt.append(entry)
    finally:
        if was_training:
            model.train()

    return ProbeResult(
        n_total=len(prompts),
        n_parsed=n_parsed,
        n_eps_match=n_eps,
        n_bg_match=n_bg,
        per_prompt=per_prompt,
    )


def save_probe_result(result: ProbeResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": result.summary(),
        "per_prompt": result.per_prompt,
    }, indent=2, default=str))

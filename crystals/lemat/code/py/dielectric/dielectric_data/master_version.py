"""Master IDs — single gatekeeper for all reads and writes.

ALL mutations to master_ids.csv MUST go through this module.  The write
functions (`append_rows`, `rewrite`) handle both the CSV I/O and the
version bump atomically, so the sidecar version file can never drift.

Reading:
    rows = read_master(path)              # list[dict]
    info = get_version(path)              # version metadata

Writing:
    append_rows(path, new_rows, reason)   # add rows, bump version
    rewrite(path, rows, reason, ...)      # full rewrite, bump version

Version file: master_ids.version.json (next to master_ids.csv)
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

FIELDNAMES = ["id", "origin", "label", "flag"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _version_path(master_ids_path: str | Path) -> Path:
    return Path(master_ids_path).with_suffix(".version.json")


def _count_rows(master_ids_path: str | Path) -> int:
    p = Path(master_ids_path)
    if not p.exists():
        return 0
    with open(p, "r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # subtract header


def _checksum(master_ids_path: str | Path, sample_bytes: int = 65536) -> str:
    """Fast partial checksum: hash first + last sample_bytes."""
    p = Path(master_ids_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes * 2:
            f.seek(-sample_bytes, 2)
            h.update(f.read(sample_bytes))
    return h.hexdigest()[:16]


def _bump(master_ids_path: str | Path, reason: str,
          rows_added: int = 0, rows_modified: int = 0) -> int:
    """Increment version number, update sidecar. Returns new version."""
    vp = _version_path(master_ids_path)
    if vp.exists():
        data = json.loads(vp.read_text(encoding="utf-8"))
    else:
        data = {"version": 0, "changelog": []}

    new_version = data["version"] + 1
    row_count = _count_rows(master_ids_path)
    cksum = _checksum(master_ids_path)

    entry = {
        "version": new_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": reason,
        "rows_added": rows_added,
        "rows_modified": rows_modified,
        "total_rows": row_count,
        "checksum": cksum,
    }

    data["version"] = new_version
    data["total_rows"] = row_count
    data["checksum"] = cksum
    data["last_updated"] = entry["timestamp"]
    data["changelog"].append(entry)

    vp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return new_version


# ---------------------------------------------------------------------------
# Public API — reads
# ---------------------------------------------------------------------------

def get_version(master_ids_path: str | Path) -> dict:
    """Read current version metadata. Returns empty dict if unversioned."""
    vp = _version_path(master_ids_path)
    if not vp.exists():
        return {}
    return json.loads(vp.read_text(encoding="utf-8"))


def read_master(master_ids_path: str | Path) -> List[dict]:
    """Read all rows from master_ids.csv. Returns list of dicts."""
    p = Path(master_ids_path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Public API — writes (gatekeeper: CSV I/O + version bump, atomic)
# ---------------------------------------------------------------------------

def append_rows(
    master_ids_path: str | Path,
    new_rows: List[tuple],
    reason: str,
) -> int:
    """Append new (id, origin, label) rows to master_ids.csv and bump version.

    Args:
        master_ids_path: Path to master_ids.csv
        new_rows: List of (id, origin, label) tuples to append
        reason: Changelog description

    Returns:
        New version number.
    """
    if not new_rows:
        return get_version(master_ids_path).get("version", 0)

    p = Path(master_ids_path)
    write_header = not p.exists()

    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FIELDNAMES)
        for rid, origin, label in new_rows:
            writer.writerow([rid, origin, label, ""])

    return _bump(p, reason=reason, rows_added=len(new_rows))


def rewrite(
    master_ids_path: str | Path,
    rows: List[dict],
    reason: str,
    rows_modified: int = 0,
) -> int:
    """Rewrite master_ids.csv entirely from row dicts and bump version.

    Args:
        master_ids_path: Path to master_ids.csv
        rows: Complete list of row dicts (must have id/origin/label/flag keys)
        reason: Changelog description
        rows_modified: Number of rows that were changed (for the changelog)

    Returns:
        New version number.
    """
    p = Path(master_ids_path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row.get("id", ""),
                "origin": row.get("origin", ""),
                "label": row.get("label", ""),
                "flag": row.get("flag", ""),
            })

    return _bump(p, reason=reason, rows_modified=rows_modified)


# ---------------------------------------------------------------------------
# One-time initialization
# ---------------------------------------------------------------------------

def init_version(master_ids_path: str | Path) -> int:
    """Initialize versioning for an existing master_ids.csv (version 1)."""
    vp = _version_path(master_ids_path)
    if vp.exists():
        existing = get_version(master_ids_path)
        print(f"Version file already exists (v{existing.get('version', '?')})")
        return existing.get("version", 0)

    row_count = _count_rows(master_ids_path)
    cksum = _checksum(master_ids_path)

    data = {
        "version": 1,
        "total_rows": row_count,
        "checksum": cksum,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "changelog": [{
            "version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": "initial version (existing file)",
            "rows_added": row_count,
            "rows_modified": 0,
            "total_rows": row_count,
            "checksum": cksum,
        }],
    }

    vp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Initialized master_ids version: v1 ({row_count:,} rows, checksum={cksum})")
    return 1

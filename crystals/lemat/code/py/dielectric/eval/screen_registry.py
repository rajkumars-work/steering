"""Screening run registry — stores screening parameters, results, and cross-references.

All screening records are stored as JSON in eval/screens/ with an index.json manifest.

Usage:
    reg = ScreenRecord.start(...)
    reg.set_generation_summary(...)
    reg.set_screening_summary(...)
    reg.finalize(...)
    reg.save()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SCREENS_DIR = Path(__file__).resolve().parent / "screens"


def _screen_id() -> str:
    """Generate a unique screen ID. Appends PID suffix to avoid collisions
    when multiple screens launch in the same second."""
    pid_suffix = f"_{os.getpid() % 10000:04d}"
    return time.strftime("%Y%m%d_%H%M%S") + pid_suffix


class ScreenRecord:
    """Tracks a single screening run."""

    def __init__(self):
        self.screen_id: str = ""
        self.data: dict = {}
        self._start_time: float = 0.0

    @classmethod
    def start(
        cls,
        official: bool,
        screening_version: str,
        ckpt_path: str,
        data_csv: str,
        params: dict,
        training_run_id: Optional[str] = None,
        data_audit_id: Optional[str] = None,
        screening_fingerprint: Optional[str] = None,
    ) -> "ScreenRecord":
        rec = cls()
        rec.screen_id = _screen_id()
        rec._start_time = time.time()

        rec.data = {
            "screen_id": rec.screen_id,
            "official": official,
            "screening_version": screening_version,
            "screening_fingerprint": screening_fingerprint,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished_at": None,
            "duration_sec": None,

            # Cross-references
            "training_run_id": training_run_id,
            "data_audit_id": data_audit_id,

            # Inputs
            "checkpoint": str(ckpt_path),
            "data_csv": str(data_csv),

            # Frozen parameters
            "parameters": params,

            # Populated during/after screening
            "generation_summary": {},
            "relaxation_summary": {},
            "screening_summary": {},
            "set_metrics": {},
            "final_status": {},
        }
        return rec

    def set_generation_summary(self, **kwargs):
        self.data["generation_summary"].update(kwargs)

    def set_relaxation_summary(self, **kwargs):
        self.data["relaxation_summary"].update(kwargs)

    def set_screening_summary(self, **kwargs):
        self.data["screening_summary"].update(kwargs)

    def set_set_metrics(self, **kwargs):
        self.data["set_metrics"].update(kwargs)

    def finalize(self, status: str = "completed", outdir: Optional[str] = None):
        self.data["status"] = status
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["duration_sec"] = round(time.time() - self._start_time, 1)
        if outdir:
            self.data["outdir"] = str(outdir)

    def save(self) -> Path:
        SCREENS_DIR.mkdir(parents=True, exist_ok=True)
        tag = "official" if self.data["official"] else "unofficial"
        filename = f"screen_{tag}_{self.screen_id}.json"
        out = SCREENS_DIR / filename
        out.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        _update_index(out, self.data)
        return out

    def save_interrupted(self, error: Optional[str] = None):
        self.data["status"] = "interrupted"
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if hasattr(self, "_start_time") and self._start_time:
            self.data["duration_sec"] = round(time.time() - self._start_time, 1)
        if error:
            self.data["final_status"]["error"] = error
        try:
            self.save()
        except Exception:
            pass


def _update_index(screen_path: Path, data: dict) -> None:
    index_path = SCREENS_DIR / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = []

    entry = {
        "file": screen_path.name,
        "screen_id": data["screen_id"],
        "official": data["official"],
        "version": data["screening_version"],
        "status": data["status"],
        "checkpoint": Path(data["checkpoint"]).name,
        "data": Path(data["data_csv"]).name,
        "training_run_id": data.get("training_run_id"),
        "data_audit_id": data.get("data_audit_id"),
        "overall_pass_rate": data.get("screening_summary", {}).get("overall_pass_rate"),
        "uniqueness": data.get("set_metrics", {}).get("uniqueness"),
        "novelty": data.get("set_metrics", {}).get("novelty"),
        "started_at": data["started_at"],
        "duration_sec": data.get("duration_sec"),
    }

    for i, e in enumerate(index):
        if e["screen_id"] == data["screen_id"]:
            index[i] = entry
            break
    else:
        index.append(entry)

    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def list_screens() -> list:
    index_path = SCREENS_DIR / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_screen(screen_id_or_file: str) -> dict:
    p = SCREENS_DIR / screen_id_or_file
    if not p.exists():
        # Try with prefix patterns
        for prefix in ("screen_official_", "screen_unofficial_"):
            alt = SCREENS_DIR / f"{prefix}{screen_id_or_file}.json"
            if alt.exists():
                p = alt
                break
    if not p.exists():
        raise FileNotFoundError(f"Screen not found: {screen_id_or_file}")
    return json.loads(p.read_text(encoding="utf-8"))


def print_screen_table(screens: Optional[list] = None):
    if screens is None:
        screens = list_screens()
    if not screens:
        print("No screening runs recorded.")
        return

    print(f"{'Screen ID':<17} {'Off?':<5} {'Ver':<4} {'Status':<12} {'Checkpoint':<30} "
          f"{'Pass%':>6} {'Uniq%':>6} {'Novel%':>7} {'Duration':>10}")
    print("─" * 110)
    for r in screens:
        dur = r.get("duration_sec")
        dur_str = f"{dur/3600:.1f}h" if dur and dur > 3600 else f"{dur:.0f}s" if dur else "-"
        opr = r.get("overall_pass_rate")
        uniq = r.get("uniqueness")
        nov = r.get("novelty")
        tag = "Y" if r.get("official") else "N"
        print(f"{r['screen_id']:<17} {tag:<5} {r.get('version', '?'):<4} {r['status']:<12} "
              f"{r.get('checkpoint', '')[:30]:<30} "
              f"{f'{opr:.1f}' if opr is not None else '-':>6} "
              f"{f'{uniq:.1f}' if uniq is not None else '-':>6} "
              f"{f'{nov:.1f}' if nov is not None else '-':>7} "
              f"{dur_str:>10}")


def compare_screens(*screen_ids: str):
    screens = [load_screen(sid) for sid in screen_ids]
    labels = [s["screen_id"] for s in screens]
    col_w = max(20, max(len(l) for l in labels) + 2)

    print(f"\n{'='*70}")
    print("SCREENING RUN COMPARISON")
    print(f"{'='*70}")

    # Parameters
    all_param_keys = set()
    for s in screens:
        all_param_keys.update(s.get("parameters", {}).keys())

    print("\nParameters:")
    header = "".ljust(25) + "".join(l.rjust(col_w) for l in labels)
    print(f"  {header}")
    for key in sorted(all_param_keys):
        vals = [str(s.get("parameters", {}).get(key, "-")) for s in screens]
        all_same = len(set(vals)) == 1
        marker = " " if all_same else "*"
        print(f"  {marker}{key:<24}" + "".join(v.rjust(col_w) for v in vals))

    # Cross-references
    print("\nCross-references:")
    for key in ("training_run_id", "data_audit_id", "screening_version", "official"):
        vals = [str(s.get(key, "-")) for s in screens]
        print(f"  {key:<24}" + "".join(v.rjust(col_w) for v in vals))

    # Generation summary
    print("\nGeneration:")
    gen_keys = set()
    for s in screens:
        gen_keys.update(s.get("generation_summary", {}).keys())
    for key in sorted(gen_keys):
        vals = [str(s.get("generation_summary", {}).get(key, "-")) for s in screens]
        print(f"  {key:<24}" + "".join(v.rjust(col_w) for v in vals))

    # Screening summary
    print("\nScreening:")
    scr_keys = set()
    for s in screens:
        scr_keys.update(s.get("screening_summary", {}).keys())
    for key in sorted(scr_keys):
        vals = [str(s.get("screening_summary", {}).get(key, "-")) for s in screens]
        print(f"  {key:<24}" + "".join(v.rjust(col_w) for v in vals))

    # Set metrics
    print("\nSet Metrics:")
    met_keys = set()
    for s in screens:
        met_keys.update(s.get("set_metrics", {}).keys())
    for key in sorted(met_keys):
        vals = [str(s.get("set_metrics", {}).get(key, "-")) for s in screens]
        print(f"  {key:<24}" + "".join(v.rjust(col_w) for v in vals))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Screening run registry")
    parser.add_argument("--list", action="store_true", help="List all screening runs")
    parser.add_argument("--show", metavar="ID", help="Show details of a screening run")
    parser.add_argument("--compare", nargs="+", metavar="ID", help="Compare screening runs")
    args = parser.parse_args()

    if args.list:
        print_screen_table()
    elif args.show:
        s = load_screen(args.show)
        print(json.dumps(s, indent=2))
    elif args.compare:
        compare_screens(*args.compare)
    else:
        parser.print_help()

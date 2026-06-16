"""Training run registry — stores hyperparameters, data info, loss curves, and final status.

All runs are stored as JSON files in data/runs/ with an index.json manifest.

Usage from ed_train.py:
    reg = RunRecord.start(args, config, extra={...})
    # during training:
    reg.log_train(step, loss, norm, lr)
    reg.log_eval(step, eval_loss)
    reg.log_checkpoint(step, path)
    # at end:
    reg.finalize(status="completed", final_checkpoint="path/to/final.pt")
    reg.save()
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"


def _run_id() -> str:
    """Generate a unique run ID: YYYYMMDD_HHMMSS_PID."""
    pid_suffix = f"_{os.getpid() % 10000:04d}"
    return time.strftime("%Y%m%d_%H%M%S") + pid_suffix


class RunRecord:
    """Tracks a single training run."""

    def __init__(self):
        self.run_id: str = ""
        self.data: dict = {}  # the full JSON-serializable record

        # In-memory accumulators (not saved directly — summarized at finalize)
        self._train_losses: List[tuple] = []   # (step, loss)
        self._train_norms: List[tuple] = []    # (step, norm)
        self._train_lrs: List[tuple] = []      # (step, lr)
        self._eval_losses: List[tuple] = []    # (step, eval_loss)

    @classmethod
    def start(cls, args, config, extra: Optional[dict] = None) -> "RunRecord":
        """Create a new run record from training args and model config."""
        rec = cls()
        rec.run_id = _run_id()

        # Resolve data path info
        data_path = Path(args.data).resolve()

        rec.data = {
            "run_id": rec.run_id,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished_at": None,
            "duration_sec": None,

            # Data
            "data": {
                "csv_path": str(data_path),
                "csv_name": data_path.name,
                "audit_id": None,
            },

            # Paths
            "paths": {
                "checkpoint_dir": str(Path(args.checkpoint).resolve()),
                "sp_model": str(Path(args.checkpoint).resolve() / "model_sp.model"),
            },

            # Hyperparameters — everything that affects training
            "hyperparameters": {
                "model_type": args.model_type,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "lr": args.lr,
                "vocab_size": args.vocab_size,
                "edim": config.edim,
                "layers": config.layers,
                "heads": config.heads,
                "dropout": config.dropout,
                "prop_dropout": args.prop_dropout,
                "grad_clip_norm": args.grad_clip_norm,
                "skip_norm": args.skip_norm,
                "ema_decay": args.ema_decay,
                "warmup_frac": getattr(args, "warmup_frac", 0.05),
                "total_batch_size": getattr(args, "total_batch_size", 524288),
                "qk_norm": getattr(args, "qk_norm", False),
                "gated_cross_attn": getattr(args, "gated_cross_attn", False),
                "encoder_lr_scale": getattr(args, "encoder_lr_scale", 1.0),
                "probe_interval": getattr(args, "probe_interval", 0),
                "compile": args.compile,
            },

            # Derived / computed at start
            "training_info": {},

            # Populated during training
            "loss_curve": [],       # [{step, loss}, ...]
            "eval_curve": [],       # [{step, eval_loss}, ...]
            "checkpoints": [],      # [{step, path}, ...]

            # Populated at finalize
            "loss_summary": {},
            "eval_summary": {},
            "norm_summary": {},
            "final_status": {},
        }

        if extra:
            rec.data["training_info"].update(extra)

        # Expert xattn specific
        if args.model_type == "expert_xattn":
            rec.data["hyperparameters"]["n_expert_xattn_layers"] = args.n_expert_xattn_layers
            if args.warmstart:
                rec.data["paths"]["warmstart"] = str(Path(args.warmstart).resolve())

        # Resume info
        rec.data["resume"] = {
            "resumed": False,
            "resume_path": None,
            "resume_step": 0,
            "fresh_optimizer": getattr(args, "fresh_optimizer", False),
        }

        rec._start_time = time.time()
        return rec

    def set_training_info(self, **kwargs):
        """Set derived training info (n_params, grad_accum, etc.)."""
        self.data["training_info"].update(kwargs)

    def set_resume_info(self, resume_path: str, resume_step: int):
        self.data["resume"]["resumed"] = True
        self.data["resume"]["resume_path"] = resume_path
        self.data["resume"]["resume_step"] = resume_step

    def set_data_info(self, **kwargs):
        """Set additional data info (version_id, row counts, etc.)."""
        self.data["data"].update(kwargs)

    # --- Logging during training ---

    def log_train(self, step: int, loss: float, norm: float, lr: float):
        """Log a training step. Called at log_interval."""
        self._train_losses.append((step, loss))
        self._train_norms.append((step, norm))
        self._train_lrs.append((step, lr))
        self.data["loss_curve"].append({"step": step, "loss": round(loss, 4)})

    def log_eval(self, step: int, eval_loss: float):
        """Log an eval loss measurement."""
        self._eval_losses.append((step, eval_loss))
        self.data["eval_curve"].append({"step": step, "eval_loss": round(eval_loss, 4)})

    def log_checkpoint(self, step: int, path: str):
        self.data["checkpoints"].append({"step": step, "path": path})

    # --- Finalization ---

    def finalize(self, status: str = "completed", final_checkpoint: Optional[str] = None,
                 final_step: Optional[int] = None, skip_count: int = 0):
        """Compute summaries and mark run as finished."""
        self.data["status"] = status
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["duration_sec"] = round(time.time() - self._start_time, 1)

        # Loss summary
        if self._train_losses:
            losses = [l for _, l in self._train_losses]
            n = len(losses)
            # First 10%, last 10%, overall
            first_n = max(1, n // 10)
            last_n = max(1, n // 10)
            self.data["loss_summary"] = {
                "n_logged": n,
                "first_loss": round(losses[0], 4),
                "last_loss": round(losses[-1], 4),
                "min_loss": round(min(losses), 4),
                "mean_first_10pct": round(sum(losses[:first_n]) / first_n, 4),
                "mean_last_10pct": round(sum(losses[-last_n:]) / last_n, 4),
            }

        # Eval summary
        if self._eval_losses:
            evals = [l for _, l in self._eval_losses]
            best_idx = evals.index(min(evals))
            self.data["eval_summary"] = {
                "n_evals": len(evals),
                "first_eval": round(evals[0], 4),
                "last_eval": round(evals[-1], 4),
                "best_eval": round(evals[best_idx], 4),
                "best_eval_step": self._eval_losses[best_idx][0],
            }

        # Norm summary
        if self._train_norms:
            norms = [n for _, n in self._train_norms]
            self.data["norm_summary"] = {
                "mean_norm": round(sum(norms) / len(norms), 4),
                "max_norm": round(max(norms), 4),
                "last_norm": round(norms[-1], 4),
            }

        # Final status
        self.data["final_status"] = {
            "final_step": final_step or (self._train_losses[-1][0] if self._train_losses else 0),
            "final_checkpoint": final_checkpoint,
            "skipped_steps": skip_count,
        }

    def save(self) -> Path:
        """Save run record to data/runs/ and update index."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        filename = f"run_{self.run_id}.json"
        run_path = RUNS_DIR / filename
        run_path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

        _update_index(run_path, self.data)
        return run_path

    def save_interrupted(self, error: Optional[str] = None):
        """Save whatever we have on crash/interrupt."""
        self.data["status"] = "interrupted"
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if hasattr(self, "_start_time"):
            self.data["duration_sec"] = round(time.time() - self._start_time, 1)
        if error:
            self.data["final_status"]["error"] = error
        # Compute partial summaries
        if self._train_losses:
            losses = [l for _, l in self._train_losses]
            self.data["loss_summary"] = {
                "n_logged": len(losses),
                "first_loss": round(losses[0], 4),
                "last_loss": round(losses[-1], 4),
                "min_loss": round(min(losses), 4),
            }
        if self._eval_losses:
            evals = [l for _, l in self._eval_losses]
            best_idx = evals.index(min(evals))
            self.data["eval_summary"] = {
                "n_evals": len(evals),
                "best_eval": round(evals[best_idx], 4),
                "best_eval_step": self._eval_losses[best_idx][0],
            }
        try:
            self.save()
        except Exception:
            pass  # best effort on crash


def _update_index(run_path: Path, data: dict) -> None:
    """Append/update entry in data/runs/index.json."""
    index_path = RUNS_DIR / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = []

    # Update existing entry or append
    entry = {
        "file": run_path.name,
        "run_id": data["run_id"],
        "status": data["status"],
        "data": data["data"].get("csv_name", ""),
        "version": data["data"].get("version_id", ""),
        "audit_id": data["data"].get("audit_id"),
        "checkpoint_dir": data.get("paths", {}).get("checkpoint_dir"),
        "model_type": data["hyperparameters"]["model_type"],
        "max_steps": data["hyperparameters"]["max_steps"],
        "lr": data["hyperparameters"]["lr"],
        "batch_size": data["hyperparameters"]["batch_size"],
        "last_loss": data.get("loss_summary", {}).get("last_loss"),
        "best_eval": data.get("eval_summary", {}).get("best_eval"),
        "started_at": data["started_at"],
        "duration_sec": data.get("duration_sec"),
    }

    # Replace if run_id already exists (e.g. interrupted then finalized)
    for i, e in enumerate(index):
        if e["run_id"] == data["run_id"]:
            index[i] = entry
            break
    else:
        index.append(entry)

    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI for listing / inspecting runs
# ---------------------------------------------------------------------------

def list_runs() -> list:
    """Load the run index."""
    index_path = RUNS_DIR / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_run(run_id_or_file: str) -> dict:
    """Load a single run record."""
    p = RUNS_DIR / run_id_or_file
    if not p.exists():
        p = RUNS_DIR / f"run_{run_id_or_file}.json"
    if not p.exists():
        raise FileNotFoundError(f"Run not found: {run_id_or_file}")
    return json.loads(p.read_text(encoding="utf-8"))


def print_run_table(runs: Optional[list] = None):
    """Print a formatted table of all runs."""
    if runs is None:
        runs = list_runs()
    if not runs:
        print("No training runs recorded.")
        return

    print(f"{'Run ID':<17} {'Status':<12} {'Data':<25} {'Steps':>7} {'LR':>9} "
          f"{'Last Loss':>10} {'Best Eval':>10} {'Duration':>10}")
    print("─" * 110)
    for r in runs:
        dur = r.get("duration_sec")
        dur_str = f"{dur/3600:.1f}h" if dur and dur > 3600 else f"{dur:.0f}s" if dur else "-"
        last_loss = r.get("last_loss")
        best_eval = r.get("best_eval")
        print(f"{r['run_id']:<17} {r['status']:<12} {r.get('data', '')[:25]:<25} "
              f"{r.get('max_steps', 0):>7} {r.get('lr', 0):>9.1e} "
              f"{last_loss if last_loss is not None else '-':>10} "
              f"{best_eval if best_eval is not None else '-':>10} "
              f"{dur_str:>10}")


def compare_runs(*run_ids: str):
    """Print a side-by-side comparison of runs."""
    runs = [load_run(rid) for rid in run_ids]
    labels = [r["run_id"] for r in runs]
    col_w = max(20, max(len(l) for l in labels) + 2)

    print(f"\n{'='*70}")
    print("TRAINING RUN COMPARISON")
    print(f"{'='*70}")

    # Key hyperparameters
    hp_keys = ["model_type", "epochs", "max_steps", "batch_size", "lr",
               "edim", "layers", "heads", "prop_dropout", "grad_clip_norm",
               "skip_norm", "ema_decay"]
    print("\nHyperparameters:")
    header = "".ljust(20) + "".join(l.rjust(col_w) for l in labels)
    print(f"  {header}")
    for key in hp_keys:
        vals = [str(r["hyperparameters"].get(key, "-")) for r in runs]
        # Highlight differences
        all_same = len(set(vals)) == 1
        marker = " " if all_same else "*"
        line = f"{marker} {key:<18}" + "".join(v.rjust(col_w) for v in vals)
        print(f"  {line}")

    # Data
    print("\nData:")
    for key in ("csv_name", "version_id", "train_rows", "eval_rows"):
        vals = [str(r["data"].get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))

    # Training info
    print("\nTraining Info:")
    for key in ("n_params", "grad_accum_steps", "steps_per_epoch", "effective_max_steps"):
        vals = [str(r.get("training_info", {}).get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))

    # Loss summary
    print("\nLoss Summary:")
    for key in ("first_loss", "last_loss", "min_loss", "mean_first_10pct", "mean_last_10pct"):
        vals = [str(r.get("loss_summary", {}).get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))

    # Eval summary
    print("\nEval Summary:")
    for key in ("first_eval", "last_eval", "best_eval", "best_eval_step"):
        vals = [str(r.get("eval_summary", {}).get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))

    # Norm summary
    print("\nNorm Summary:")
    for key in ("mean_norm", "max_norm", "last_norm"):
        vals = [str(r.get("norm_summary", {}).get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))

    # Final status
    print("\nFinal Status:")
    for key in ("status", "duration_sec", "final_step", "final_checkpoint", "skipped_steps"):
        if key == "status":
            vals = [r.get(key, "-") for r in runs]
        elif key == "duration_sec":
            vals = []
            for r in runs:
                d = r.get(key)
                vals.append(f"{d/3600:.1f}h" if d and d > 3600 else f"{d:.0f}s" if d else "-")
        else:
            vals = [str(r.get("final_status", {}).get(key, "-")) for r in runs]
        print(f"  {key:<18}" + "".join(v.rjust(col_w) for v in vals))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training run registry")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--show", metavar="RUN_ID", help="Show details of a run")
    parser.add_argument("--compare", nargs="+", metavar="RUN_ID", help="Compare runs side-by-side")
    args = parser.parse_args()

    if args.list:
        print_run_table()
    elif args.show:
        run = load_run(args.show)
        print(json.dumps(run, indent=2))
    elif args.compare:
        compare_runs(*args.compare)
    else:
        parser.print_help()

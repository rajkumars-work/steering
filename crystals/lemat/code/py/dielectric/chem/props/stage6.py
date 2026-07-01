"""
Stage-6: MLIP relaxation + property evaluation.

Consumes stage-5 rows (with structure_path), relaxes the structures with MLIP,
and computes dielectric-related properties using MLIP surrogates/phonons.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime

from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor

from .dielectric import Dielectrics, LEGACY_PROPS, MLIP_PROPS


DEFAULT_CHECKPOINTS_DIR = "/data/assets/checkpoints"
DEFAULT_OUTPUT_DIR = "/data/assets/runs/di"
SURROGATE_GPUS = 0.5
LOCAL_GPUS = 1.0
ENCODER_GPUS = 0.5  # two encoders at 0.25 each


def _set_checkpoints_env(checkpoints_dir: Optional[str]) -> None:
    if checkpoints_dir:
        os.environ["ATLAS_CHECKPOINTS_DIR"] = checkpoints_dir
    else:
        os.environ.setdefault("ATLAS_CHECKPOINTS_DIR", DEFAULT_CHECKPOINTS_DIR)


def _load_stage5_structures(
    stage5_rows: Iterable[Dict[str, object]],
) -> Tuple[List[Atoms], List[Dict[str, object]]]:
    atoms_list: List[Atoms] = []
    meta_list: List[Dict[str, object]] = []
    for row in stage5_rows:
        path = str(row.get("structure_path", "")).strip()
        if not path:
            print("Stage-6: skipping row with empty structure_path")
            continue
        if not os.path.exists(path):
            print(f"Stage-6: missing structure_path: {path}")
            continue
        try:
            atoms = ase_read(path)
        except Exception as exc:
            print(f"Stage-6: failed to read {path}: {exc}")
            continue
        atoms_list.append(atoms)
        meta_list.append(dict(row))
    return atoms_list, meta_list


def _merge_props(row: Dict[str, object], props: Dict[str, object]) -> Dict[str, object]:
    merged = dict(row)
    merged.update(props)
    return merged


def stage6_relax_and_props(
    stage5_rows: List[Dict[str, object]],
    batch_size: int = 4,
    checkpoints_dir: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    parallel_batches: int = 1,
    num_gpus: int = 0,
    keep_failed: bool = False,
    timeout_s: float = 0.0,
    wait_interval_s: float = 30.0,
) -> List[Dict[str, object]]:
    """Relax structures and compute MLIP properties.

    Args:
        stage5_rows: Rows from stage-5 (must include structure_path).
        batch_size: Structures per MLIP batch.
        checkpoints_dir: Override for ATLAS_CHECKPOINTS_DIR.
        keep_failed: If True, include rows with stage6_error set.
        timeout_s: Per-batch timeout in seconds (0 disables timeout).
        wait_interval_s: Poll interval in seconds while waiting on Ray tasks.
    """
    if not stage5_rows:
        return []

    _set_checkpoints_env(checkpoints_dir)

    atoms_list, meta_list = _load_stage5_structures(stage5_rows)
    if not atoms_list:
        return []

    os.makedirs(output_dir, exist_ok=True)

    if parallel_batches == 1:
        return _run_serial(
            atoms_list,
            meta_list,
            batch_size=batch_size,
            output_dir=output_dir,
            keep_failed=keep_failed,
        )

    return _run_parallel(
        stage5_rows,
        batch_size=batch_size,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        parallel_batches=parallel_batches,
        num_gpus=num_gpus,
        keep_failed=keep_failed,
        timeout_s=timeout_s,
        wait_interval_s=wait_interval_s,
    )


def stage6_fieldnames(base_fields: List[str]) -> List[str]:
    fields: List[str] = []
    for name in base_fields:
        if name not in fields:
            fields.append(name)
    if "relaxed_structure_path" not in fields:
        fields.append("relaxed_structure_path")
    for name in MLIP_PROPS + LEGACY_PROPS + ["stage6_error"]:
        if name not in fields:
            fields.append(name)
    return fields


def _run_serial(
    atoms_list: List[Atoms],
    meta_list: List[Dict[str, object]],
    batch_size: int,
    output_dir: str,
    keep_failed: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    with Dielectrics() as di:
        for start in range(0, len(atoms_list), batch_size):
            batch_atoms = atoms_list[start : start + batch_size]
            batch_meta = meta_list[start : start + batch_size]
            try:
                props_batch, relaxed_structs = di.compute(
                    batch_atoms,
                    relax=True,
                    return_relaxed_structures=True,
                )
            except Exception as exc:
                print(f"Stage-6: batch compute failed (items {start}-{start + len(batch_atoms) - 1}): {exc}")
                for atoms, row in zip(batch_atoms, batch_meta, strict=False):
                    _handle_single_failure(
                        di,
                        atoms,
                        row,
                        output_dir,
                        results,
                        keep_failed,
                    )
                continue

            for row, props, relaxed in zip(batch_meta, props_batch, relaxed_structs, strict=False):
                relaxed_atoms = AseAtomsAdaptor.get_atoms(relaxed)
                rel_path = _write_relaxed_structure(
                    relaxed_atoms, row, output_dir, len(results)
                )
                merged = _merge_props(row, props)
                merged["relaxed_structure_path"] = rel_path
                results.append(merged)
    return results


def _handle_single_failure(
    di: Dielectrics,
    atoms: Atoms,
    row: Dict[str, object],
    output_dir: str,
    results: List[Dict[str, object]],
    keep_failed: bool,
    tag: str = "",
) -> None:
    try:
        props, relaxed_structs = di.compute(
            [atoms],
            relax=True,
            return_relaxed_structures=True,
        )
        relaxed = relaxed_structs[0]
        relaxed_atoms = AseAtomsAdaptor.get_atoms(relaxed)
        rel_path = _write_relaxed_structure(
            relaxed_atoms, row, output_dir, len(results), tag=tag
        )
        merged = _merge_props(row, props[0])
        merged["relaxed_structure_path"] = rel_path
        results.append(merged)
    except Exception as exc_single:
        print(f"Stage-6: failed to compute properties for {row.get('structure_path')}: {exc_single}")
        if keep_failed:
            results.append(_merge_props(row, {"stage6_error": str(exc_single)}))


def _run_parallel(
    stage5_rows: List[Dict[str, object]],
    batch_size: int,
    output_dir: str,
    checkpoints_dir: Optional[str],
        parallel_batches: int,
    num_gpus: int,
    keep_failed: bool,
    timeout_s: float,
    wait_interval_s: float,
) -> List[Dict[str, object]]:
    import ray
    from atlas.test_utils.ray_test_setup import setup_ray_cluster

    if not ray.is_initialized():
        setup_ray_cluster()

    available_gpus = int(ray.available_resources().get("GPU", 0))
    if num_gpus > 0:
        available_gpus = min(available_gpus, num_gpus)

    if available_gpus <= 0:
        print("Stage-6: no GPUs detected by Ray; falling back to serial execution.")
        atoms_list, meta_list = _load_stage5_structures(stage5_rows)
        return _run_serial(
            atoms_list,
            meta_list,
            batch_size=batch_size,
            output_dir=output_dir,
            keep_failed=keep_failed,
        )

    shared_gpu_overhead = SURROGATE_GPUS * 2
    per_batch_gpu = LOCAL_GPUS + ENCODER_GPUS
    max_batches = max(1, int((available_gpus - shared_gpu_overhead) // per_batch_gpu))

    if parallel_batches <= 0:
        parallel_batches = max_batches
    else:
        parallel_batches = min(parallel_batches, max_batches)

    if parallel_batches <= 1:
        atoms_list, meta_list = _load_stage5_structures(stage5_rows)
        return _run_serial(
            atoms_list,
            meta_list,
            batch_size=batch_size,
            output_dir=output_dir,
            keep_failed=keep_failed,
        )

    batches = [
        stage5_rows[i : i + batch_size]
        for i in range(0, len(stage5_rows), batch_size)
    ]

    @ray.remote
    def _compute_batch(rows: List[Dict[str, object]], batch_id: str) -> List[Dict[str, object]]:
        _set_checkpoints_env(checkpoints_dir)
        os.makedirs(output_dir, exist_ok=True)

        atoms_list, meta_list = _load_stage5_structures(rows)
        if not atoms_list:
            return []

        results: List[Dict[str, object]] = []
        with Dielectrics(
            name_suffix=batch_id,
            local_gpus=LOCAL_GPUS,
            surrogate_gpus=SURROGATE_GPUS,
            reuse_surrogates=True,
            kill_surrogates=False,
            shutdown_ray=False,
        ) as di:
            try:
                props_batch, relaxed_structs = di.compute(
                    atoms_list,
                    relax=True,
                    return_relaxed_structures=True,
                )
            except Exception as exc:
                print(f"Stage-6: batch compute failed in worker {batch_id}: {exc}")
                for atoms, row in zip(atoms_list, meta_list, strict=False):
                    _handle_single_failure(
                        di,
                        atoms,
                        row,
                        output_dir,
                        results,
                        keep_failed,
                        tag=batch_id,
                    )
                return results

            for row, props, relaxed in zip(meta_list, props_batch, relaxed_structs, strict=False):
                relaxed_atoms = AseAtomsAdaptor.get_atoms(relaxed)
                rel_path = _write_relaxed_structure(
                    relaxed_atoms, row, output_dir, len(results), tag=batch_id
                )
                merged = _merge_props(row, props)
                merged["relaxed_structure_path"] = rel_path
                results.append(merged)

        return results

    pending: Dict[object, Tuple[str, List[Dict[str, object]], float, float]] = {}
    for batch in batches:
        batch_id = str(uuid.uuid4())[:8]
        ref = _compute_batch.remote(batch, batch_id)
        start = time.time()
        pending[ref] = (batch_id, batch, start, start)

    results: List[Dict[str, object]] = []
    while pending:
        ready, _ = ray.wait(
            list(pending.keys()),
            num_returns=1,
            timeout=wait_interval_s,
        )

        for ref in ready:
            batch_id, batch_rows, _start, _last_log = pending.pop(ref)
            try:
                chunk = ray.get(ref)
            except Exception as exc:
                print(f"Stage-6: batch {batch_id} failed in ray.get: {exc}")
                if keep_failed:
                    for row in batch_rows:
                        results.append(_merge_props(row, {"stage6_error": str(exc)}))
                continue
            results.extend(chunk)

        now = time.time()

        if not ready:
            # Heartbeat: emit per-batch liveness while waiting.
            for ref, (batch_id, _rows, start, last_log) in list(pending.items()):
                if now - last_log >= wait_interval_s:
                    elapsed = int(now - start)
                    print(f"Stage-6: batch {batch_id} still running after {elapsed}s")
                    pending[ref] = (batch_id, _rows, start, now)

        if timeout_s:
            timed_out = [
                ref
                for ref, (batch_id, _rows, start, _last_log) in pending.items()
                if now - start > timeout_s
            ]
            for ref in timed_out:
                batch_id, batch_rows, start, _last_log = pending.pop(ref)
                try:
                    ray.cancel(ref, force=True)
                except Exception as exc:
                    print(f"Stage-6: failed to cancel batch {batch_id}: {exc}")
                elapsed = int(now - start)
                msg = f"timeout after {elapsed}s (batch {batch_id})"
                print(f"Stage-6: {msg}")
                if keep_failed:
                    for row in batch_rows:
                        results.append(_merge_props(row, {"stage6_error": msg}))

    return results


def _sanitize_filename(value: str) -> str:
    if not value:
        return "unknown"
    cleaned = []
    for ch in value:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _write_relaxed_structure(
    atoms: Atoms,
    row: Dict[str, object],
    output_dir: str,
    index: int,
    tag: str = "",
) -> str:
    formula = _sanitize_filename(str(row.get("composition_formula", "")))
    rank = str(row.get("structure_rank", ""))
    suffix = f"s{rank}" if rank else "s"
    time_str = datetime.now().strftime("%d_%H")
    filename = f"{formula}_{suffix}_{time_str}_{index}.extxyz"
    path = os.path.join(output_dir, filename)
    ase_write(path, atoms)
    return path

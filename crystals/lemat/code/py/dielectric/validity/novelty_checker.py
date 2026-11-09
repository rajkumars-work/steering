"""Backward-compatible wrapper — canonical lives in chem.novelty."""

from chem.novelty import (
    DEFAULT_LEMAT_PARQUET as LEMATBULK_PARQUET,
    LematNoveltyChecker as NoveltyChecker,
    check_lemat_novelty as check_novelty_lematbulk,
    get_lematbulk_checker,
)

__all__ = [
    "LEMATBULK_PARQUET",
    "NoveltyChecker",
    "check_novelty_lematbulk",
    "get_lematbulk_checker",
]

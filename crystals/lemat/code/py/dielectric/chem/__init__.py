"""chem — Materials chemistry utilities for dielectric property discovery.

Lightweight re-exports of the most-used public functions.  Heavy
dependencies (MACE, Ray, xgboost, ...) are imported lazily inside each
sub-module so that ``import chem`` stays fast.
"""

# Validity / screening
from .check_validity import is_realistic

# Pareto front
from .pareto_front import ParetoFront

# Material classification
from .material_classifier import classify_material

# Dielectric flagging
from .flag_dielectric import flag_dielectric

__all__ = [
    "is_realistic",
    "ParetoFront",
    "classify_material",
    "flag_dielectric",
]

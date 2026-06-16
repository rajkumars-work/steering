"""GPU property calculators (relaxation, dielectrics, phonons)."""

from .relax import get_calculator, relax_structure

# Lazy import: dielectrics depends on atlas/torch_sim which may not be installed
# or may have version conflicts. Only import when actually needed.
def __getattr__(name):
    if name in ("Dielectrics", "compute_eps_0"):
        from .dielectric import Dielectrics, compute_eps_0
        globals()["Dielectrics"] = Dielectrics
        globals()["compute_eps_0"] = compute_eps_0
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_calculator",
    "relax_structure",
    "Dielectrics",
    "compute_eps_0",
]

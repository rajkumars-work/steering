"""MACE / e3nn compatibility shim — import this before any MACE model loading.

Old MACE checkpoints store TorchScript bytes in e3nn's __codegen__ state that
are incompatible with current PyTorch/e3nn versions. This module patches the
deserialization and regenerates all compiled code after model loading.

Usage:
    # At the top of any module that loads MACE:
    import chem.props.mace_compat  # noqa: apply patch

    # Or explicitly:
    from chem.props.mace_compat import patch_mace_model
    calc = MACECalculator(...)
    patch_mace_model(calc)
"""

import io
import pickle

import torch
from e3nn.util.codegen._mixin import CodeGenMixin as _CodeGenMixin


# ---------------------------------------------------------------------------
# 1. Patch __setstate__ on CodeGenMixin (handles deserialization)
# ---------------------------------------------------------------------------

def _codegen_setstate_compat(self, d: dict) -> None:
    d = d.copy()
    codegen_state = d.pop("__codegen__", None)
    if hasattr(super(_CodeGenMixin, self), "__setstate__"):
        super(_CodeGenMixin, self).__setstate__(d)
    else:
        self.__dict__.update(d)
    if codegen_state is not None:
        for fname, value in codegen_state.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                buffer_type, buffer = value
            else:
                buffer_type, buffer = "torchscript", value
            if buffer_type == "fx":
                smod = pickle.loads(buffer)
            elif buffer_type == "torchscript":
                smod = torch.jit.load(io.BytesIO(buffer))
            else:
                raise NotImplementedError(f"Unknown codegen buffer_type: {buffer_type!r}")
            setattr(self, fname, smod)
        self.__codegen__ = list(codegen_state.keys())


_CodeGenMixin.__setstate__ = _codegen_setstate_compat


# ---------------------------------------------------------------------------
# 2. Regenerate compiled code on a loaded MACE model
# ---------------------------------------------------------------------------

def patch_mace_model(calc):
    """Regenerate all e3nn compiled code on a loaded MACE calculator.

    Call this after MACECalculator(...) or mace_mp() to fix:
      - SphericalHarmonics.sph_func
      - Activation.paths
      - Linear._compiled_main
      - TensorProduct._compiled_main_left_right
    """
    from e3nn import get_optimization_defaults
    from e3nn.nn._activation import Activation
    from e3nn.o3._spherical_harmonics import SphericalHarmonics, _spherical_harmonics
    from e3nn.o3._linear import Linear as E3nnLinear, _codegen_linear
    from e3nn.o3._tensor_product import TensorProduct as E3nnTP
    from e3nn.o3._tensor_product._codegen import (
        codegen_tensor_product_left_right,
        codegen_tensor_product_right,
    )

    # Get the underlying ASE calculator's models
    models = getattr(calc, 'models', None)
    if models is None:
        # Try MACECalculator wrapper patterns
        ase_calc = getattr(calc, 'ase_calc', None)
        if ase_calc is not None:
            models = getattr(ase_calc, 'models', None)
    if models is None:
        return

    jit_mode = get_optimization_defaults().get("jit_mode", "default")
    device = next(iter(models[0].parameters())).device if models else "cpu"

    for m in models:
        for mod in m.modules():
            if isinstance(mod, SphericalHarmonics) and not hasattr(mod, "sph_func"):
                if jit_mode == "script":
                    mod.sph_func = torch.jit.script(_spherical_harmonics)
                elif jit_mode == "compile":
                    mod.sph_func = torch.compile(_spherical_harmonics, fullgraph=True)
                else:
                    mod.sph_func = _spherical_harmonics

            if isinstance(mod, Activation) and not hasattr(mod, "paths"):
                mod.paths = [
                    (mul, (l, p), act)
                    for (mul, (l, p)), act in zip(mod.irreps_in, mod.acts)
                ]

            if isinstance(mod, E3nnLinear) and not hasattr(mod, "_compiled_main"):
                graphmod, _, _ = _codegen_linear(
                    mod.irreps_in, mod.irreps_out, mod.instructions,
                    shared_weights=mod.shared_weights,
                    optimize_einsums=mod._optimize_einsums,
                )
                mod._codegen_register({"_compiled_main": graphmod})

            if isinstance(mod, E3nnTP) and not hasattr(mod, "_compiled_main_left_right"):
                try:
                    code_lr = codegen_tensor_product_left_right(
                        mod.irreps_in1, mod.irreps_in2, mod.irreps_out,
                        mod.instructions,
                        shared_weights=mod.shared_weights,
                        specialized_code=mod._specialized_code,
                        optimize_einsums=mod._optimize_einsums,
                    )
                    codegen = {"_compiled_main_left_right": code_lr}
                    if not hasattr(mod, "_compiled_main_right"):
                        try:
                            code_r = codegen_tensor_product_right(
                                mod.irreps_in1, mod.irreps_in2, mod.irreps_out,
                                mod.instructions,
                                shared_weights=mod.shared_weights,
                                specialized_code=mod._specialized_code,
                                optimize_einsums=mod._optimize_einsums,
                            )
                            codegen["_compiled_main_right"] = code_r
                        except Exception:
                            pass
                    mod._codegen_register(codegen)
                except Exception:
                    pass

        # Move regenerated modules to the correct device
        m.to(device)

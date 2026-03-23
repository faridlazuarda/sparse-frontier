"""Compatibility shim for vLLM version-sensitive imports.

Centralizes all imports that changed between vLLM 0.11.x and 0.18.x so the
rest of the codebase can use a single, stable import path.
"""

import importlib.metadata
from packaging.version import Version

_vllm_version: Version | None = None


def get_vllm_version() -> Version:
    global _vllm_version
    if _vllm_version is None:
        _vllm_version = Version(importlib.metadata.version("vllm"))
    return _vllm_version


# ---------------------------------------------------------------------------
# flash_attn_varlen_func  (stable across 0.11–0.18)
# ---------------------------------------------------------------------------
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: E402

# ---------------------------------------------------------------------------
# flash_attn_with_kvcache
#   vLLM <= ~0.16: available via compiled ext in vllm.vllm_flash_attn
#   vLLM >= 0.17:  removed — fall back to standalone flash-attn package
#
# Lazy-loaded so that prefill-only users (FlexPrefill, Block-Sparse, etc.)
# are not forced to install the standalone flash-attn package.
# ---------------------------------------------------------------------------
_flash_attn_with_kvcache = None


def flash_attn_with_kvcache(*args, **kwargs):
    global _flash_attn_with_kvcache
    if _flash_attn_with_kvcache is None:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                flash_attn_with_kvcache as _fn,
            )
        except ImportError:
            try:
                from flash_attn.flash_attn_interface import (
                    flash_attn_with_kvcache as _fn,
                )
            except ImportError:
                raise ImportError(
                    "flash_attn_with_kvcache is not available. "
                    "vLLM >= 0.17 no longer bundles this kernel. "
                    "Install the standalone flash-attn package: "
                    "pip install flash-attn --no-build-isolation"
                ) from None
        _flash_attn_with_kvcache = _fn
    return _flash_attn_with_kvcache(*args, **kwargs)


# ---------------------------------------------------------------------------
# get_open_port
#   vLLM <= 0.11: vllm.utils.get_open_port
#   vLLM >= 0.12: vllm.utils.network_utils.get_open_port
# ---------------------------------------------------------------------------
try:
    from vllm.utils import get_open_port  # noqa: E402
except ImportError:
    from vllm.utils.network_utils import get_open_port  # noqa: E402

# ---------------------------------------------------------------------------
# Distributed helpers  (stable across 0.11–0.18)
# ---------------------------------------------------------------------------
from vllm.distributed import (  # noqa: E402
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

# ---------------------------------------------------------------------------
# Version-conditional helpers
#
# Keep all "which vLLM version needs what" knowledge here so callers don't
# have to know raw version thresholds.
# ---------------------------------------------------------------------------

def get_extra_llm_kwargs() -> dict:
    """Return version-dependent keyword arguments for the vllm.LLM constructor."""
    kwargs: dict = {}
    # disable_hybrid_kv_cache_manager was removed in vLLM 0.18.
    # It fixes gemma3 sliding-window cache handling on older versions.
    if get_vllm_version() < Version("0.18"):
        kwargs["disable_hybrid_kv_cache_manager"] = True
    return kwargs


def configure_vllm_environment() -> None:
    """Set vLLM environment variables that vary by version."""
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # VLLM_USE_V1 and VLLM_FLASH_ATTN_VERSION were removed in vLLM 0.17
    # (v1 is default; FA version is configured via --attention-config).
    if get_vllm_version() < Version("0.17"):
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"


__all__ = [
    "get_vllm_version",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_open_port",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "tensor_model_parallel_all_gather",
    "get_extra_llm_kwargs",
    "configure_vllm_environment",
]

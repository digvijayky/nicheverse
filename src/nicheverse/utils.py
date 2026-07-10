"""Determinism, environment capture, and integrity helpers for reproducible runs.

This module provides the small set of utilities that the rest of the package
uses to (a) seed all random number generators in a single call,
(b) snapshot the host environment to a JSON file so reviewers can compare
package versions and CUDA build numbers, and (c) hash arrays and files to
detect drift across reruns.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_VALUE = ":4096:8"


def seed_everything(seed: int = 9, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Parameters
    ----------
    seed
        Integer seed applied to ``random``, ``numpy``, ``torch.manual_seed``,
        and ``torch.cuda.manual_seed_all``.
    deterministic
        If True, request deterministic cuDNN ops and call
        ``torch.use_deterministic_algorithms(True, warn_only=True)``. If any
        op the model uses lacks a deterministic kernel, a warning will be
        emitted at runtime; the run will still complete, but bit-for-bit
        reproducibility is not guaranteed.

    Notes
    -----
    For full bit-identical CUDA runs, the environment variable
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` must be set BEFORE the first ``torch``
    import in the process. This function sets it in ``os.environ`` for
    completeness, but if ``torch`` was already imported, setting the variable
    here has no effect on cuBLAS for the current process. To guarantee the
    setting takes effect, export it in your shell or set it before launching
    Python (e.g. ``CUBLAS_WORKSPACE_CONFIG=:4096:8 python ...``).
    See ``DETERMINISM.md`` for the full reproducibility contract.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if _CUBLAS_ENV not in os.environ:
        os.environ[_CUBLAS_ENV] = _CUBLAS_VALUE
        if "torch" in sys.modules:
            logger.warning(
                "%s was set after torch import; cuBLAS may not pick it up in this process. "
                "Export it before launching Python for guaranteed effect.",
                _CUBLAS_ENV,
            )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            logger.warning(
                "torch.use_deterministic_algorithms failed (%s); "
                "some non-deterministic ops may still be active.",
                exc,
            )


def _safe_version(pkg: str) -> str | None:
    """Return ``importlib.metadata.version(pkg)``, falling back to module ``__version__``."""
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def env_snapshot() -> dict[str, Any]:
    """Capture Python, OS, key library versions, CUDA info, and (best-effort) git SHA.

    Returns
    -------
    dict
        A JSON-safe dict suitable for ``json.dumps``. Missing or unimportable
        libraries appear as ``None``.
    """
    pkg_version = _safe_version("nicheverse")
    info: dict[str, Any] = {
        "nicheverse_version": pkg_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    # scikit-learn ships under the import name ``sklearn`` but the dist name ``scikit-learn``.
    pkg_names = {
        "scanpy": "scanpy",
        "anndata": "anndata",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
        "pyarrow": "pyarrow",
    }
    for key, dist in pkg_names.items():
        info[key] = _safe_version(dist)
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["git_sha"] = sha
    except Exception:
        info["git_sha"] = None
    return info


def write_env_snapshot(path: str | Path) -> Path:
    """Write :func:`env_snapshot` to ``path`` (as JSON) and return ``path`` as ``Path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env_snapshot(), indent=2))
    return path


def sha256_array(arr: np.ndarray) -> str:
    """Stable SHA256 of an ndarray's content, dtype, and shape.

    Hashes ``str(dtype)``, ``str(shape)``, then ``tobytes()`` of the
    contiguous array. Two arrays with different dtypes but identical content
    intentionally produce different hashes.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    """SHA256 of a file, streamed in ``block``-sized chunks (default 1 MiB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

"""
Invoke operations on large models in parallel.
"""

from importlib.metadata import PackageNotFoundError, version

from ._worker import is_mp_worker, is_worker

try:
    __version__ = version("parinvoke")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["is_worker", "is_mp_worker"]

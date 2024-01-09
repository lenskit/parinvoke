"""
Invoke operations on large models in parallel.
"""

from importlib.metadata import PackageNotFoundError, version

from ._worker import is_mp_worker, is_worker
from .isolate import run_sp
from .parallel import invoker
from .sharing import persist

try:
    __version__ = version("parinvoke")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["invoker", "run_sp", "persist", "is_worker", "is_mp_worker"]

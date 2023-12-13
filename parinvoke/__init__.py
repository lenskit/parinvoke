"""
Invoke operations on large models in parallel.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parinvoke")
except PackageNotFoundError:
    # package is not installed
    pass

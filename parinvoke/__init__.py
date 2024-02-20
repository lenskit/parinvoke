# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Invoke operations on large models in parallel.
"""

from importlib.metadata import PackageNotFoundError, version

from ._worker import is_mp_worker, is_worker
from .context import InvokeContext

try:
    __version__ = version("parinvoke")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["InvokeContext", "is_worker", "is_mp_worker"]

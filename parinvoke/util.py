# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from contextlib import contextmanager


@contextmanager
def set_env_var(var: str, val: str | None):
    """
    Context manager to set an environment variable and restore it.
    Primarily used for testing.
    """
    old_val = os.environ.get(var, None)
    try:
        if val is None:
            if old_val is not None:
                del os.environ[var]
        else:
            os.environ[var] = val
        yield
    finally:
        if old_val is not None:
            os.environ[var] = old_val
        elif val is not None:
            del os.environ[var]

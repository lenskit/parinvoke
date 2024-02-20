# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Configuration and environment support for ParInvoke.
"""

import logging
import multiprocessing as mp
import os
import warnings
from collections.abc import Generator
from typing import Callable, NamedTuple, Optional, TypeVar, overload

_log = logging.getLogger(__name__)
T = TypeVar


class VarName(NamedTuple):
    name: str
    deprecated: bool = False


class InvokeConfig:
    """
    Configuration settings for the parallel backend.

    Args:
        prefix:
            A prefix for environment variables to recognize. If the prefix is
            ``FOO``, then for a variable ``PARINVOKE_X`` (like
            ``PARINVOKE_NUM_PROCS``), ``FOO_X`` is also recognized and takes
            precedence over ``PARINVOKE_X``.
        max_default:
            See :attr:`max_default`.
        core_div:
            See :attr:`core_div`.
    """

    env_prefixes: list[str]
    aliases: dict[str, list[VarName]]

    proc_counts: Optional[list[int]]
    """
    The number of processes to use.
    """
    max_default: Optional[int]
    """
    The maximum parallelism if no parallelism is specified explicitly.
    """
    core_div: int
    """
    A divisor to scale down the number of cores.
    """
    level: int
    """
    The parallelism nesting level.  See :ref:`nesting-levels`.
    """

    def __init__(
        self,
        nprocs: int | list[int] | None = None,
        *,
        prefix: str | None = None,
        max_default: int | None = None,
        core_div: int = 1,
        level: int = 0,
    ):
        self.env_prefixes = ["PARINVOKE"]
        self.aliases = {}

        if isinstance(nprocs, list):
            self.proc_counts = nprocs
        elif nprocs is not None:
            self.proc_counts = [nprocs]
        else:
            self.proc_counts = None

        if prefix is not None:
            self.env_prefixes.insert(0, prefix)

        self.max_default = max_default
        self.core_div = core_div
        self.level = level

    @staticmethod
    def default():
        return InvokeConfig()

    def add_alias(self, alias: str, name: str, *, deprecated: bool = False):
        """
        Add an alias for an environment variable.
        """
        if name not in self.aliases:
            self.aliases[name] = []
        self.aliases[name].append(VarName(alias, deprecated))

    @overload
    def env_var(self, name: str) -> tuple[str, str] | None:
        ...

    @overload
    def env_var(self, name: str, cast: Callable[[str], T]) -> tuple[str, T] | None:
        ...

    def env_var(self, name: str, cast: Callable[[str], object] | None = None):
        """
        Get an environment variable by name, optionally casting, and looking up
        with all configured prefixes.
        """
        for vn in self._var_names(name):
            val = os.environ.get(vn.name, None)
            if val is not None:
                if vn.deprecated:
                    _log.warning("configured deprecated environmetn variable %s", vn.name)
                    warnings.warn(
                        f"deprecated environment variable {vn.name} configured", DeprecationWarning
                    )
                if cast is not None:
                    return vn.name, cast(val)
                else:
                    return vn.name, val

    def _var_names(self, name: str) -> Generator[VarName, None, None]:
        yield from self.aliases.get(name, [])
        for pfx in self.env_prefixes:
            yield VarName(f"{pfx}_{name}")

    def _configure_count(self):
        """
        Ensure the processor counts are configured.
        """
        if self.proc_counts is not None:
            return

        npv = self.env_var("NUM_PROCS")
        if npv is not None:
            vn, nprocs = npv
            _log.debug("found process count config in %s=%s", vn, nprocs)
            self.proc_counts = [int(s) for s in nprocs.split(",")]
        else:
            nprocs = max(mp.cpu_count() // self.core_div, 1)
            if self.max_default is not None:
                nprocs = min(nprocs, self.max_default)
            self.proc_counts = [nprocs, self.core_div]

    def proc_count(self, *, level: int | None = None) -> int:
        """
        Get the number of desired jobs for multiprocessing operations.

        This count can come from a number of sources, in decreasing order of
        precedence:

        * The value provided to the :class:`ParallelConfig` constructor.
        * The ``PARINVOKE_NUM_PROCS`` environment variable and its aliases and
          alternate prefixes (see :ref:`env-vars`).
        * The number of CPUs, as returned by :func:`mp.cpu_count`, capped by
          :attr:`max_default`.

        Args:
            level:
                A process nesting level, overriding :attr:`level`.  See
                :ref:`nesting-levels`.

        Returns:
            int: The number of processes desired.
        """

        self._configure_count()
        assert self.proc_counts is not None
        if level is None:
            level = self.level

        if level >= len(self.proc_counts):
            return 1
        else:
            return self.proc_counts[level]

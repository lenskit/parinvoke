# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterator,
    TypeVar,
    cast,
)

import seedbank
from numpy.random import SeedSequence
from threadpoolctl import threadpool_limits

from parinvoke.logging import log_queue
from parinvoke.sharing import PersistedModel, persist

from ._worker import initialize_worker
from .config import ParallelConfig

T = TypeVar("T")
R = TypeVar("R")
_log = logging.getLogger(__name__)


def _mp_invoke_worker(*args: Any):
    model = __work_model.get()
    return __work_func(model, *args)


def _initialize_mp_worker(
    model: PersistedModel[object],
    func: bytes,
    threads: int,
    log_queue: mp.Queue[logging.LogRecord] | None,
    seed: SeedSequence | None,
):
    seed = seedbank.derive_seed(mp.current_process().name, base=seed)
    initialize_worker(log_queue, seed, True)
    global __work_model, __work_func

    # disable BLAS threading
    threadpool_limits(limits=1, user_api="blas")

    __work_model = model
    # deferred function unpickling to minimize imports before initialization
    __work_func = pickle.loads(func)

    _log.debug("worker %d ready (process %s)", os.getpid(), mp.current_process())


def invoker(
    model: T,
    func: Callable[Concatenate[T, ...], R],
    n_jobs: int | None = None,
    *,
    persist_method: str | None = None,
    config: ParallelConfig | None = None,
) -> ModelOpInvoker[T, R]:
    """
    Get an appropriate invoker for performing oeprations on ``model``.

    Args:
        model(obj): The model object on which to perform operations.
        func(function): The function to call.  The function must be pickleable.
        n_jobs(int or None):
            The number of processes to use for parallel operations.  If ``None``, will
            call :func:`proc_count` with a maximum default process count of 4.
        persist_method(str or None):
            The persistence method to use.  Passed as ``method`` to
            :func:`lenskit.sharing.persist`.

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.
    """
    if config is None:
        config = ParallelConfig.default()
    if n_jobs is None:
        n_jobs = config.proc_count()

    if n_jobs == 1:
        return InProcessOpInvoker(model, func)
    else:
        return ProcessPoolOpInvoker(model, func, n_jobs, persist_method, config)


class ModelOpInvoker(ABC, Generic[T, R]):
    """
    Interface for invoking operations on a model, possibly in parallel.  The operation
    invoker is configured with a model and a function to apply, and applies that function
    to the arguments supplied in `map`.  Child process invokers also route logging messages
    to the parent process, so logging works even with multiprocessing.

    An invoker is a context manager that calls :meth:`shutdown` when exited.
    """

    @abstractmethod
    def map(self, *iterables: Any) -> Iterator[R]:
        """
        Apply the configured function to the model and iterables.  This is like :py:func:`map`,
        except it supplies the invoker's model as the first object to ``func``.

        Args:
            iterables: Iterables of arguments to provide to the function.

        Returns:
            iterable: An iterable of the results.
        """
        pass

    def shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.shutdown()


class InProcessOpInvoker(ModelOpInvoker[T, R]):
    model: T | None

    def __init__(self, model: T, func: Callable[Concatenate[T, ...], R]):
        _log.info("setting up in-process worker")
        if isinstance(model, PersistedModel):
            self.model = model.get()
        else:
            self.model = model
        self.function = func

    def map(self, *iterables: Any) -> Iterator[R]:
        assert self.model is not None
        proc = partial(self.function, self.model)
        return map(proc, *iterables)

    def shutdown(self):
        self.model = None


class ProcessPoolOpInvoker(ModelOpInvoker[T, R]):
    _close_key = None

    def __init__(
        self,
        model: T,
        func: Callable[Concatenate[T, ...], R],
        n_jobs: int,
        persist_method: str | None,
        config: ParallelConfig,
    ):
        key: PersistedModel[T]
        if isinstance(model, PersistedModel):
            _log.debug("model already persisted")
            key = cast(PersistedModel[T], model)
        else:
            _log.debug("persisting model with method %s", persist_method)
            key = persist(model, method=persist_method)
            self._close_key = key

        _log.debug("persisting function")
        func_pkl = pickle.dumps(func)
        ctx = mp.get_context("spawn")
        _log.info("setting up ProcessPoolExecutor w/ %d workers", n_jobs)
        kid_tc = config.proc_count(level=1)
        self.executor = ProcessPoolExecutor(
            n_jobs,
            ctx,
            _initialize_mp_worker,
            (key, func_pkl, kid_tc, log_queue(ctx), seedbank.root_seed()),
        )

    def map(self, *iterables: Any) -> Iterator[R]:
        return cast(Iterator[R], self.executor.map(_mp_invoke_worker, *iterables))

    def shutdown(self):
        self.executor.shutdown()
        if self._close_key is not None:
            self._close_key.close()
            del self._close_key

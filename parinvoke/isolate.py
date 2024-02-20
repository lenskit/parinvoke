# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
from typing import Any, Callable, ParamSpec, TypeVar

import seedbank
from numpy.random import SeedSequence

from parinvoke._worker import initialize_worker
from parinvoke.context import InvokeContext
from parinvoke.logging import log_queue

T = TypeVar("T")
P = ParamSpec("P")
_log = logging.getLogger(__name__)


def _sp_worker(
    log_queue: mp.Queue[logging.LogRecord],
    seed: SeedSequence,
    res_queue: mp.Queue[tuple[bool, Any | Exception]],
    context: InvokeContext,
    func_pkl: bytes,
    args: list[Any],
    kwargs: dict[str, Any],
):
    initialize_worker(log_queue, seed, context=context)
    _log.debug("unpickling worker function")
    func = pickle.loads(func_pkl)
    _log.debug("running %s in worker", func)
    try:
        res = func(*args, **kwargs)
        _log.debug("completed successfully")
        res_queue.put((True, res))
    except Exception as e:
        _log.error("failed, transmitting error %r", e)
        res_queue.put((False, e))


def run_sp(context: InvokeContext, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a function in a subprocess and return its value.  This is for achieving subprocess
    isolation, not parallelism.  The subprocess is configured so things like logging work
    correctly, and is initialized with a derived random seed.
    """
    mp_ctx = mp.get_context("spawn")
    rq = mp_ctx.SimpleQueue()
    seed = seedbank.derive_seed()
    # we pre-pickle the function to defer imports
    func_pkl = pickle.dumps(func, pickle.HIGHEST_PROTOCOL)
    worker_args = (log_queue(mp_ctx), seed, rq, context, func_pkl, args, kwargs)
    _log.debug("spawning subprocess to run %s", func)
    proc = mp_ctx.Process(target=_sp_worker, args=worker_args)
    proc.start()
    _log.debug("waiting for process %s to return", proc)
    success, payload = rq.get()
    _log.debug("received success=%s", success)
    _log.debug("waiting for process %s to exit", proc)
    proc.join()
    if proc.exitcode:
        _log.error("subprocess failed with code %d", proc.exitcode)
        raise RuntimeError("subprocess failed with code " + str(proc.exitcode))
    if success:
        return payload
    else:
        _log.error("subprocess raised exception: %s", payload)
        raise ChildProcessError("error in child process", payload)

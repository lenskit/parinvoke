# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Code to support worker processes.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import multiprocessing as mp
import os
import pickle
from typing import Any, TypeVar

import seedbank
from numpy.random import SeedSequence
from threadpoolctl import threadpool_limits

from parinvoke.context import InvokeContext
from parinvoke.sharing import PersistedModel

T = TypeVar("T")
_log = logging.getLogger(__name__)
__is_worker = False
__is_mp_worker = False
child_persist_context: InvokeContext | None = None


def is_worker() -> bool:
    "Query whether the process is a worker, either for MP or for isolation."
    return __is_worker


def is_mp_worker() -> bool:
    "Query whether the current process is a multiprocessing worker."
    return __is_mp_worker


def initialize_worker(
    log_queue: mp.Queue[logging.LogRecord] | None,
    seed: SeedSequence | None,
    multi: bool = False,
    context: InvokeContext | None = None,
):
    "Initialize a worker process."
    global __is_worker, __is_mp_worker, child_persist_context
    child_persist_context = context
    __is_worker = True
    __is_mp_worker = multi
    faulthandler.enable()
    if seed is not None:
        seedbank.initialize(seed)
    if log_queue is not None:
        h = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        h.setLevel(logging.DEBUG)

    _log.debug("worker %s initialized", mp.current_process().name)


def mp_invoke_worker(*args: Any):
    model = __work_model.get()
    return __work_func(model, *args)


def initialize_mp_worker(
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
    threadpool_limits(limits=threads, user_api="blas")

    __work_model = model
    # deferred function unpickling to minimize imports before initialization
    __work_func = pickle.loads(func)

    _log.debug("worker %d ready (process %s)", os.getpid(), mp.current_process())

"""
Code to support worker processes.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import multiprocessing as mp
from typing import TypeVar

import seedbank
from numpy.random import SeedSequence

T = TypeVar("T")
_log = logging.getLogger(__name__)
__is_worker = False
__is_mp_worker = False


def is_worker() -> bool:
    "Query whether the process is a worker, either for MP or for isolation."
    return __is_worker


def is_mp_worker() -> bool:
    "Query whether the current process is a multiprocessing worker."
    return __is_mp_worker


def initialize_worker(
    log_queue: mp.Queue[logging.LogRecord] | None, seed: SeedSequence | None, multi: bool = False
):
    "Initialize a worker process."
    global __is_worker, __is_mp_worker
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

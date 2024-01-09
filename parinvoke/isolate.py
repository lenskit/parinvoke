import logging
import multiprocessing as mp
from typing import Any, Callable, ParamSpec, TypeVar

import seedbank
from numpy.random import SeedSequence

from parinvoke._worker import initialize_worker
from parinvoke.logging import log_queue

T = TypeVar("T")
P = ParamSpec("P")
_log = logging.getLogger(__name__)


def _sp_worker(
    log_queue: mp.Queue[logging.LogRecord],
    seed: SeedSequence,
    res_queue: mp.Queue[tuple[bool, T | Exception]],
    func: Callable[..., T],
    args: list[Any],
    kwargs: dict[str, Any],
):
    initialize_worker(log_queue, seed)
    _log.debug("running %s in worker", func)
    try:
        res = func(*args, **kwargs)
        _log.debug("completed successfully")
        res_queue.put((True, res))
    except Exception as e:
        _log.error("failed, transmitting error %r", e)
        res_queue.put((False, e))


def run_sp(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a function in a subprocess and return its value.  This is for achieving subprocess
    isolation, not parallelism.  The subprocess is configured so things like logging work
    correctly, and is initialized with a derived random seed.
    """
    ctx = mp.get_context("spawn")
    rq = ctx.SimpleQueue()
    seed = seedbank.derive_seed()
    worker_args = (log_queue(ctx), seed, rq, func, args, kwargs)
    _log.debug("spawning subprocess to run %s", func)
    proc = ctx.Process(target=_sp_worker, args=worker_args)
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

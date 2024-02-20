import logging
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Concatenate, Iterator, cast

import seedbank

from parinvoke._worker import initialize_mp_worker, mp_invoke_worker
from parinvoke.config import ParallelConfig
from parinvoke.invoker import ModelOpInvoker, R, T
from parinvoke.logging import log_queue
from parinvoke.sharing import PersistedModel, persist

_log = logging.getLogger(__name__)


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
            initialize_mp_worker,
            (key, func_pkl, kid_tc, log_queue(ctx), seedbank.root_seed()),
        )

    def map(self, *iterables: Any) -> Iterator[R]:
        return cast(Iterator[R], self.executor.map(mp_invoke_worker, *iterables))

    def shutdown(self):
        self.executor.shutdown()
        if self._close_key is not None:
            self._close_key.close()
            del self._close_key

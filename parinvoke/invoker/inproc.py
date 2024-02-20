import logging
from functools import partial
from typing import Any, Callable, Concatenate, Iterator

from parinvoke.invoker import ModelOpInvoker, P, R, T
from parinvoke.sharing import PersistedModel

_log = logging.getLogger(__name__)


class InProcessOpInvoker(ModelOpInvoker[T, R]):
    model: T | None

    def __init__(self, model: T, func: Callable[Concatenate[T, P], R]):
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

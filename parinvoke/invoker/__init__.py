from abc import ABC, abstractmethod
from typing import Any, Callable, Concatenate, Generic, Iterator, TypeVar

from parinvoke.config import ParallelConfig

T = TypeVar("T")
R = TypeVar("R")


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
        from .inproc import InProcessOpInvoker

        return InProcessOpInvoker(model, func)
    else:
        from .pool import ProcessPoolOpInvoker

        return ProcessPoolOpInvoker(model, func, n_jobs, persist_method, config)

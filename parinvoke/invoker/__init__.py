from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, ParamSpec, TypeVar

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


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

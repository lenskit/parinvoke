"""
Parallel processing contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import Traceback
from typing import Callable, Concatenate, ParamSpec, TypeVar

from parinvoke.config import ParallelConfig
from parinvoke.parallel import ModelOpInvoker
from parinvoke.sharing import PersistedModel

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class Context(ABC):
    """
    Context for parallel and subprocess invocation.  This is the primary entry
    point for parinvoke operations.

    The context manages general configuration as well as resources like shared
    memory managers.  It must be explicitly started (:meth:`setup`), and then
    closed after use (:meth:`teardown`).  It supports the context manager
    protocol to automate this process.
    """

    config: ParallelConfig

    def __init__(self, config: ParallelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def persist(model: T) -> PersistedModel[T]:
        """
        Persist a model for cross-process sharing.

        This will return a persisted model that can be used to reconstruct the model
        in a worker process (using :meth:`PersistedModel.get`).

        Args:
            model:
                The model to persist.

        Returns:
            The persisted object.
        """
        raise NotImplementedError()

    def setup(self):
        """
        Initialize the context so it is ready to run.

        .. note::
            Even though the current superclass implemenation does nothing, subclasses
            **must** call it.
        """
        pass

    def teardown(self):
        """
        Close the context, cleaning up temporary objects.

        .. note::
            Even though the current superclass implemenation does nothing, subclasses
            **must** call it.
        """
        pass

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Traceback | None):
        self.teardown()

    def invoker(
        self,
        model: T,
        func: Callable[Concatenate[T, ...], R],
        n_jobs: int | None = None,
    ) -> ModelOpInvoker[T, R]:
        """
        Get an appropriate invoker for performing oeprations on ``model``.

        Args:
            model: The model object on which to perform operations.
            func: The function to call.  The function must be pickleable.
            n_jobs:
                The number of processes to use for parallel operations.

        Returns:
            An invoker to perform operations on the model.
        """
        raise NotImplementedError()

    def run_sp(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Run a function in a subprocess and return its value.  This is for
        achieving subprocess isolation, not parallelism.  The subprocess is
        configured so things like logging work correctly, and is initialized
        with a derived random seed.
        """
        raise NotImplementedError()

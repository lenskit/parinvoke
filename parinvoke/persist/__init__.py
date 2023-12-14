"""
Sharing persistence for parallel processing models.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Literal, TypeVar

_log = logging.getLogger(__name__)

_store_state = threading.local()

T = TypeVar("T")


def _save_mode():
    return getattr(_store_state, "mode", "save")


@contextmanager
def sharing_mode():
    """
    Context manager to tell models that pickling will be used for cross-process
    sharing, not model persistence.
    """
    old = _save_mode()
    _store_state.mode = "share"
    try:
        yield
    finally:
        _store_state.mode = old


def in_share_context():
    """
    Query whether sharing mode is active.  If ``True``, we are currently in a
    :func:`sharing_mode` context, which means model pickling will be used for
    cross-process sharing.
    """
    return _save_mode() == "share"


class PersistedModel[T](ABC):
    """
    A persisted model for inter-process model sharing.

    These objects can be pickled for transmission to a worker process.

    .. note::
        Subclasses need to override the pickling protocol to implement the
        proper pickling implementation.
    """

    is_owner: bool | Literal["transfer"]
    """
    Flag indicating whether this object is the owner of the persisted model. The
    owner is expected to release the associated resources in :meth:`close`.
    """

    @abstractmethod
    def get(self) -> T:
        """
        Get the persisted model, reconstructing it if necessary.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Release the persisted model resources.  Should only be called in the
        parent process (will do nothing in a child process).
        """
        raise NotImplementedError()

    def transfer(self):
        """
        Mark an object for ownership transfer.  This object, when pickled, will
        unpickle into an owning model that frees resources when closed. Used to
        transfer ownership of shared memory resources from child processes to
        parent processes.  Such an object should only be unpickled once.

        The default implementation sets the ``is_owner`` attribute to ``'transfer'``.

        Returns:
            ``self`` (for convenience)
        """
        if not self.is_owner:
            warnings.warn("non-owning objects should not be transferred", stacklevel=1)
        else:
            self.is_owner = "transfer"
        return self


def persist[T](
    model: T, *, method: str | Callable[[T], PersistedModel[T]] | None = None
) -> PersistedModel[T]:
    """
    Persist a model for cross-process sharing.

    This will return a persisted model that can be used to reconstruct the model
    in a worker process (using :meth:`PersistedModel.get`).

    If no method is provided, this function automatically selects a model persistence
    strategy from the the following, in order:

    1. If `LK_TEMP_DIR` is set, use :mod:`binpickle` in shareable mode to save
       the object into the LensKit temporary directory.
    2. If :mod:`multiprocessing.shared_memory` is available, use :mod:`pickle`
       to save the model, placing the buffers into shared memory blocks.
    3. Otherwise, use :mod:`binpickle` in shareable mode to save the object
       into the system temporary directory.

    Args:
        model(obj):
            The model to persist.
        method(str or None):
            The method to use.  Can be one of ``binpickle`` or ``shm``.

    Returns:
        PersistedModel: The persisted object.
    """
    persist: Callable[[T], PersistedModel[T]] | None = None
    if method is not None:
        if method == "binpickle":
            persist = persist_binpickle
        elif method == "shm":
            persist = persist_shm
        elif isinstance(method, Callable):
            persist = method
        else:
            raise ValueError("invalid method %s: must be one of binpickle, shm, or a function")

    if persist is None:
        if SHM_AVAILABLE and "LK_TEMP_DIR" not in os.environ:
            persist = persist_shm
        else:
            persist = persist_binpickle

    return persist(model)


from .binpickle import BPKPersisted, persist_binpickle  # noqa: E402,F401
from .shm import SHM_AVAILABLE, SHMPersisted, persist_shm  # noqa: E402,F401

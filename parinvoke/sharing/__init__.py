# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Sharing persistence for parallel processing models.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)


class PersistedModel(ABC, Generic[T]):
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

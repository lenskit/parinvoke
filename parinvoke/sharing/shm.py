# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import logging
import multiprocessing.shared_memory as shm
import pickle
import sys
from multiprocessing.managers import SharedMemoryManager
from typing import Any, NamedTuple, Optional, TypeVar

from parinvoke.config import InvokeConfig
from parinvoke.context import InvokeContext

from . import PersistedModel
from ._sharedpickle import SharedPicklerMixin

# we have encountered a number of bugs on Windows
SHM_AVAILABLE = sys.platform != "win32"

_log = logging.getLogger(__name__)
T = TypeVar("T")


class SharedPickler(pickle.Pickler, SharedPicklerMixin):
    pass


class SHMBlock(NamedTuple):
    start: int
    end: int


def persist_shm(model: T) -> SHMPersisted[T]:
    """
    Persist a model using :mod:`multiprocessing.shared_memory`.

    Args:
        model: The model to persist.

    Returns:
        PersistedModel: The persisted object.
    """

    buffers: list[pickle.PickleBuffer] = []

    out = io.BytesIO()
    pickler = SharedPickler(out, 5, buffer_callback=buffers.append)
    pickler.dump(model)
    data = out.getvalue()

    total_size = sum(memoryview(b).nbytes for b in buffers)
    _log.info(
        "serialized %s to %d pickle bytes with %d buffers of %d bytes",
        model,
        len(data),
        len(buffers),
        total_size,
    )

    if buffers:
        # blit the buffers to the SHM block
        _log.debug("preparing to share %d buffers", len(buffers))
        memory = shm.SharedMemory(create=True, size=total_size)
        cur_offset = 0
        blocks: list[SHMBlock] = []
        for i, buf in enumerate(buffers):
            ba = buf.raw()
            blen = ba.nbytes
            bend = cur_offset + blen
            _log.debug("saving %d bytes in buffer %d/%d", blen, i + 1, len(buffers))
            memory.buf[cur_offset:bend] = ba
            blocks.append(SHMBlock(cur_offset, bend))
            cur_offset = bend
    else:
        memory = None
        blocks = []

    return SHMPersisted[T](data, memory, blocks)


class SHMContext(InvokeContext):
    """
    BinPickle context using shared memory.
    """

    owner: bool = False
    manager: SharedMemoryManager

    def __init__(self, config: InvokeConfig | None = None) -> None:
        if config is None:
            config = InvokeConfig.default()
        super().__init__(config)

        self.owner = True
        self.manager = SharedMemoryManager()

    def setup(self):
        if not self.owner:  # no cover
            raise RuntimeError("cannot set up worker-process context")
        super().setup()
        self.manager.start()

    def teardown(self):
        if not self.owner:  # no cover
            raise RuntimeError("cannot tear down worker-process context")
        super().teardown()
        self.manager.shutdown()

    def __getstate__(self):
        return {
            "config": self.config,
            "@mgr_address": self.manager.address,
        }

    def __setstate__(self, state: dict[str, Any]):
        self.config = state["config"]
        self.manager = SharedMemoryManager(state["@mgr_address"])

    def persist(self, model: T) -> SHMPersisted[T]:
        buffers: list[pickle.PickleBuffer] = []

        out = io.BytesIO()
        pickler = SharedPickler(out, 5, buffer_callback=buffers.append)
        pickler.dump(model)
        data = out.getvalue()

        total_size = sum(memoryview(b).nbytes for b in buffers)
        _log.info(
            "serialized %s to %d pickle bytes with %d buffers of %d bytes",
            model,
            len(data),
            len(buffers),
            total_size,
        )

        if buffers:
            # blit the buffers to the SHM block
            _log.debug("preparing to share %d buffers", len(buffers))
            memory = self.manager.SharedMemory(size=total_size)
            cur_offset = 0
            blocks: list[SHMBlock] = []
            for i, buf in enumerate(buffers):
                ba = buf.raw()
                blen = ba.nbytes
                bend = cur_offset + blen
                _log.debug("saving %d bytes in buffer %d/%d", blen, i + 1, len(buffers))
                memory.buf[cur_offset:bend] = ba
                blocks.append(SHMBlock(cur_offset, bend))
                cur_offset = bend
        else:
            memory = None
            blocks = []

        return SHMPersisted[T](data, memory, blocks)


class SHMPersisted(PersistedModel[T]):
    pickle_data: bytes
    blocks: list[SHMBlock]
    memory: Optional[shm.SharedMemory] = None
    _model: Optional[T] = None

    def __init__(self, data: bytes, memory: shm.SharedMemory | None, blocks: list[SHMBlock]):
        self.pickle_data = data
        self.blocks = blocks
        self.memory = memory
        self.is_owner = True

    def get(self):
        if self._model is None:
            _log.debug("loading model from shared memory")
            buffers: list[memoryview] = []
            for bs, be in self.blocks:
                assert self.memory is not None, "persisted object with blocks has no shared memory"
                buffers.append(self.memory.buf[bs:be])

            self._model = pickle.loads(self.pickle_data, buffers=buffers)

        return self._model

    def close(self, unlink: bool = True):
        self._model = None

        _log.debug("releasing SHM buffers")
        if self.memory is not None:
            self.memory.close()
            if unlink and self.is_owner and self.is_owner != "transfer":
                self.memory.unlink()
                self.is_owner = False
            self.memory = None

    def __getstate__(self):
        return {
            "pickle_data": self.pickle_data,
            "blocks": self.blocks,
            "memory": self.memory,
            "is_owner": True if self.is_owner == "transfer" else False,
        }

    def __del__(self):
        self.close(False)

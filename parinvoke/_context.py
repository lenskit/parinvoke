# type: ignore
from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Mapping
from multiprocessing import SimpleQueue
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any


class PIContext(SpawnContext):
    def SimpleQueue(self) -> FastQ:
        return FastQ(ctx=self)


def _p5_recv(self: Connection) -> object:
    buf = self.recv_bytes()
    return pickle.loads(buf)


def _p5_send(self: Connection, obj: object):
    buf = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    self.send_bytes(buf)


class FastQ(SimpleQueue):
    """
    SimpleQueue subclass that uses Pickle5 instead of default pickling.
    """

    def __init__(self, *, ctx: SpawnContext | None):
        super().__init__(ctx=ctx)
        self.__patch()

    def __patch(self):
        # monkey-patch the sockets to use pickle5
        self._reader.recv = _p5_recv.__get__(self._reader)
        self._writer.send = _p5_send.__get__(self._writer)

    def get(self):
        with self._rlock:
            res = self._reader.recv_bytes()
        return pickle.loads(res)

    def put(self, obj):
        bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        # follow SimpleQueue, need to deal with _wlock being None
        if self._wlock is None:
            self._writer.send_bytes(bytes)
        else:
            with self._wlock:
                self._writer.send_bytes(bytes)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__patch()


PIContext.INSTANCE = PIContext()

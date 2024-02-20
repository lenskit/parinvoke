# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, TypeVar, cast

import binpickle

from parinvoke.config import InvokeConfig

from ..context import InvokeContext
from . import PersistedModel
from ._sharedpickle import SharedPicklerMixin

_log = logging.getLogger(__name__)
T = TypeVar("T")


class SharingBinPickler(binpickle.BinPickler, SharedPicklerMixin):
    pass


def persist_binpickle(
    model: T, dir: str | Path | None = None, file: str | Path | None = None
) -> BPKPersisted[T]:
    """
    Persist a model using binpickle.

    Args:
        model: The model to persist.
        dir: The temporary directory for persisting the model object.
        file: The file in which to save the object.

    Returns:
        PersistedModel: The persisted object.
    """
    if file is not None:
        path = Path(file)
    else:
        if dir is None:
            dir = os.environ.get("LK_TEMP_DIR", None)
        fd, path = tempfile.mkstemp(suffix=".bpk", prefix="lkpy-", dir=dir)
        os.close(fd)
        path = Path(path)

    _log.debug("persisting %s to %s", model, path)
    with SharingBinPickler.mappable(path) as bp:
        bp.dump(model)
    return BPKPersisted[T](path)


class BPKContext(InvokeContext):
    dir: Path | None

    def __init__(self, dir: str | Path | None = None, config: InvokeConfig | None = None) -> None:
        if config is None:
            config = InvokeConfig.default()
        super().__init__(config)

        if dir is None:
            td = config.env_var("TEMP_DIR")
            if td is not None:
                name, dir = td
                _log.debug("configured from $%s: %s", name, dir)

        self.dir = Path(dir) if dir else None

    def persist(self, model: T) -> BPKPersisted[T]:
        fd, path = tempfile.mkstemp(suffix=".bpk", prefix="lkpy-", dir=self.dir)
        os.close(fd)
        path = Path(path)

        _log.debug("persisting %s to %s", model, path)
        with SharingBinPickler.mappable(path) as bp:
            bp.dump(model)
        return BPKPersisted[T](path)

    def teardown(self):
        super().teardown()
        if not self.dir:
            return

        for file in self.dir.glob("*.bpk"):
            _log.debug("removing %s", file)
            file.unlink()

        self.dir.unlink()


class BPKPersisted(PersistedModel[T]):
    path: Path
    _bpk_file: Optional[binpickle.BinPickleFile]
    _model: Optional[T]

    def __init__(self, path: Path):
        self.path = path
        self.is_owner = True
        self._bpk_file = None
        self._model = None

    def get(self) -> T:
        if self._bpk_file is None:
            _log.debug("loading %s", self.path)
            self._bpk_file = binpickle.BinPickleFile(self.path, direct=True)
            self._model = cast(T, self._bpk_file.load())
        assert self._model is not None
        return self._model

    def close(self, unlink: bool = True):
        if self._bpk_file is not None:
            self._model = None
            try:
                _log.debug("closing BPK file")
                try:
                    self._bpk_file.close()
                except BufferError:
                    _log.debug("could not close %s, collecting garbage and retrying", self.path)
                    gc.collect()
                    self._bpk_file.close()
            except (BufferError, IOError) as e:
                _log.warn("error closing %s: %s", self.path, e)
            self._bpk_file = None

        if self.is_owner and unlink:
            assert self._model is None
            if unlink:
                _log.debug("deleting %s", self.path)
                try:
                    self.path.unlink()
                except IOError as e:
                    _log.warn("could not remove %s: %s", self.path, e)
            self.is_owner = False

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_bpk_file"] = None
        d["_model"] = None
        if self.is_owner == "transfer":
            d["is_owner"] = True
        else:
            d["is_owner"] = False
        return d

    def __del___(self):
        self.close(False)

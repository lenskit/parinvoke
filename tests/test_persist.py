# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pytest import mark

from parinvoke import sharing
from parinvoke.sharing._sharedpickle import SharedPicklerMixin
from parinvoke.util import set_env_var


class TestSharable:
    array: NDArray[np.float64]
    transpose: NDArray[np.float64]
    flipped: bool = False

    def __init__(self, array: NDArray[np.float64]):
        self.array = array
        self.transpose = array.T

    def _shared_getstate(self):
        return {"array": self.array}

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        if "transpose" not in state:
            self.transpose = self.array.T
            self.flipped = True


def test_sharing_mode():
    "Ensure sharing mode decorator turns on sharing"
    assert not sharing.in_share_context()

    with sharing.sharing_mode():
        assert sharing.in_share_context()

    assert not sharing.in_share_context()


def test_persist_bpk():
    matrix = np.random.randn(1000, 100)
    share = sharing.persist_binpickle(matrix)
    try:
        assert share.path.exists()
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_shared_getstate():
    mat = np.random.randn(100, 50)
    tso = TestSharable(mat)

    # quick make sure base serialization works
    native = pickle.dumps(tso, pickle.HIGHEST_PROTOCOL)
    nres = pickle.loads(native)

    assert nres.array is not tso.array
    assert np.all(nres.array == tso.array)
    assert np.all(nres.transpose == tso.transpose)
    assert not nres.flipped

    out = io.BytesIO()
    pickler = SharedPicklerMixin(out, pickle.HIGHEST_PROTOCOL)
    pickler.dump(tso)

    out = out.getvalue()

    res = pickle.loads(out)

    assert res.array is not tso.array
    assert np.all(res.array == tso.array)
    assert np.all(res.transpose == tso.transpose)
    assert res.flipped


@mark.skipif(not sharing.SHM_AVAILABLE, reason="shared_memory not available")
def test_persist_shm():
    matrix = np.random.randn(1000, 100)
    share = sharing.persist_shm(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist():
    "Test default persistence"
    matrix = np.random.randn(1000, 100)
    share = sharing.persist(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_dir(tmp_path: Path):
    "Test persistence with a configured directory"
    matrix = np.random.randn(1000, 100)
    with set_env_var("LK_TEMP_DIR", os.fspath(tmp_path)):
        share = sharing.persist(matrix)
        assert isinstance(share, sharing.binpickle.BPKPersisted)

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_method():
    "Test persistence with a specified method"
    matrix = np.random.randn(1000, 100)

    share = sharing.persist(matrix, method="binpickle")
    assert isinstance(share, sharing.binpickle.BPKPersisted)

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()

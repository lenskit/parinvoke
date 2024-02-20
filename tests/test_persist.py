# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pytest import fixture, skip

from parinvoke.sharing.binpickle import BPKContext
from parinvoke.sharing.shm import SHM_AVAILABLE, SharedPickler, SHMContext


@fixture
def shm_context():
    if not SHM_AVAILABLE:
        skip("shared memory not available")

    with SHMContext() as shm:
        yield shm


@fixture
def bpk_context():
    with BPKContext() as bpk:
        yield bpk


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
    pickler = SharedPickler(out, pickle.HIGHEST_PROTOCOL)
    pickler.dump(tso)

    out = out.getvalue()

    res = pickle.loads(out)

    assert res.array is not tso.array
    assert np.all(res.array == tso.array)
    assert np.all(res.transpose == tso.transpose)
    assert res.flipped


def test_persist_bpk(bpk_context: BPKContext):
    matrix = np.random.randn(1000, 100)
    share = bpk_context.persist(matrix)
    try:
        assert share.path.exists()
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_shm(shm_context: SHMContext):
    matrix = np.random.randn(1000, 100)
    share = shm_context.persist(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()

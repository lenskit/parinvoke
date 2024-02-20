# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import multiprocessing as mp
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from pytest import approx, fixture, mark  # type: ignore

from parinvoke import InvokeContext, is_mp_worker, is_worker

_log = logging.getLogger(__name__)


@fixture
def ctx():
    with InvokeContext.default() as ctx:
        yield ctx


def _mul_op(m: npt.NDArray[np.float64], v: npt.NDArray[np.float64]):
    return m @ v


def _worker_status(blob: str, *args: Any):
    _log.info("in worker %s", mp.current_process().name)
    return os.getpid(), is_worker(), is_mp_worker()


@mark.parametrize("n_jobs", [None, 1, 2, 8])
def test_invoke_matrix(ctx: InvokeContext, n_jobs: int | None):
    matrix = np.random.randn(100, 100)
    vectors = [np.random.randn(100) for _i in range(100)]
    with ctx.invoker(matrix, _mul_op, n_jobs) as inv:
        mults = inv.map(vectors)
        for rv, v in zip(mults, vectors):
            act_rv = matrix @ v
            assert act_rv == approx(rv, abs=1.0e-6)


def test_mp_is_worker(ctx: InvokeContext):
    with ctx.invoker("foo", _worker_status, 2) as loop:
        res = list(loop.map(range(10)))
        assert all([w for (_pid, w, _mpw) in res])
        assert all([mpw for (_pid, _w, mpw) in res])

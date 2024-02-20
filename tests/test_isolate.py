# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# type: ignore
import logging
import multiprocessing as mp
import os

import numpy as np
from seedbank import root_seed

import pytest
from pytest import fixture, raises

from parinvoke import InvokeContext, is_mp_worker, is_worker
from parinvoke.sharing.binpickle import BPKContext
from parinvoke.sharing.shm import SHM_AVAILABLE, SHMContext

_log = logging.getLogger(__name__)

_static_isw = is_worker()
_static_is_mpw = is_mp_worker()


@fixture
def ctx():
    with InvokeContext.default() as ctx:
        yield ctx


def _static_worker_status(_msg):
    return os.getpid(), _static_isw, _static_is_mpw


def _worker_status(blob, *args):
    _log.info("in worker %s", mp.current_process().name)
    return os.getpid(), is_worker(), is_mp_worker()


def _sp_matmul(a1, a2, *, fail=False):
    _log.info("in worker process")
    if fail:
        raise RuntimeError("you rang?")
    else:
        return a1 @ a2


def _sp_matmul_p(a1, a2, *, fail=False):
    _log.info("in worker process")
    ctx = InvokeContext.current()
    return ctx.persist(a1 @ a2).transfer()


def test_run_sp(ctx: InvokeContext):
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    res = ctx.run_sp(_sp_matmul, a1, a2)
    assert np.all(res == a1 @ a2)


def test_run_sp_fail(ctx: InvokeContext):
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    with raises(ChildProcessError):
        ctx.run_sp(_sp_matmul, a1, a2, fail=True)


@pytest.mark.parametrize("method", [None, "binpickle", "shm"])
def test_run_sp_persist(method):
    if method == "shm" and not SHM_AVAILABLE:
        pytest.skip("SHM backend not available")

    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    if method is None:
        ctx = InvokeContext.default()
    elif method == "shm":
        ctx = SHMContext()
    elif method == "binpickle":
        ctx = BPKContext()

    with ctx:
        res = ctx.run_sp(_sp_matmul_p, a1, a2)
        try:
            assert res.is_owner
            assert np.all(res.get() == a1 @ a2)
        finally:
            res.close()


def test_sp_is_worker(ctx: InvokeContext):
    pid, w, mpw = ctx.run_sp(_worker_status, "fishtank")
    assert pid != os.getpid()
    assert w
    assert not mpw


def test_sp_is_worker_static(ctx: InvokeContext):
    pid, w, mpw = ctx.run_sp(_static_worker_status, "fishtank")
    assert pid != os.getpid()
    assert w
    assert not mpw


def _get_seed():
    return root_seed()


def test_sp_random_seed(ctx: InvokeContext):
    init = root_seed()
    seed = ctx.run_sp(_get_seed)
    # we should spawn a seed for the worker
    assert seed.entropy == init.entropy
    assert seed.spawn_key == (init.n_children_spawned - 1,)

# type: ignore
import logging
import multiprocessing as mp
import os

import numpy as np
from seedbank import root_seed

import pytest
from pytest import raises

from parinvoke import is_mp_worker, is_worker
from parinvoke.isolate import run_sp
from parinvoke.persist import SHM_AVAILABLE, persist

_log = logging.getLogger(__name__)


def _worker_status(blob, *args):
    _log.info("in worker %s", mp.current_process().name)
    return os.getpid(), is_worker(), is_mp_worker()


def _sp_matmul(a1, a2, *, fail=False):
    _log.info("in worker process")
    if fail:
        raise RuntimeError("you rang?")
    else:
        return a1 @ a2


def _sp_matmul_p(a1, a2, *, method=None, fail=False):
    _log.info("in worker process")
    return persist(a1 @ a2, method=method).transfer()


def test_run_sp():
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    res = run_sp(_sp_matmul, a1, a2)
    assert np.all(res == a1 @ a2)


def test_run_sp_fail():
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    with raises(ChildProcessError):
        run_sp(_sp_matmul, a1, a2, fail=True)


@pytest.mark.parametrize("method", [None, "binpickle", "shm"])
def test_run_sp_persist(method):
    if method == "shm" and not SHM_AVAILABLE:
        pytest.skip("SHM backend not available")

    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    res = run_sp(_sp_matmul_p, a1, a2, method=method)
    try:
        assert res.is_owner
        assert np.all(res.get() == a1 @ a2)
    finally:
        res.close()


def test_sp_is_worker():
    pid, w, mpw = run_sp(_worker_status, "fishtank")
    assert pid != os.getpid()
    assert w
    assert not mpw


def _get_seed():
    return root_seed()


def test_sp_random_seed():
    init = root_seed()
    seed = run_sp(_get_seed)
    # we should spawn a seed for the worker
    assert seed.entropy == init.entropy
    assert seed.spawn_key == (init.n_children_spawned - 1,)

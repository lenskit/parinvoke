import logging
import multiprocessing as mp
import os

import numpy as np
import pytest
from pytest import approx, mark, raises

_log = logging.getLogger(__name__)


def _mul_op(m, v):
    return m @ v


def _worker_status(blob, *args):
    _log.info("in worker %s", mp.current_process().name)
    return os.getpid(), is_worker(), is_mp_worker()


@mark.parametrize("n_jobs", [None, 1, 2, 8])
def test_invoke_matrix(n_jobs):
    matrix = np.random.randn(100, 100)
    vectors = [np.random.randn(100) for i in range(100)]
    with invoker(matrix, _mul_op, n_jobs) as inv:
        mults = inv.map(vectors)
        for rv, v in zip(mults, vectors):
            act_rv = matrix @ v
            assert act_rv == approx(rv, abs=1.0e-6)


def test_mp_is_worker():
    with invoker("foo", _worker_status, 2) as loop:
        res = list(loop.map(range(10)))
        assert all([w for (pid, w, mpw) in res])
        assert all([mpw for (pid, w, mpw) in res])

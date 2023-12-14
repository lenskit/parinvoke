import os

import numpy as np

from parinvoke import persist
from parinvoke.util import set_env_var

from pytest import mark


def test_sharing_mode():
    "Ensure sharing mode decorator turns on sharing"
    assert not persist.in_share_context()

    with persist.sharing_mode():
        assert persist.in_share_context()

    assert not persist.in_share_context()


def test_persist_bpk():
    matrix = np.random.randn(1000, 100)
    share = persist.persist_binpickle(matrix)
    try:
        assert share.path.exists()
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


@mark.skipif(not persist.SHM_AVAILABLE, reason="shared_memory not available")
def test_persist_shm():
    matrix = np.random.randn(1000, 100)
    share = persist.persist_shm(matrix)
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
    share = persist.persist(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_dir(tmp_path):
    "Test persistence with a configured directory"
    matrix = np.random.randn(1000, 100)
    with set_env_var("LK_TEMP_DIR", os.fspath(tmp_path)):
        share = persist.persist(matrix)
        assert isinstance(share, persist.BPKPersisted)

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

    share = persist.persist(matrix, method="binpickle")
    assert isinstance(share, persist.BPKPersisted)

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()

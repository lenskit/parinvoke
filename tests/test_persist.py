# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import numpy as np

from pytest import mark

from parinvoke import sharing
from parinvoke.util import set_env_var


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

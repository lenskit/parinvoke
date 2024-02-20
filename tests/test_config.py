# This file is part of parinvoke.
# Copyright (C) 2020-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import multiprocessing as mp

from parinvoke.config import InvokeConfig
from parinvoke.util import set_env_var


def test_proc_count_default():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = InvokeConfig()
        assert cfg.proc_count() == mp.cpu_count()
        assert cfg.proc_count(level=1) == 1


def test_proc_count_div():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = InvokeConfig(core_div=2)
        assert cfg.proc_count() == mp.cpu_count() // 2
        assert cfg.proc_count(level=1) == 2


def test_proc_count_env():
    with set_env_var("PARINVOKE_NUM_PROCS", "17"):
        cfg = InvokeConfig()
        assert cfg.proc_count() == 17
        assert cfg.proc_count(level=1) == 1


def test_proc_count_max():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = InvokeConfig(max_default=1)
        assert cfg.proc_count() == 1


def test_proc_count_nest_env():
    with set_env_var("PARINVOKE_NUM_PROCS", "7,3"):
        cfg = InvokeConfig()
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1


def test_proc_count_nest_env_prefix():
    with set_env_var("TEST_NUM_PROCS", "7,3"):
        cfg = InvokeConfig(prefix="TEST")
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1


def test_proc_count_nest_env_prefix_fallback():
    with set_env_var("PARINVOKE_NUM_PROCS", "7,3"):
        cfg = InvokeConfig(prefix="TEST")
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1

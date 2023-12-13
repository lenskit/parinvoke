import multiprocessing as mp
import os
from contextlib import contextmanager

from parinvoke.config import ParallelConfig


@contextmanager
def set_env_var(var, val):
    "Set an environment variable & restore it."
    is_set = var in os.environ
    old_val = None
    if is_set:
        old_val = os.environ[var]
    try:
        if val is None:
            if is_set:
                del os.environ[var]
        else:
            os.environ[var] = val
        yield
    finally:
        if is_set:
            os.environ[var] = old_val
        elif val is not None:
            del os.environ[var]


def test_proc_count_default():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = ParallelConfig()
        assert cfg.proc_count() == mp.cpu_count()
        assert cfg.proc_count(level=1) == 1


def test_proc_count_div():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = ParallelConfig(core_div=2)
        assert cfg.proc_count() == mp.cpu_count() // 2
        assert cfg.proc_count(level=1) == 2


def test_proc_count_env():
    with set_env_var("PARINVOKE_NUM_PROCS", "17"):
        cfg = ParallelConfig()
        assert cfg.proc_count() == 17
        assert cfg.proc_count(level=1) == 1


def test_proc_count_max():
    with set_env_var("PARINVOKE_NUM_PROCS", None):
        cfg = ParallelConfig(max_default=1)
        assert cfg.proc_count() == 1


def test_proc_count_nest_env():
    with set_env_var("PARINVOKE_NUM_PROCS", "7,3"):
        cfg = ParallelConfig()
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1


def test_proc_count_nest_env_prefix():
    with set_env_var("TEST_NUM_PROCS", "7,3"):
        cfg = ParallelConfig(prefix="TEST")
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1


def test_proc_count_nest_env_prefix_fallback():
    with set_env_var("PARINVOKE_NUM_PROCS", "7,3"):
        cfg = ParallelConfig(prefix="TEST")
        assert cfg.proc_count() == 7
        assert cfg.proc_count(level=1) == 3
        assert cfg.proc_count(level=2) == 1

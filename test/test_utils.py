"""Test cases for utility functions."""
import tempfile as tmp
import pytest
import numpy as np
from sdnet.utils import run_simulations


_tempdir = tmp.mkdtemp()

def _func(N, lo, hi):
    return np.random.randint(lo, hi, (N, N))


@pytest.mark.parametrize('params', [
    ({'N': 10, 'lo': 0, 'hi': 10}, ),
    ({'N': 100, 'lo': 0, 'hi': 20}, )
])
@pytest.mark.parametrize('n,n_jobs,use_persistence', [
    (5, 2, True), (5, 4, False)
])
def test_run_simulations(params, n, n_jobs, use_persistence):
    results = run_simulations(_func, params, n=n, n_jobs=4, cachedir=_tempdir,
                              use_persistence=use_persistence)
    assert isinstance(results, list)
    assert results

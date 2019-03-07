"""Test cases for utility functions."""
import pytest
import numpy as np
from sdnet.utils import run_simulations, get_walk2


def _func(N, lo, hi):
    return np.random.randint(lo, hi, (N, N))


@pytest.mark.parametrize('params', [[(10, 0, 10, ), (100, 0, 20, )]])
@pytest.mark.parametrize('n,n_jobs', [(5, 2), (5, 4)])
def test_run_simulations(params, n, n_jobs):
    results = run_simulations(_func, params, n=n, n_jobs=n_jobs)
    assert isinstance(results, list)
    assert len(results) == len(params) * n

def test_get_walk2():
    N = 10
    A = np.random.randint(0, 2, (N, N))
    A2 = A@A
    for i in range(N):
        assert np.array_equal(get_walk2(A, i), A2[i, :])

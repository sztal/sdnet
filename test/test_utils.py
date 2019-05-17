"""Test cases for utility functions."""
import pytest
from pytest import approx
import numpy as np
from sdnet.utils import run_simulations, get_walk2
from sdnet.utils import get_distance_least_upper_bound
from sdnet.utils import Lp_norm, Lp_dist, manhattan_dist
from sdnet.utils import make_dist_matrix, transitivity_local_undirected


def _func(N, lo, hi):
    return np.random.randint(lo, hi, (N, N))


@pytest.mark.parametrize('params', [[(10, 0, 10, ), (100, 0, 20, )]])
@pytest.mark.parametrize('n,n_jobs', [(5, 2), (5, 4)])
def test_run_simulations(params, n, n_jobs):
    results = run_simulations(_func, params, n=n, n_jobs=n_jobs)
    assert isinstance(results, list)
    assert len(results) == len(params) * n

@pytest.mark.parametrize('X', [
    np.random.uniform(0, 10, (20, 1)),
    np.random.uniform(0, 10, (20, 2))
])
def test_make_dist_matrix(X):
    D = make_dist_matrix(X, manhattan_dist)
    assert D.dtype == np.float
    assert D.shape == (X.shape[0], X.shape[0])
    assert not (D == 0).all()

def test_get_walk2():
    N = 10
    A = np.random.randint(0, 2, (N, N))
    A2 = A@A
    for i in range(N):
        assert np.array_equal(get_walk2(A, i), A2[i, :])

@pytest.mark.parametrize('n_edges,expected', [(2, 2), (4, 4)])
def test_get_distance_least_upper_bound(n_edges, expected):
    X = np.array([
        [0, 1, 2],
        [3, 0, 4],
        [5, 6, 0]
    ])
    output = get_distance_least_upper_bound(X, n_edges)
    assert output == expected

@pytest.mark.parametrize('u', [np.array([-1, 1, 2])])
@pytest.mark.parametrize('p,expected', [(1, 4), (2, 6**(1/2)), (3, 10**(1/3))])
def test_Lp_norm(u, p, expected):
    output = Lp_norm(u, p)
    assert output == approx(expected)

@pytest.mark.parametrize('u,v', [(np.array([-1,1,2]), np.array([1,2,2]))])
@pytest.mark.parametrize('p,expected', [(1, 3), (2, 5**(1/2)), (3, 9**(1/3))])
def test_Lp_dist(u, v, p, expected):
    output = Lp_dist(u, v, p=p, normalized=False)
    output_norm = Lp_dist(u, v, p=p, normalized=True)
    assert output == approx(expected)
    u_norm = u / Lp_norm(u, p=p)
    v_norm = v / Lp_norm(v, p=p)
    assert output_norm == approx(Lp_dist(u_norm, v_norm, p=p, normalized=False))

@pytest.mark.parametrize('u,v,expected', [
    (np.array([1]), np.array([3]), 2.0),
    (np.array([-1,1,2]), np.array([1,2,2]), 3.0)
])
def test_manhattan_dist(u, v, expected):
    output = manhattan_dist(u, v)
    assert output == approx(expected)

@pytest.mark.parametrize('A,expected', [
    (np.array([[0,1,0,0,1],[1,0,0,1,1],[0,0,0,0,1],[0,1,0,0,1],[1,1,1,1,0]]),
     np.array([1.0, 2/3, np.nan, 1.0, 1/3]))
])
@pytest.mark.parametrize('average', [False, True])
def test_transitivity_local_undirected(A, average, expected):
    output = transitivity_local_undirected(A, average=average)
    if average:
        assert output.mean() == expected[~np.isnan(expected)].mean()
    else:
        np.testing.assert_equal(output, expected)

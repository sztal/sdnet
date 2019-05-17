"""Test network models."""
import pytest
import numpy as np
from numpy.random import uniform
from numba import njit
from sdnet.networks import random_network, make_adjacency_matrix
from sdnet.networks import get_edgelist, rewire_edges
from sdnet.utils import make_dist_matrix


@njit
def _measure(x, y):
    exp = (x + np.fabs(y)).sum()
    return 2**-exp

@pytest.mark.parametrize('N,p,k', [
    (1000, None, 15),
    (1000, 15/999, None),
])
@pytest.mark.parametrize('directed', [ True, False ])
def test_random_network(N, p, k, directed):
    np.random.seed(1010)
    X = random_network(N, p, k, directed=directed)
    assert abs(X.sum(axis=1).mean() - 15) < 1
    assert X.shape == (N, N)


@pytest.mark.parametrize('X', [np.array([[1, 0], [0, -1], [1, 1]])])
@pytest.mark.parametrize('symmetric', [True, False])
def test_make_dist_matrix(X, symmetric):
    P = make_dist_matrix(X, _measure, symmetric=symmetric)
    if symmetric:
        assert np.array_equal(P, P.T)
    else:
        assert not np.array_equal(P, P.T)


@pytest.mark.parametrize('P', [uniform(0, 1, (250, 250))])
@pytest.mark.parametrize('directed', [True, False])
def test_make_adjacency_matrix(P, directed):
    np.random.seed(303)
    A = make_adjacency_matrix(P, directed)
    if not directed:
        assert np.array_equal(A, A.T)

@pytest.mark.parametrize('directed', [False, True])
def test_get_edgelist(adj_matrix, directed):
    A = adj_matrix
    E = get_edgelist(A, directed=directed)
    A_edges = set((i, j) for i, j in zip(*np.nonzero(A)))
    E_edges = set((i, j) for i, j in zip(E[:, 0], E[:, 1]))
    assert A_edges == E_edges
    if not directed:
        for i in range(0, E.shape[0], 2):
            i1, j1, u1 = E[i]
            i2, j2, u2 = E[i+1]
            assert i1 == j2
            assert j1 == i2
            assert u1 == u2 + 1

@pytest.mark.parametrize('_A', [
    np.array([[0,1,0],[1,0,0],[0,0,0]]),
    make_adjacency_matrix(np.random.uniform(0, .1, (250, 250)))
])
@pytest.mark.parametrize('p', [0, 1])
@pytest.mark.parametrize('directed', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rewire_edges(_A, p, directed, copy):
    A = _A.copy()
    A0 = A
    A  = rewire_edges(A, p=p, directed=directed, copy=copy)
    assert not copy or A0 is not A
    assert p == 1 or np.array_equal(A, _A)
    assert p == 0 or not np.array_equal(A, _A)
    assert directed or np.array_equal(A, A.T)

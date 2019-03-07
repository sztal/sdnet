"""Test network models."""
import pytest
import numpy as np
from numpy.random import uniform
from numba import njit
from sdnet.networks import random_network
from sdnet.networks import generate_adjacency_matrix
from sdnet.networks import random_geometric_graph_nb as random_geometric_graph


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
def test_random_geometric_graph(X, symmetric):
    P = random_geometric_graph(X, _measure, symmetric=symmetric)
    if symmetric:
        assert np.array_equal(P, P.T)
    else:
        assert not np.array_equal(P, P.T)


@pytest.mark.parametrize('P', [uniform(0, 1, (250, 250))])
@pytest.mark.parametrize('directed', [True, False])
def test_generate_adjacency_matrix(P, directed):
    np.random.seed(303)
    A = generate_adjacency_matrix(P, directed)
    if not directed:
        assert np.array_equal(A, A.T)

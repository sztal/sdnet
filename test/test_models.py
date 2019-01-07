"""Tests for `sdnet` module."""
# pylint: disable=E0611
# pylint: disable=R0914
import pytest
import numpy as np
from numpy.random import uniform
import networkx as nx
from networkx.algorithms import clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms import average_shortest_path_length
from sdnet.networks import random_network
from sdnet.networks import stochastic_block_model, generate_adjacency_matrix


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
def test_stochastic_block_model(X, symmetric):
    P = stochastic_block_model(X, _measure, symmetric=symmetric)
    if symmetric:
        assert np.array_equal(P, P.T)
    else:
        assert not np.array_equal(P, P.T)


@pytest.mark.parametrize('P', [uniform(0, 1, (250, 250))])
@pytest.mark.parametrize('directed', [True, False])
def test_generate_adjacency_matrix(P, directed):
    np.random.seed(303)
    A = generate_adjacency_matrix(P, directed)
    if directed:
        assert not np.array_equal(A, A.T)
    else:
        assert np.array_equal(A, A.T)

@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
class TestSegregationProcess:

    @pytest.mark.parametrize('nsteps', [1, 2, 4])
    def test_sp_d2_uniform(self, sp_d2_uniform, nsteps):
        sp = sp_d2_uniform
        A = sp.A
        A0 = A.copy()
        h0 = sp.h
        sp.run(nsteps)
        assert A.sum() == A0.sum()
        assert A.shape == A0.shape
        assert not np.array_equal(sp.A.sum(axis=1), A0.sum(axis=1))
        assert not np.array_equal(A, A0)
        assert sp.h < h0

    @pytest.mark.parametrize('nsteps', [1, 2, 4])
    def test_spc_d2_uniform(self, spc_d2_uniform, nsteps):
        sp = spc_d2_uniform
        A = sp.A
        A0 = A.copy()
        h0 = sp.h
        sp.run(nsteps)
        assert A.sum() == A0.sum()
        assert A.shape == A0.shape
        assert not np.array_equal(sp.D, A0.sum(axis=1))
        assert not np.array_equal(A, A0)
        assert sp.h < h0
        assert A.sum(axis=1).max() > A0.sum(axis=1).max()

    @pytest.mark.slow
    def test_spc_d2_uniform_graph(self, spc_d2_uniform):
        sp = spc_d2_uniform
        A = sp.A
        A0 = A.copy()
        G0 = nx.from_numpy_matrix(A0)
        clust0 = sum(clustering(G0).values()) / sp.n_nodes
        deg0 = degree_assortativity_coefficient(G0)
        comp0 = next(G0.subgraph(c) for c in nx.connected_components(G0))
        avg0 = average_shortest_path_length(comp0)
        sp.run(50)
        G = nx.from_numpy_matrix(A)
        clust = sum(clustering(G).values()) / sp.n_nodes
        deg = degree_assortativity_coefficient(G)
        comp = next(G.subgraph(c) for c in nx.connected_components(G))
        avg = average_shortest_path_length(comp)
        assert clust > clust0
        assert deg > deg0
        assert avg/avg0 <= 1.5 or avg/avg0 >= 0.5

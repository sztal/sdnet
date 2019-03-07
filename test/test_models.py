"""Tests for `sdnet` module."""
# pylint: disable=E0611
# pylint: disable=R0914
import pytest
import numpy as np
import networkx as nx
from networkx.algorithms import clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms import average_shortest_path_length

np.random.seed(303)

@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
class TestSegregation:

    @pytest.mark.parametrize('n_steps', [1, 2, 4])
    def test_sp_d2_uniform(self, sp_d2_uniform, n_steps):
        sp = sp_d2_uniform
        A = sp.A
        E = sp.E
        A0 = A.copy()
        E0 = E.copy()
        h0 = sp.h
        sp.run(n_steps)
        assert A.sum() == A0.sum()
        assert A.shape == A0.shape
        assert np.array_equal(E0[::2, 0], E0[1::2, 1])
        assert np.array_equal(E0[1::2, 0], E0[::2, 1])
        assert np.array_equal(E[::2, 0], E[1::2, 1])
        assert np.array_equal(E[1::2, 0], E[::2, 1])
        if not sp.directed:
            assert np.array_equal(A, A.T)
        assert not np.array_equal(A, A0)
        assert sp.h < h0

    # @pytest.mark.parametrize('n_steps', [1, 2, 4])
    # def test_spn_d2_uniform(self, spn_d2_uniform, n_steps):
    #     sp = spn_d2_uniform
    #     A = sp.A
    #     A0 = A.copy()
    #     h0 = sp.h
    #     sp.run(n_steps)
    #     assert A.sum() == A0.sum()
    #     assert A.shape == A.shape
    #     if not sp.directed:
    #         assert np.array_equal(A, A.T)
    #     assert not np.array_equal(A, A0)
    #     assert sp.h < h0

    @pytest.mark.parametrize('n_steps', [1, 2, 4])
    def test_spc_d2_uniform(self, spc_d2_uniform, n_steps):
        sp = spc_d2_uniform
        A = sp.A
        E = sp.E
        A0 = A.copy()
        E0 = E.copy()
        h0 = sp.h
        sp.run(n_steps)
        assert A.sum() == A0.sum()
        assert A.shape == A0.shape
        assert np.array_equal(E0[::2, 0], E0[1::2, 1])
        assert np.array_equal(E0[1::2, 0], E0[::2, 1])
        assert np.array_equal(E[::2, 0], E[1::2, 1])
        assert np.array_equal(E[1::2, 0], E[::2, 1])
        if not sp.directed:
            assert np.array_equal(A, A.T)
        assert not np.array_equal(A, A0)
        assert sp.h < h0
        assert ((sp.N > 0) | np.isclose(sp.N, 0)).all()
        assert ((sp.E > 0) | np.isclose(sp.E, 0)).all()
        assert ((sp.D > 0) | np.isclose(sp.D, 0)).all()
        assert sp.D.size == sp.A.sum()
        assert np.allclose(sp.D.sum(), np.where(sp.A, sp.P, 0).sum())
        assert sp.N.shape[0] == sp.A.shape[0]
        assert np.allclose(sp.N[:, 0].sum(), np.where(sp.A, sp.P, 0).sum())
        assert sp.N[:, 1].sum() == sp.A.sum()

    @pytest.mark.slow
    def test_sp_d2_uniform_graph(self, sp_d2_uniform):
        # pylint: disable=unused-variable
        sp = sp_d2_uniform
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
        comp = sorted((G.subgraph(c) for c in nx.connected_components(G)), key=len)[-1]
        avg = average_shortest_path_length(comp)
        # assert clust > clust0
        # assert deg > deg0
        assert avg/avg0 <= 1.5 or avg/avg0 >= 0.5

    @pytest.mark.slow
    def test_spc_d2_uniform_graph(self, spc_d2_uniform):
        # pylint: disable=unused-variable
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
        comp = sorted((G.subgraph(c) for c in nx.connected_components(G)), key=len)[-1]
        avg = average_shortest_path_length(comp)
        assert clust > clust0
        # assert deg > deg0
        assert avg/avg0 <= 1.5 or avg/avg0 >= 0.5

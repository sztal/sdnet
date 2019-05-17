"""Tests for `sdnet` module."""
# pylint: disable=E0611
# pylint: disable=R0914
from collections import OrderedDict
import pytest
import numpy as np
from sdnet import SDA
from sdnet.utils import make_dist_matrix, manhattan_dist


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
@pytest.mark.filterwarnings("ignore:overflow encountered")
class TestSDA:

    @pytest.mark.parametrize('k', [30, 60])
    @pytest.mark.parametrize('alpha', [8, 16])
    @pytest.mark.parametrize('directed', [False, True])
    def test_from_dist_matrix(self, dist_matrix, k, alpha, directed):
        sda = SDA.from_dist_matrix(dist_matrix, k, alpha, directed=directed)
        D = dist_matrix
        P = sda.P
        A = sda.adjacency_matrix(sparse=False)
        assert P.shape == D.shape
        assert (abs(sda.P - 1) <= 1).all()
        assert abs(P.sum(axis=1).mean() - k) <= .1
        assert not np.array_equal(D, D.T) or np.array_equal(P, P.T)
        assert P[np.nonzero(A)].mean() > P.mean()
        assert A.shape == D.shape
        assert np.isin(A, (0, 1)).all()
        assert directed or np.array_equal(A, A.T)
        assert isinstance(sda.to_dict(), OrderedDict)

    @pytest.mark.parametrize('k', [30, 60])
    @pytest.mark.parametrize('alpha', [1, 8, 16])
    @pytest.mark.parametrize('directed', [False, True])
    def test_sda_on_lognormal_data(self, d2_lognormal, k, alpha, directed):
        D = make_dist_matrix(d2_lognormal, manhattan_dist)
        sda = SDA.from_dist_matrix(D, k, alpha, directed=directed)
        P = sda.P
        A = sda.adjacency_matrix(sparse=False)
        assert P.shape == D.shape
        assert (abs(sda.P - 1) <= 1).all()
        assert abs(P.sum(axis=1).mean() - k) <= .5
        assert not np.array_equal(D, D.T) or np.array_equal(P, P.T)
        assert P[np.nonzero(A)].mean() > P.mean()
        assert A.shape == D.shape
        assert np.isin(A, (0, 1)).all()
        assert directed or np.array_equal(A, A.T)
        assert isinstance(sda.to_dict(), OrderedDict)

    @pytest.mark.parametrize('k', [30, 60])
    @pytest.mark.parametrize('alpha', [4, 8])
    @pytest.mark.parametrize('weights', [None, (20, 30)])
    @pytest.mark.parametrize('directed', [False, True])
    def test_from_weighted_dist_matrices(self, k, alpha, weights, directed):
        dm = [
            make_dist_matrix(np.random.uniform(0, 1, (250, 2)), manhattan_dist),
            make_dist_matrix(np.random.normal(100, 15, (250, 10)), manhattan_dist)
        ]
        sda = SDA.from_weighted_dist_matrices(k, alpha, dm, weights,
                                              directed=directed)
        P = sda.P
        assert abs(P.sum(axis=1).mean() - k) <= .1
        assert ((P >= 0) | (P <= 1)).all()

    @pytest.mark.parametrize('k', [30, 60])
    @pytest.mark.parametrize('alpha', [1, 4, 8])
    @pytest.mark.parametrize('directed', [False, True])
    @pytest.mark.parametrize('sort', [False, True])
    @pytest.mark.parametrize('simplify', [False, True])
    def test_conf_model(self, dist_matrix, k, alpha, directed, sort, simplify):
        D = dist_matrix
        sda = SDA.from_dist_matrix(D, k, alpha, p_rewire=0, directed=directed)
        P = sda.P.copy()
        degseq = np.random.negative_binomial(1, 1/(1+k), (sda.N,))
        if directed:
            degseq = np.ceil(degseq/2).astype(int)
            degseq = np.column_stack((degseq, degseq))
            degseq[0, 0] += 1
            degseq[-1, 1] += 1
        if not directed and degseq.sum() % 2 != 0:
            degseq[np.random.choice(degseq.size)] += 1
        sda.set_degseq(degseq, sort=sort)
        A = sda.conf_model(sparse=False, simplify=simplify)
        if simplify:
            assert (np.diag(A) == 0).all()
            assert np.isin(A, (0, 1)).all()
        elif directed:
            assert np.array_equal(sda.degseq[:, 0], A.sum(axis=1))
            assert np.array_equal(sda.degseq[:, 1], A.sum(axis=0))
        else:
            assert np.array_equal(sda.degseq, A.sum(axis=1))
        assert directed or np.array_equal(A, A.T)
        assert np.array_equal(P, sda.P)
        assert P[np.nonzero(A)].mean() > P.mean()

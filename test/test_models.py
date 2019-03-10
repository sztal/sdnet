"""Tests for `sdnet` module."""
# pylint: disable=E0611
# pylint: disable=R0914
from collections import OrderedDict
import pytest
import numpy as np
from sdnet import SESNetwork

np.random.seed(303)

@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
class TestSegregation:

    @pytest.mark.parametrize('pa_exp', [0, 1])
    @pytest.mark.parametrize('p_local', [0, 1])
    @pytest.mark.parametrize('p_flow', [0, 1])
    def test_ses_d2_uniform(self, adj_matrix, dist_matrix, pa_exp, p_local, p_flow):
        A = adj_matrix
        D = dist_matrix
        sp = SESNetwork(A, D, pa_exp=pa_exp, p_local=p_local, p_flow=p_flow)
        E = sp.E
        A0 = A.copy()
        E0 = E.copy()
        h0 = sp.homogeneity
        sp.run()
        assert A.sum() == A0.sum()
        assert A.shape == A0.shape
        if not sp.directed:
            assert np.array_equal(A, A.T)
        assert not np.array_equal(A, A0)
        assert np.array_equal(E0[::2, 0], E0[1::2, 1])
        assert np.array_equal(E0[1::2, 0], E0[::2, 1])
        assert np.array_equal(E[::2, 0], E[1::2, 1])
        assert np.array_equal(E[1::2, 0], E[::2, 1])
        assert (sp.E >= 0).all()
        assert sp.E.shape[0] == A.sum()
        assert (sp.K >= 0).all()
        assert sp.homogeneity < h0
        assert isinstance(sp.to_dict(), OrderedDict)

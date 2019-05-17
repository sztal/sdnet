"""Test parallel processing with models."""
# pylint: disable=W0612
from time import time
import pytest
import numpy as np
from joblib import Parallel, delayed
from sdnet import SDA
from sdnet.utils import manhattan_dist, make_dist_matrix

ALPHA = [1, 2, 4, 16]

def run_job(alpha):
    np.random.seed(999)
    X = np.random.uniform(0, 1, (250, 2))
    D = make_dist_matrix(X, manhattan_dist, symmetric=True)
    sda = SDA.from_dist_matrix(D, k=30, alpha=alpha, directed=False)
    return sda

@pytest.mark.slow
@pytest.mark.parametrize('n_jobs', [1, 2, 4])
def test_scp_parallel(n_jobs):
    start = time()
    results = Parallel(n_jobs=n_jobs)(delayed(run_job)(a) for a in ALPHA)
    end = time()
    elapsed = end - start

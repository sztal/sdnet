"""Test parallel processing with models."""
# pylint: disable=W0612
from time import time
import pytest
import numpy as np
from joblib import Parallel, delayed
from sdnet.models import SegregationWithClustering
from sdnet.networks import random_network

EXPONENTS = [0, 1/4, 1/2, 3/4, 1]

def run_job(pe_exponent):
    np.random.seed(999)
    X = np.random.uniform(0, 1, (250, 2))
    A = random_network(250, k=10, directed=False)
    sp = SegregationWithClustering(A, X, directed=False, pa_exponent=pe_exponent)
    sp.run(50)
    return sp.A, sp.hseries

@pytest.mark.slow
@pytest.mark.parametrize('n_jobs', [1, 2, 4])
def test_scp_parallel(n_jobs):
    start = time()
    results = Parallel(n_jobs=n_jobs)(delayed(run_job)(pa) for pa in EXPONENTS)
    end = time()
    elapsed = end - start

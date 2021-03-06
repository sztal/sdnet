"""*PyTest* configuration and general purpose fixtures."""
# pylint: disable=W0621
import pytest
import numpy as np
from sdnet.networks import random_network
from sdnet.utils import euclidean_dist, make_dist_matrix


def pytest_addoption(parser):
    """Custom `pytest` command-line options."""
    parser.addoption(
        '--benchmarks', action='store_true', default=False,
        help="Run benchmarks (instead of tests)."
    )
    parser.addoption(
        '--slow', action='store_true', default=False,
        help="Run slow tests / benchmarks."""
    )

def pytest_collection_modifyitems(config, items):
    """Modify test runner behaviour based on `pytest` settings."""
    run_benchmarks = config.getoption('--benchmarks')
    run_slow = config.getoption('--slow')
    if run_benchmarks:
        skip_test = \
            pytest.mark.skip(reason="Only benchmarks are run with --benchmarks")
        for item in items:
            if 'benchmark' not in item.keywords:
                item.add_marker(skip_test)
    else:
        skip_benchmark = \
            pytest.mark.skip(reason="Benchmarks are run only with --run-benchmark")
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_benchmark)
    if not run_slow:
        skip_slow = pytest.mark.skip(reason="Slow tests are run only with --slow")
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)


# Fixtures --------------------------------------------------------------------

K = 30
N_NODES = 100
RANDOM_SEED = 423423

@pytest.fixture(scope='session')
def d2_uniform():
    np.random.seed(RANDOM_SEED)
    X = np.random.uniform(0, 1, (N_NODES, 2))
    return X

@pytest.fixture(scope='session')
def d2_lognormal():
    np.random.seed(RANDOM_SEED)
    X = np.random.lognormal(10, 2, (N_NODES, 1))
    return X

@pytest.fixture(scope='session')
def dist_matrix(d2_uniform):
    X = d2_uniform
    return make_dist_matrix(X, euclidean_dist, symmetric=True)

@pytest.fixture(scope='function')
def adj_matrix():
    np.random.seed(RANDOM_SEED)
    A = random_network(N_NODES, k=K, directed=False)
    return A

"""*PyTest* configuration and general purpose fixtures."""
# pylint: disable=W0621
import pytest
import numpy as np
from sdnet.networks import random_network, distance_matrix_nb
from sdnet.utils import norm_manhattan_dist


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

N_NODES = 250

@pytest.fixture(scope='session')
def d2_uniform():
    np.random.seed(999)
    X = np.random.uniform(0, 1, (N_NODES, 2))
    return X

@pytest.fixture(scope='session')
def dist_matrix(d2_uniform):
    X = d2_uniform
    return distance_matrix_nb(X, norm_manhattan_dist, symmetric=True)

@pytest.fixture(scope='function')
def adj_matrix():
    np.random.seed(999)
    A = random_network(N_NODES, k=15, directed=False)
    return A

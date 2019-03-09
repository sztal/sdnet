"""Simulation related functions and utilities."""
from itertools import repeat, chain
from joblib import Parallel, delayed
import numpy as np
from numba import njit


@njit
def norm_manhattan_dist(u, v):
    return np.abs(u - v).mean()


def run_simulations(func, params, n=1, n_jobs=4, out_func=None):
    """Run in parallel.

    Parameters
    ----------
    func : callable
        Function that takes parameters as inputs.
    params : iterable
        Iterable of parameters' values.
        They are passed to the `func` as ``*args``.
    n : int
        How many repetition for every combination of parameters.
    n_jobs : int
        Number of parallel jobs to run.
    out_func : callable or None
        Optional function for processing output.
    """
    pars = chain.from_iterable(repeat(params, n))
    results = Parallel(n_jobs=n_jobs)(delayed(func)(*p) for p in pars)
    if out_func is not None:
        results = out_func(results)
    return results

def get_distance_least_upper_bound(P, n_edges):
    """Get least upper bound for distances in a dataset.

    Parameters
    ----------
    P : (N, N) array_like
        Distance matrix.
    n_edges : int
        Number of edges.
    """
    dists = np.hstack((
        P[np.triu_indices_from(P, k=1)],
        P[np.tril_indices_from(P, k=-1)]
    ))
    dists.sort()
    least_upper_dist = dists[n_edges - 1]
    return least_upper_dist

def get_walk2(A, i):
    """Get length 2 walks for a given node in an adjacency matrix.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    i : int
        Index of a node.
    """
    return A[np.nonzero(A[i, :])].sum(axis=0)

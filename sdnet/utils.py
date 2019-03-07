"""Simulation related functions and utilities."""
from itertools import repeat, chain
from joblib import Parallel, delayed
import numpy as np
from numba import njit


@njit
def norm_manhattan_dist(u, v):
    return np.abs(u - v).mean()

@njit
def inverse_distance(u, v, alpha=1):
    return 1 / np.sqrt(np.sum((u-v)**2))**(-alpha)


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

def get_sorted_dists(P):
    """Get sorted distances from a distance matrix.

    Parameters
    ----------
    P : (N, N) array_like
        Distance matrix.
    """
    dists = np.hstack((
        P[np.triu_indices_from(P, k=1)],
        P[np.tril_indices_from(P, k=-1)]
    ))
    dists.sort()
    return dists

def get_distance_least_upper_bound(P, n_edges):
    """Get least upper bound for distances in a dataset.

    Parameters
    ----------
    P : (N, N) array_like
        Distance matrix.
    n_edges : int
        Number of edges.
    """
    dists = get_sorted_dists(P)
    least_upper_dist = dists[n_edges].max()
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

def normalize_minmax(X, copy=True):
    """Min-Max normalize a data array.

    Parameters
    ----------
    X : (N, k) array_like
        A data array.
    copy : bool
        Shoul copy be created.
    """
    if copy:
        X = X.copy()
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X

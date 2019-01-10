"""Simulation related functions and utilities."""
import os
from hashlib import md5
import tempfile as tmp
from itertools import repeat, chain
import joblib
from joblib import Parallel, delayed
import numpy as np
from numba import njit


@njit
def norm_manhattan_dist(u, v):
    return np.abs(u - v).mean()


def run_simulations(func, params, n=1, n_jobs=4, out_func=None,
                    cachedir=None, use_persistence=True):
    """Run simulation experiments.

    Parameters
    ----------
    func : callable
        Function that takes parameters as inputs.
    params : iterable
        Iterable of parameters' values.
        They are passed to the `func` as ``kwargs``.
    n : int
        How many repetition for every combination of parameters.
    n_jobs : int
        Number of parallel jobs to run.
    out_func : callable or None
        Optional function for processing output.
    cachedir : str
        Path to the cache directory.
    use_persistence : bool
        Should persistence be used.
        Runs are identified by hashes of parameters.
    """
    params = list(chain.from_iterable(repeat(params, n)))
    key = ''.join(str(x) for x in params)
    m = md5()
    m.update(key.encode())
    key = m.hexdigest()

    if use_persistence:
        cachedir = tmp.mkdtemp() if cachedir is None else cachedir
        filepath = os.path.join(cachedir, key)

    if use_persistence and os.path.exists(filepath):
        results = joblib.load(filepath)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(func)(**kw) for kw in params)
        if use_persistence:
            if not os.path.exists(cachedir):
                os.makedirs(cachedir, exist_ok=True)
            joblib.dump(results, filepath)

    if out_func is not None:
        results = out_func(results)
    return results


def estimate_maximium_homophily(P, n_edges):
    """Estimate maximum homophily for a dataset.

    Parameters
    ----------
    P : (N, N) array_like
        Distance matrix.
    n_edges : int
        Number of edges.
    """
    avg_dist = P.mean()
    dists = np.hstack((
        P[np.triu_indices_from(P, k=1)],
        P[np.tril_indices_from(P, k=-1)]
    ))
    dists.sort()
    least_upper_dist = dists[n_edges].max()
    homophily = least_upper_dist / avg_dist
    return homophily

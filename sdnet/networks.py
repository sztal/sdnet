"""Simple network models and related utilities."""
import numpy as np
from numpy.random import random, uniform
from numba import njit


@njit
def _rng_undirected_nb(X, p):
    for i in range(X.shape[0]):
        for j in range(i):
            if random() <= p:
                X[i, j] = X[j, i] = 1
    return X

def random_network(N, p=None, k=None, directed=False):
    """Generate a random network.

    Parameters
    ----------
    N : int
        Number of nodes.
    p : float
        Edge formation probability.
        Should be set to ``None`` if `k` is used.
    k : float
        Average node degree.
        Should be set to ``None`` if `p` is used.
    directed : bool
        Should network be directed.

    Notes
    -----
    `p` or `k` (but not both) must be not ``None``.

    Returns
    -------
    (N, N) array_like
        Adjacency matrix of a graph.
    """
    if p is None and k is None:
        raise TypeError("Either 'p' or 'k' must be used")
    elif p is not None and k is not None:
        raise TypeError("'p' and 'k' can not be used at the same time")
    elif k is not None:
        if k > N-1:
            raise ValueError(f"average degree of {k:.4} can not be attained with {N} nodes")
        p = k / (N-1)
    if directed:
        X = np.where(uniform(0, 1, (N, N)) <= p, 1, 0)
        np.fill_diagonal(X, 0)
    else:
        X = np.zeros((N, N), dtype=int)
        X = _rng_undirected_nb(X, p)
    return X


def random_geometric_graph(X, measure, symmetric=True):
    """Generate a random geometric graph.

    Parameters
    ----------
    X : array_like (N, k)
        Dataset with nodes' features.
        One row is one node.
    measure : callable
        Measure function that takes two main arguments which are
        feature vectors for two nodes.
    symmetric : bool
        Is the measure function symmetric in the two main arguments.

    Returns
    -------
    (N, N) array_like
        Edge formation probability matrix.
    """
    N = X.shape[0]
    P = np.zeros((N, N))
    if symmetric:
        for i in range(N):
            for j in range(i):
                P[i, j] = P[j, i] = measure(X[i], X[j])
    else:
        for i in range(N):
            for j in range(N):
                P[i, j] = measure(X[i], X[j])
    return P

random_geometric_graph_nb = njit(random_geometric_graph)


@njit
def _gen_am_undirected_nb(P, A):
    for i in range(A.shape[0]):
        for j in range(i):
            if random() <= P[i, j]:
                A[i, j] = A[j, i] = 1
    return A

def generate_adjacency_matrix(P, directed=False):
    """Generate adjacency matrix from edge formation probabilities.

    Parameters
    ----------
    P : (N, N) array_like
        Edge formation probability matrix.
    directed : bool
        Should network be directed.
    """
    if directed:
        A = np.where(uniform(0, 1, P.shape) <= P, 1, 0)
        A = A.astype(int)
        np.fill_diagonal(A, 0)
    else:
        A = np.zeros_like(P, dtype=int)
        A = _gen_am_undirected_nb(P, A)
    return A

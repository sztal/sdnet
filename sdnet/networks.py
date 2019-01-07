"""Simple network models and related utilities."""
from random import sample, uniform
import numpy as np


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
        X = np.where(np.random.random_sample((N, N)) <= p, 1, 0)
        np.fill_diagonal(X, 0)
    else:
        X = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(i):
                if np.random.random_sample() <= p:
                    X[i, j] = X[j, i] = 1
    return X


def stochastic_block_model(X, measure, symmetric=False, **kwds):
    """Generate a network based on a generalized stochastic block model.

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
    **kwds :
        Keyword arguments passed to the measure function.

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
                P[i, j] = P[j, i] = measure(X[i], X[j], **kwds)
    else:
        for i in range(N):
            for j in range(N):
                P[i, j] = measure(X[i], X[j], **kwds)
    return P


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
        A = np.where(np.random.random_sample(P.shape) <= P, 1, 0)
        A = A.astype(int)
        np.fill_diagonal(A, 0)
    else:
        N = P.shape[0]
        A = np.zeros_like(P, dtype=int)
        for i in range(N):
            for j in range(i):
                if np.random.random_sample() <= P[i, j]:
                    A[i, j] = A[j, i] = 1
    return A

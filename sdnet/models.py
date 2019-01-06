"""Process and network models."""
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


class SegregationProcess:
    """Segregation process.

    Attributes
    ----------
    A : (N, N) array_like
        Adjacency matrix.
    X : (N, k) array_like
        Nodes' features dataset.
    directed : bool
        Is the network directed.
    D : (N,) array_like
        Node degree vector.
    H : (N,) array_like
        Heterogeneity vector.
    DM : (N, N) array_like
        Distance matrix.
    m : int
        Number of vertices.
    k : int
        Number of edges.
    t : int
        Number of simulation steps already run.
        A simulation step corresponds to `k` rewiring attempts.
    """
    def __init__(self, A, X, directed=False):
        """Initialization method."""
        if A.shape[0] != X.shape[0]:
            raise AttributeError("A and X must have the same number of rows")
        self.A = A
        self.X = X
        self.E = set(tuple(x) for x in np.argwhere(A))
        self.DM = None
        self._compute_distance_matrix()
        self.k = len(self.E)
        self.directed = directed
        self.t = 0
        self.hseries = [self.H.mean()]
        self._steps = 0

    @property
    def m(self):
        return self.A.shape[0]

    @property
    def D(self):
        return self.A.sum(axis=1)

    @property
    def H(self):
        Hvec = self.DM.sum(axis=1) / (self.A != 0).sum(axis=1)
        Hvec[np.isnan(Hvec)] = 0
        return Hvec

    def _compute_distance_matrix(self):
        dm = np.zeros_like(self.A, dtype=float)
        for i, j in self.E:
            dm[i, j] = self.dist(self.X[i, :], self.X[j, :])
        self.DM = dm

    def remove_edge(self, i, j):
        """Remove an edge and perform related operations."""
        self.E.remove((i, j))
        self.A[i, j] = 0
        self.DM[i, j] = 0
        if not self.directed:
            self.E.remove((j, i))
            self.A[j, i] = 0
            self.DM[j, i] = 0

    def add_edge(self, i, j):
        """Add an edge and perform related operations."""
        d = self.dist(self.X[i, :], self.X[j, :])
        if d < 0 or d > 1:
            raise ValueError("distance values must be between 0 and 1")
        self.E.add((i, j))
        self.A[i, j] = 1
        self.DM[i, j] = d
        if not self.directed:
            self.E.add((j, i))
            self.A[j, i] = 1
            self.DM[j, i] = d

    def dist(self, u, v):
        """Distance function.

        Default distance function is a mean
        of absolute difference between the feature vectors
        and assumes that the features range from 0 to 1.
        This ensures that distances also always range from 0 to 1,
        so when multiplied with the `p` parameters
        it yields a proper probability distribution.

        Parameters
        ----------
        u : (k,) array_like
            Feature vector of the first node.
        v : (k,) array_like
            Feature vector of the second node.

        Returns
        -------
        float
            Distance. Must be between 0 and 1.
        """
        return np.abs(u - v).mean()

    def select_node(self, i):
        """Select new adjacent node."""
        return sample([
            j for j in range(self.m)
            if self.A[i, j] == 0 and i != j
        ], 1).pop()

    def rewire(self):
        """Do edge rewiring."""
        i, j = sample(self.E, 1).pop()
        d = self.DM[i, j]
        if uniform(0, 1) <= d:
            self.remove_edge(i, j)
            i = sample((i, j), 1).pop()
            j = self.select_node(i)
            self.add_edge(i, j)
        self._steps += 1
        if self._steps == self.k:
            self._steps = 0
            self.t += 1
            self.hseries.append(self.H.mean())

    def has_converged(self):
        """Has the process converged."""
        return False

    def run(self, nsteps=None):
        """Run segregation process.

        Parameters
        ----------
        nsteps : int or None
            Run segregation for `nsteps` steps.
            A step correspond to the number of rewiring attempts
            equal to the number of edges in the network.
            If ``None`` then stops when a condition
            defined in the :py:meth:`has_converged` is met.
        """
        if nsteps is None:
            while not self.has_converged():
                self.rewire()
        else:
            t0 = self.t
            while self.t < t0 + nsteps:
                self.rewire()


class SegregationWithClustering(SegregationProcess):
    """Segregation process with enhanced clustering.

    Enhanced clustering is induced by additional rules
    for edge rewiring. New neighbours may be selected only
    from friends of friends. Moreover, selection probability
    is proportional to node degrees following the rule
    of preferential attachment.
    """
    def __init__(self, A, X, pa_exponent=1, directed=False):
        """Initialization method.

        pa_exponent : float
            Exponent for the preferential attachment stage.
        """
        super().__init__(A, X, directed=directed)
        self.pa_exponent = pa_exponent
        self.A2 = A@A

    def remove_edge(self, i, j):
        super().remove_edge(i, j)
        self.A2[i, :] -= self.A[j, :]
        if not self.directed:
            self.A2[j, :] -= self.A[i, :]

    def add_edge(self, i, j):
        super().add_edge(i, j)
        self.A2[i, :] += self.A[j, :]
        if not self.directed:
            self.A2[j, :] += self.A[i, :]

    def select_node(self, i):
        nodes = self.A2[i, :] * np.where(self.A[i, :] == 0, 1, 0)
        nodes[i] = 0
        nodes = np.nonzero(nodes)[0]
        if nodes.size == 0:
            return super().select_node(i)
        degrees = np.take(self.D, nodes)**self.pa_exponent
        j = np.random.choice(nodes, 1, p=degrees / degrees.sum())[0]
        return j

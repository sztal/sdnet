"""Segregation driven models."""
import numpy as np
from numpy.random import choice, uniform


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
    h : float
        Average edge length (distance).
    n_nodes : int
        Number of nodes.
    n_edges : int
        Number of edges.
    nsteps : int
        Number of simulation steps already run.
        A simulation step corresponds to `k` rewiring attempts.
    """
    def __init__(self, A, X, directed=False):
        """Initialization method."""
        self.A = A
        X = X - X.min(axis=0)
        self.X = X / X.max(axis=0)
        self.E = np.argwhere(A)
        self.V = np.zeros((self.E.shape[0],), dtype=float)
        for k in range(self.E.shape[0]):
            i, j = self.E[k]
            self.V[k] = self.dist(self.X[i], self.X[j])
        self.directed = directed
        self.hseries = [self.V.mean()]
        self.nsteps = 0
        self._niter = 0

    @property
    def n_edges(self):
        return self.E.shape[0]

    @property
    def n_nodes(self):
        return self.A.shape[0]

    @property
    def h(self):
        return self.hseries[-1]

    def dist(self, u, v):
        """Distance function."""
        return np.abs(u - v).mean()

    def remove_edge(self, i, j):
        """Remove an edge."""
        self.A[i, j] = 0
        if not self.directed:
            self.A[j, i] = 0

    def add_edge(self, k, l, i, j):
        """Add an edge."""
        d = self.dist(self.X[i, :], self.X[j, :])
        self.A[i, j] = 1
        self.E[k, :] = i, j
        self.V[k] = d
        if not self.directed:
            self.A[j, i] = 1
            self.E[l, :] = j, i
            self.V[l] = d

    def select_node(self, i):
        """Select a new adjacent node."""
        choices = np.arange(self.n_nodes)[self.A[i, :] == 0]
        choices = choices[choices != i]
        return choice(choices)

    def rewire(self):
        """Do edge rewiring."""
        k = choice(self.E.shape[0])
        i, j = self.E[k, :]
        l = np.where((self.E[:, 0] == j) & (self.E[:, 1] == i))[0][0]
        d = self.V[k]
        if uniform() <= d:
            self.remove_edge(i, j)
            i = choice((i, j))
            j = self.select_node(i)
            self.add_edge(k, l, i, j)
        self._niter += 1
        if self._niter % self.n_edges == 0:
            self.nsteps += 1
            self.hseries.append(self.V.mean())

    def has_converged(self):
        """Has the process converged."""
        return False

    def run(self, n):
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
        n0 = self.nsteps
        while self.nsteps < n0 + n:
            self.rewire()


class SegregationWithClustering(SegregationProcess):
    """Segregation process with enhanced clustering.

    Enhanced clustering is induced by additional rules
    for edge rewiring. New neighbours may be selected only
    from friends of friends. Moreover, selection probability
    is proportional to node degrees following the rule
    of preferential attachment.

    Attributes
    ----------
    pa_exponent : float
        Exponent for the preferential attachment stage.
    """
    def __init__(self, A, X, directed=False, pa_exponent=3/4):
        """Initialization method."""
        super().__init__(A, X, directed=directed)
        self.pa_exponent = pa_exponent
        self.A2 = A@A
        self.D = A.sum(axis=1)

    def remove_edge(self, i, j):
        super().remove_edge(i, j)
        self.A2[i, :] -= self.A[j, :]
        self.D[i] -= 1
        if not self.directed:
            self.A2[j, :] -= self.A[i, :]
            self.D[j] -= 1

    def add_edge(self, k, l, i, j):
        super().add_edge(k, l, i, j)
        self.A2[i, :] += self.A[j, :]
        self.D[i] += 1
        if not self.directed:
            self.A2[j, :] += self.A[i, :]
            self.D[j] += 1

    def select_node(self, i):
        nodes = self.A2[i, :] * np.where(self.A[i, :] == 0, 1, 0)
        nodes[i] = 0
        nodes = np.nonzero(nodes)[0]
        if nodes.size == 0:
            return super().select_node(i)
        degrees = np.take(self.D, nodes)**self.pa_exponent
        j = np.random.choice(nodes, 1, p=degrees / degrees.sum())[0]
        return j

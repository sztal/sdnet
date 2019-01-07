"""Segregation driven models."""
from random import sample
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
        Avergae edge length (distance).
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
        self.edgeset = {
            tuple(x): self.dist(self.X[x[0]], self.X[x[1]])
            for x in np.argwhere(A)
        }
        self.directed = directed
        self.hseries = [sum(self.edgeset.values()) / self.n_edges]
        self.nsteps = 0
        self._niter = 0

    @property
    def n_edges(self):
        return len(self.edgeset)

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
        del self.edgeset[(i, j)]
        self.A[i, j] = 0
        if not self.directed and (j, i) in self.edgeset:
            del self.edgeset[(j, i)]
            self.A[j, i] = 0

    def add_edge(self, i, j):
        """Add an edge."""
        d = self.dist(self.X[i, :], self.X[j, :])
        self.edgeset[(i, j)] = d
        self.A[i, j] = 1
        if not self.directed:
            self.edgeset[(j, i)] = d
            self.A[j, i] = 1

    def select_node(self, i):
        """Select a new adjacent node."""
        choices = np.arange(self.n_nodes)[(self.A[i, :] == 0)]
        choices = choices[choices != i]
        return choice(choices)

    def rewire(self):
        """Do edge rewiring."""
        i, j = sample(self.edgeset.keys(), 1).pop()
        d = self.edgeset[(i, j)]
        if uniform() <= d:
            self.remove_edge(i, j)
            i = sample((i, j), 1).pop()
            j = self.select_node(i)
            self.add_edge(i, j)
        self._niter += 1
        if self._niter % self.n_edges == 0:
            self.nsteps += 1
            self.hseries.append(sum(self.edgeset.values()) / self.n_edges)

    # def has_converged(self):
    #     """Has the process converged."""
    #     return False

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

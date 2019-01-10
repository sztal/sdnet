"""Segregation driven models."""
# pylint: disable=R0902
from time import time
import numpy as np
from numpy.random import choice, uniform
from sdnet.utils import norm_manhattan_dist


class SegregationProcess:
    """Segregation process.

    Attributes
    ----------
    A : (N, N) array_like
        Adjacency matrix.
    X : (N, k) array_like
        Nodes' features dataset.
    P : (N, N) array_like
        Precomputed distance matrix. Optional.
    homophily : float
        Hompohily value. Must be between 0 and 1.
    initial_diversity : float
        Initial diversity.
        If ``None`` then the average edge length is taken.
        In principle, the proper value would be the average distance
        between all nodes, but this quantity is expensive to compute.
    directed : bool
        Is the network directed.
    steps_with_no_change : int
        Number of simulation steps with no change in diversity
        after which the process stops.
    h : float
        Average edge length (distance).
    n_nodes : int
        Number of nodes.
    n_edges : int
        Number of edges.
    n_steps : int
        Number of simulation steps already run.
        A simulation step corresponds to `k` rewiring attempts.
    coverged : bool
        Convergence flag.
    """
    def __init__(self, A, X, P=None, homophily=0.2,
                 directed=False, steps_with_no_change=5):
        """Initialization method."""
        self.A = A
        self.P = P
        X = X - X.min(axis=0)
        self.X = X / X.max(axis=0)
        self.directed = directed
        self.E = self._get_edgelist(self.A)
        if self.P is not None:
            self.V = self.P[np.nonzero(self.A)]
        else:
            self.V = np.zeros((self.E.shape[0],), dtype=float)
            for k in range(self.E.shape[0]):
                i, j, _ = self.E[k]
                self.V[k] = self.dist(self.X[i], self.X[j])
        self.homophily = homophily
        self.avg_cut_p = self.V.mean()
        self.steps_with_no_change = steps_with_no_change
        self.hseries = [self.avg_cut_p]
        self.n_steps = 0
        self.n_iter = 0
        self.converged = False
        self._th = (1 - self.homophily) * self.avg_cut_p

    @property
    def n_edges(self):
        return self.E.shape[0]

    @property
    def n_nodes(self):
        return self.A.shape[0]

    @property
    def n_unhappy(self):
        return self._get_unhappy().size

    @property
    def h(self):
        return self.hseries[-1]

    def _get_edgelist(self, A):
        E = np.argwhere(A)
        E = E[E.sum(axis=1).argsort()]
        sum_idx = E.sum(axis=1).reshape(E.shape[0], 1)
        max_idx = E.max(axis=1).reshape(E.shape[0], 1)
        E = np.hstack((E, max_idx, sum_idx))
        E = E[np.argsort(E[:, -1])]
        E = E[np.argsort(E[:, -2], kind='mergesort')]
        if self.directed:
            E[:, 2] = -1
        else:
            E[::2, 2] = np.arange(1, E.shape[0], 2)
            E[1::2, 2] = np.arange(0, E.shape[0], 2)
        return E[:, :3]

    def _get_unhappy(self):
        return np.where(self.V > self._th)[0]

    def dist(self, u, v):
        """Distance function."""
        return norm_manhattan_dist(u, v)

    def remove_edge(self, i, j):
        """Remove an edge."""
        self.A[i, j] = 0
        if not self.directed:
            self.A[j, i] = 0

    def add_edge(self, i, j, u, v):
        """Add an edge."""
        if self.P is not None:
            d = self.P[i, j]
        else:
            d = self.dist(self.X[i], self.X[j])
        self.A[i, j] = 1
        self.E[u, :2] = i, j
        self.V[u] = d
        if not self.directed:
            self.A[j, i] = 1
            self.E[v, :2] = j, i
            self.V[v] = d

    def select_node(self, i):
        """Select a new adjacent node."""
        choices = np.arange(self.n_nodes)[self.A[i, :] == 0]
        choices = choices[choices != i]
        return choice(choices)

    def rewire(self):
        """Do edge rewiring."""
        unhappy = self._get_unhappy()
        if unhappy.size == 0:
            self.converged = True
            return
        u = choice(unhappy)
        i, j, v = self.E[u, :]
        p = self.V[u]
        if uniform() <= p:
            self.remove_edge(i, j)
            i = choice((i, j))
            j = self.select_node(i)
            self.add_edge(i, j, u, v)
        self.n_iter += 1

    def has_converged(self):
        """Has the process converged or get stuck."""
        if self.converged:
            return True
        last_h = self.hseries[-self.steps_with_no_change:]
        return len(last_h) >= self.steps_with_no_change \
            and np.unique(last_h).size == 1

    def do_step(self):
        for _ in range(self.n_edges):
            self.rewire()
            if self.has_converged():
                return
        self.n_steps += 1
        self.hseries.append(self.V.mean())

    def run(self, n, verbose=1):
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
        for i in range(n):
            start = time()
            self.do_step()
            end = time()
            e = end - start
            if verbose > 0:
                print(f"Simulation step {i+1}/{n} finished in {e:.4} seconds ...\r", end="")
            if self.has_converged():
                return
        if verbose > 0:
            print("\nReady.")


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
    small_world_p : float
        Probability of random rewiring instead of a preferential one.
    """
    def __init__(self, A, X, P=None, homophily=0.2, directed=False,
                 pa_exponent=1, small_world_p=0.01):
        """Initialization method."""
        super().__init__(A, X, P=P, homophily=homophily, directed=directed)
        self.pa_exponent = pa_exponent
        self.small_world_p = small_world_p
        self.A2 = A@A

    def remove_edge(self, i, j):
        self.A2[i, :] -= self.A[j, :]
        self.A2[:, j] -= self.A[i, :]
        if not self.directed:
            self.A2[j, :] -= self.A[i, :]
            self.A2[:, i] -= self.A[j, :]
        super().remove_edge(i, j)

    def add_edge(self, i, j, u, v):
        super().add_edge(i, j, u, v)
        self.A2[i, :] += self.A[j, :]
        self.A2[:, j] += self.A[i, :]
        if not self.directed:
            self.A2[j, :] += self.A[i, :]
            self.A2[:, i] += self.A[j, :]

    def select_node(self, i):
        if uniform() <= self.small_world_p:
            return super().select_node(i)
        sep2 = self.A2[i, :] * np.where(self.A[i, :] == 0, 1, 0)
        sep2[i] = 0
        sep2_idx = np.nonzero(sep2)[0]
        sep2 = sep2[sep2 != 0]
        if sep2_idx.size == 0:
            return super().select_node(i)
        sep2 = sep2**self.pa_exponent
        weights = sep2
        j = choice(sep2_idx, 1, p=weights / weights.sum())[0]
        return j

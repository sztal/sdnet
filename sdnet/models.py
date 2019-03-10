"""Segregation driven models."""
# pylint: disable=R0902
from collections import OrderedDict
from time import time
from random import randint, uniform, choice as _choice
import numpy as np
from numpy.random import choice
from scipy.sparse import csr_matrix
from sdnet.utils import get_walk2, get_distance_least_upper_bound


class SESNetwork:
    """Spatially embedded segregation network model.

    Attributes
    ----------
    A : (N, N) array_like
        Adjacency matrix.
    D : (N, N) array_like
        Distance matrix.
    E : (M, 3) array_like
        Edgelist array. This is an integer array with three columns:

        1. Source node index.
        2. Target node index.
        3. Index of the dual edge in the edgelist (used only when symmetric).

    K : (N,) array_like
        Degree distribution.
    h : float
        Characteristic scale for the homophily process.
    pa_exp : float
        Preferential attachment exponent.
    p_local : float
        Probability of local instead of global selection of an adjacent node.
    p_flow : float
        Probability of flow network evolution mode instead of rewiring evolution.
    directed : bool
        Is the network directed.
    threshold : float
        Convergence threshold on average distance.
    alpha : float
        Homophily process exponent.
        In most cases it can be left with the default value,
        since it is automatically adjusted to ensure convergence
        in a reasonable time.
    alpha0 : float
        Initial value of alpha.
    alpha_adjustment_steps : int
        Number of simulation steps after which alpha is increased to prevent
        very slow convergence. No adjustement if non-positive.
    alpha_adjustment_increment : int
        Increment size for adjusting `alpha`.
    alpha_adjusted : bool
        Indicates if `alpha` was adjusted during the runtime.
    converged : bool
        Convergence flag.
    check_convergence : bool
        Should process stop after reaching convergence criterion.
        If not, then it runs for a predetermined number of steps.
    n_steps : int
        Number of Monte Carlo steps.
    n_iter : int
        Number of rewirings.
    runtime : float
        Running time of the process.
    """
    def __init__(self, A, D, h=0.4, pa_exp=0, p_local=0, p_flow=0,
                 directed=False, check_convergence=True, alpha=2,
                 alpha_adjustment_steps=5, alpha_adjustement_increment=2):
        """Initialization method."""
        self.A = A
        self.D = D
        self.E = self._get_edgelist(self.A)
        self.K = self.A.sum(axis=1)
        self.h = h
        self.alpha = alpha
        self.pa_exp = pa_exp
        self.p_local = p_local
        self.p_flow = p_flow
        self.directed = directed
        self.init_dist_mean = self.D[np.nonzero(self.A)].mean()
        self.dist_mean = self.init_dist_mean
        self.dist_lub = get_distance_least_upper_bound(self.D, self.n_edges)
        self.hseries = [self.init_dist_mean]
        self.converged = False
        self.check_convergence = check_convergence
        self.alpha_adjustement_steps = alpha_adjustment_steps
        self.alpha_adjustement_increment = alpha_adjustement_increment
        self.alpha_adjusted = False
        self.alpha0 = self.alpha
        self.n_steps = 0
        self.n_iter = 0
        self.runtime = None

    def __repr__(self):
        cn = self.__class__.__name__
        fl  = ('D' if self.directed else 'U') + \
            ('C' if self.check_convergence else '-')
        nn = self.n_nodes
        ne = self.n_edges
        h  = self.h
        a  = self.alpha
        pa = self.pa_exp
        pl = self.p_local
        pf = self.p_flow
        return f"<{cn} {fl} {nn}/{ne} h={h} alpha={a} pa_exp={pa} p_local={pl} p_flow={pf}>"

    @property
    def n_nodes(self):
        return self.A.shape[0]

    @property
    def n_edges(self):
        return self.E.shape[0]

    @property
    def homogeneity(self):
        return self.hseries[-1]

    @property
    def threshold(self):
        return (1-self.h)*(self.init_dist_mean-self.dist_lub) + self.dist_lub

    def _get_edgelist(self, A):
        E = np.argwhere(A)
        E = E[E.sum(axis=1).argsort()]
        sum_idx = E.sum(axis=1)
        max_idx = E.max(axis=1)
        max1c_idx = (E[:, 0] > E[:, 1])
        E = E[np.lexsort((max1c_idx, sum_idx, max_idx))]
        dual = np.arange(E.shape[0])
        dual[::2] += 1
        dual[1::2] -= 1
        E = np.hstack((E, dual.reshape(E.shape[0], 1)))
        return E

    def get_p(self, i, j):
        """Get edge deletion probability.

        Parameters
        ----------
        i : int
            Source node index.
        j : int
            Target node index.
        """
        d = self.D[i, j]
        r = 1 / (1 + (d/self.threshold)**self.alpha)
        return 1 - r

    def remove_edge(self, i, j):
        """Remove an edge.

        Parameters
        ----------
        i : int
            Index of the source node.
        j : int
            Index of the target node.
        p : float
            Edge deletion probability.
        """
        self.A[i, j] = 0
        self.K[i] -= 1
        self.dist_mean -= self.D[i, j] / self.n_edges
        if not self.directed:
            self.A[j, i] = 0
            self.K[j] -= 1
            self.dist_mean -= self.D[j, i] / self.n_edges

    def add_edge(self, i, j, u, v):
        """Add an edge.

        Parameters
        ----------
        i : int
            Index of the source node.
        j : int
            Index of the target node.
        u : int
            Index of the source-target edge in the edgelist array.
        v : int
            Index of the target-source edge in the edgelist array.
        """
        self.A[i, j] = 1
        self.E[u, :2] = i, j
        self.K[i] += 1
        self.dist_mean += self.D[i, j] / self.n_edges
        if not self.directed:
            self.A[j, i] = 1
            self.E[v, :2] = j, i
            self.K[j] += 1
            self.dist_mean += self.D[j, i] / self.n_edges

    def select_edge(self):
        """Select an edge to rewire."""
        u = randint(0, self.n_edges - 1)
        i, j, v = self.E[u]
        return i, j, u, v

    def select_node(self, i):
        """Select a new adjacent node.

        Parameters
        ----------
        i : int
            Index of the source node.
        """
        idx = np.where(self.A[i, :] == 0)[0]
        idx = idx[idx != i]
        if self.pa_exp != 0:
            weights = self.K[idx]**self.pa_exp
            weights = weights / weights.sum()
            return choice(idx, p=weights)
        return _choice(idx)

    def select_node_local(self, i):
        """Select a new adjacent node from local environment.

        Parameters
        ----------
        i : int
            Index of the source node.
        """
        walk2 = get_walk2(self.A, i) * np.where(self.A[i, :] == 0, 1, 0)
        walk2[i] = 0
        idx = np.nonzero(walk2)[0]
        walk2 = walk2[walk2 != 0]
        if idx.size == 0:
            return self.select_node(i)
        weights = walk2**self.pa_exp
        return choice(idx, p=weights / weights.sum())

    def rewire(self):
        """Rewire an edge."""
        i, j, u, v = self.select_edge()
        p = self.get_p(i, j)
        if uniform(0, 1) <= p:
            self.remove_edge(i, j)
        else:
            return
        if self.p_flow == 0:
            i = _choice((i, j))
        elif self.p_flow == 1 or uniform(0, 1) <= self.p_flow:
            i = randint(0, self.n_nodes - 1)
        else:
            i = _choice((i, j))
        if self.p_local == 0:
            j = self.select_node(i)
        elif self.p_local == 1 or uniform(0, 1) <= self.p_local:
            j = self.select_node_local(i)
        else:
            j = self.select_node(i)
        self.add_edge(i, j, u, v)
        if self.dist_mean <= self.threshold:
            self.converged = True

    def run_step(self):
        """Run a Monte Carlo step."""
        for _ in range(self.n_edges):
            if self.check_convergence and self.converged:
                break
            self.rewire()
            self.n_iter += 1
        self.hseries.append(self.dist_mean)

    def run(self, n=None, verbose=1):
        """Run segregation process.

        Parameters
        ----------
        n : int or None
            Run segregation for `n` Monte Carlo steps.
            Run until convergence if ``None``.
        verbose : int
            Should information about the step be printed.
        """
        if n is None and not self.check_convergence:
            raise ValueError("'n' must be defined 'check_convergence' is 'False'")
        start = time()
        i = 0
        while True:
            if self.alpha_adjustement_steps > 0 and self.n_steps > 0 \
            and self.n_steps % self.alpha_adjustement_steps == 0:
                self.alpha += self.alpha_adjustement_increment
                if not self.alpha_adjusted:
                    self.alpha_adjusted = True
            i += 1
            if (self.check_convergence and self.converged) \
            or (n is not None and i > n):
                break
            start = time()
            self.run_step()
            end = time()
            e = end - start
            if verbose > 0:
                s = i if n is None else f"{i}/{n}"
                print(f"{self} simulation step {s} finished in {e:.4} seconds ...\r", end="")
            self.n_steps += 1
        if verbose > 0:
            print("\nReady.")
        self.runtime = time() - start

    def to_dict(self, use_sparse_matrix=True):
        """Dump to (orderd) dictionary.

        Parameters
        ----------
        use_sparse_matrix : bool
            Should sparse (CSR) adjacency matrix be used.
        """
        return OrderedDict([
            ('N', self.n_nodes),
            ('n_edges', self.n_edges),
            ('A', csr_matrix(self.A) if use_sparse_matrix else self.A),
            ('h', self.h),
            ('pa_exp', self.pa_exp),
            ('p_local', self.p_local),
            ('p_flow', self.p_flow),
            ('directed', self.directed),
            ('h0', self.hseries[0]),
            ('h1', self.hseries[-1]),
            ('threshold', self.threshold),
            ('converged', self.converged),
            ('check_convergence', self.check_convergence),
            ('alpha', self.alpha),
            ('alpha0', self.alpha0),
            ('alpha_adjustment_steps', self.alpha_adjustement_steps),
            ('alpha_adjustment_increment', self.alpha_adjustement_increment),
            ('alpha_adjusted', self.alpha_adjusted),
            ('n_iter', self.n_iter),
            ('n_steps', self.n_steps),
            ('runtime', self.runtime)
        ])

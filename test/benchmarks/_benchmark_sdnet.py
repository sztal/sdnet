"""Benchmarks for `sdnet` module."""
# pylint: disable=unused-import
import numpy as np
from sdnet import Segregation, SegregationClustering
from sdnet.networks import random_network, random_geometric_graph_nb
from sdnet.networks import generate_adjacency_matrix
from sdnet.utils import norm_manhattan_dist


K = 30
N = 4000
HM = 0.4
PA = 2

X = np.random.uniform(0, 1, (N, 2))
A = random_network(X.shape[0], k=K, directed=False)
P = random_geometric_graph_nb(X, norm_manhattan_dist, symmetric=True)

sp   = Segregation(A.copy(), P, homophily=HM)
spc  = SegregationClustering(A.copy(), P, homophily=HM, pa_exponent=PA)

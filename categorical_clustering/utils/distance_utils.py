import numpy as np
from numba import njit, jit


@njit
def squared_euclidean_distance(centroid, indices):
    result = 0
    for i in range(len(indices)):
        result += (1 - centroid[indices[i]]) ** 2
    return result


@njit
def hamming_distance(a, b):
    return np.sum(a != b)

#
# Created by Umanga Bista.
#

import numpy as np
from numpy.core.umath_tests import inner1d

class IllegalArgumentError(ValueError):
    pass

def norm(A):
    '''
        returns rowwise 2 norm of Array/Matrix numpy.ndarray
        input =>
            - A : data matrix (n * d dims) or d dim vector
        output: =>
            - norm: rowwise norm (n dim vector)
    '''
    return np.linalg.norm(A, ord = 2, axis = -1)

def dot(A, b):
    '''
        dot between each row of Matrix A and vector b
        input =>
            - A : n * d matrix
            - b: d dim vector
        output: =>
            - dot: n dim vector with each component representing rowwise dot product
                between matrix A and b
    '''
    # if A.ndim == 1 and b.ndim == 1:
    #     return A.dot(b)
    # elif A.ndim == 2 and b.ndim == 1:
    #     return np.einsum('ij,j', A, b) ## optimizations
    # elif A.ndim == 2 and b.ndim == 2:
    #     return np.einsum('ij,ij->i', A, b) ## optimizations
    # else:
    #     raise IllegalArgumentError("invalid dimensions of A or b")

    ## might be buggy, but optimized and would work in any of the above condition
    return inner1d(A, b)

def dist(A, b, method):
    '''
        distance between each row of Matrix A and vector b
        input =>
            - A : n * d matrix
            - b: d dim vector
        output: =>
            - dist: n dim vector with each component representing rowwise dist
                between matrix A and b
    '''
    if method == 'cosine':
        return 1. - (1.0 * dot(A, b)) / (norm(A) * norm(b))

    elif method == 'euclid':
        return norm(A-b)

    else:
        raise IllegalArgumentError('invalid method {}'.format(method))

def search(docs_vec, query_vec, dist_measure, k = 5):
    '''
        returns the most similar rows in docs_vec against query_vec
        input =>
            - docs_vec : n*d matrix
            - query_vec : d dim vector
            - dist_measure: distance measure `cosine` or `euclid`
            - k : number of most similar rows in docs_vec to return
        output: =>
            - idxs: indices of best matching rows
    '''
    distance = dist(docs_vec, query_vec, dist_measure)
    idxs = np.argsort(distance)[:k]
    return idxs

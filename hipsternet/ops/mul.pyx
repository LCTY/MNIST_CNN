import cython
# from cython.parallel import prange, parallel
from libc.math cimport floor
import numpy as np
cimport numpy as np
ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t
@cython.boundscheck(False)
def mul(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, int mul_bit, int adder_tree_length):
    cdef int row_len = a.shape[0]
    cdef int col_len = b.shape[1]
    cdef int row, col, inner
    print('row ={}\t col = {} ashape={} mul={}'.format(row_len,col_len,a.shape[1],row_len*col_len*a.shape[1]))
    cdef np.ndarray[DTYPE_t, ndim=1] adder_tree = np.zeros(adder_tree_length)
    cdef DTYPE_t tmp, add_tmp
    cdef DTYPE_t max_mul=0
    cdef DTYPE_t min_mul=0
    cdef DTYPE_t max_add=0
    cdef DTYPE_t min_add=0
    # cdef DTYPE_t adder_tree[adder_tree_length]
    cdef np.ndarray[DTYPE_t, ndim=2] product = np.zeros([row_len, col_len])

    for row in range(row_len):
        for col in range(col_len):
            for inner in range(a.shape[1]):
                tmp = a[row, inner] * b[inner, col]
                max_mul = max(tmp, max_mul)
                min_mul = min(tmp, min_mul)
                tmp = floor(tmp * (2**mul_bit)) / (2**mul_bit)
                for i in range(adder_tree_length):
                    if adder_tree[i] != 0:
                        tmp += adder_tree[i]
                        max_add = max(tmp, max_add)
                        min_add = min(tmp, min_add)
                        adder_tree[i] = 0
                    else:
                        adder_tree[i]=tmp
                        break
            tmp = 0
            for i in range(adder_tree_length):
                if adder_tree[i] != 0:
                    tmp += adder_tree[i]
                    max_add = max(tmp, max_add)
                    min_add = min(tmp, min_add)
                    adder_tree[i] = 0
            product[row, col] = tmp

    return product,max_mul,min_mul,max_add,min_add
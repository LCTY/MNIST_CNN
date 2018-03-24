import cython
from cython.parallel import prange, parallel
from libc.math cimport floor
import numpy as np
cimport numpy as np
ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t
@cython.boundscheck(False)
def mul(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, int add_bit, int mul_bit):
    cdef int row_len = a.shape[0]
    cdef int col_len = b.shape[1]
    cdef int row, col, inner
    # cdef long long int max_mul,min_mul
    # cdef long long int max_add,min_add
    # cdef DTYPE_t ma = 0, mi = 10
    print('row ={}\t col = {} ashape={} mul={}'.format(row_len,col_len,a.shape[1],row_len*col_len*a.shape[1]))
    cdef DTYPE_t tmp, add_tmp
    cdef DTYPE_t max_mul=0
    cdef DTYPE_t min_mul=0
    cdef DTYPE_t max_add=0
    cdef DTYPE_t min_add=0
    cdef np.ndarray[DTYPE_t, ndim=2] product = np.zeros([row_len, col_len])
    with nogil:
        for row in prange(row_len, schedule='static'):
            for col in range(col_len):
                for inner in range(a.shape[1]):
                    tmp = a[row, inner] * b[inner, col]
                    max_mul = max(tmp, max_mul)
                    min_mul = min(tmp, min_mul)
                    add_tmp = product[row, col]+ floor(tmp * (2**mul_bit)) / (2**mul_bit)
                    product[row, col] = floor(add_tmp * (2**add_bit)) / (2**add_bit)
                    max_add = max(product[row, col], max_add)
                    min_add = min(product[row, col], min_add)

    return product,max_mul,min_mul,max_add,min_add
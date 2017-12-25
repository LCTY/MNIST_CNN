from hipsternet.im2col import *
import numpy as np
a = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
orig = im2col_indices(a,3,3)
print(orig)
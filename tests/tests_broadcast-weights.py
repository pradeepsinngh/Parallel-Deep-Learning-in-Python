#!/usr/bin/python
# Bcast will not check memory ordering of the buffer.  If sending buffer
# is arranged in Fortran-contiguous order, but receiving buffers are in
# C-contiguous order, the values in receiving buffers will be different
# from sending buffer's values.
#
# Note:
# 1. Concatenate a Fortran-order matrix will get a C-order vector.
# 2. ascontiguousarray and asfortranarray can convert a matrix between
#    Fortran-order and C-order.
# 3. array-to-array operations will create a C-order array, even the two
#    arrays are in Fortran-order.
# 4. array-to-scalar operations will create an array with original order.

import numpy as np
import scipy.io as sio

from mpi4py import MPI


def load_weights():
    '''
    >>> theta1, theta2 = load_weights()

    >>> theta1.shape
    (25, 401)

    >>> theta2.shape
    (10, 26)

    #>>> theta1.sum()
    #9.242643928120069

    #>>> theta2.sum()
    #-100.08344384930399
    '''
    weights = sio.loadmat('weights.mat') 
    return (weights['Theta1'].astype('float32'), weights['Theta2'].astype('float32'))


comm = MPI.COMM_WORLD


# Case1: the theta1 matrices contain different values, albeit the
# summations are the same.
if comm.rank == 0:
    theta1 = load_weights()[0]  # flag F_CONTIGUOUS is true
    #theta1 = np.concatenate(theta1)
    #theta1 = np.require(theta1[0], requirements=['F'])

    #print('theta1.flags: {}'.format(theta1.flags))
    #print('theta1.dtype: {}'.format(theta1.dtype))
else:
    theta1 = np.require(np.zeros((25,401), dtype='float32'))  # flag C_CONTIGUOUS is true
    #theta1 = np.require(np.zeros((25,401), dtype='float32'), requirements=['F'])
    #theta1 = np.zeros(401, dtype='float32')

print('[{0}] theta1 is aligned in Fortran order (before Bcast): {1}'.format(comm.rank, np.isfortran(theta1)))
comm.Bcast([theta1, MPI.FLOAT], root=0)
print('[{0}] theta1 is aligned in Fortran order (after Bcast): {1}'.format(comm.rank, np.isfortran(theta1)))
comm.Barrier()
print('[{0}] theta1: {1}'.format(comm.rank, theta1))


# Case2: Sending buffer is arranged from Fortran-order to C-order in
#  advance, receive buffer is arranged in C-order.
if comm.rank == 0:
    theta1 = load_weights()[0]  # flag F_CONTIGUOUS is true
    theta1 = np.ascontiguousarray((theta1))
else:
    theta1 = np.require(np.zeros((25,401), dtype='float32'))  # flag C_CONTIGUOUS is true
print('[{0}] theta1 is aligned in Fortran order (before Bcast): {1}'.format(comm.rank, np.isfortran(theta1)))
comm.Bcast([theta1, MPI.FLOAT], root=0)
#theta1 = theta1.reshape(25, 401)
print('[{0}] theta1 is aligned in Fortran order (after Bcast): {1}'.format(comm.rank, np.isfortran(theta1)))
comm.Barrier()
print('[{0}] theta1: {1}'.format(comm.rank, theta1))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

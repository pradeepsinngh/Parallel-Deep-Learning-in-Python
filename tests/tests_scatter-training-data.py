#!/usr/bin/python

"""
>>> 10 + 20
30
"""

import numpy as np
import scipy.io as sio

from mpi4py import MPI


def hello(msg):
    ''' hello
    >>> hello('world')
    hello world
    '''
    print('hello ' + msg)


def load_inputs():
    '''
    >>> inputs = load_inputs()

    Size of dataset:
    >>> inputs.shape
    (5000, 400)

    Sum of entire dataset:
    >>> inputs.sum()
    262678.26015968173

    Sum of 1st half dataset:
    >>> inputs[:len(inputs)/2].sum()
    132986.76649618745

    Sum of 2nd half dataset:
    >>> inputs[len(inputs)/2:].sum()
    129691.49366349436

    '''
    return sio.loadmat('data.mat')['X']


comm = MPI.COMM_WORLD

#inputs = load_inputs()
inputs = np.asarray(range(20))
send_buf = np.empty((comm.size, 10))
send_buf[0] = inputs[:10]
send_buf[1] = inputs[10:]
recv_buf = np.zeros(10)
#recv_buf = np.zeros((inputs.shape[0]/comm.size, inputs.shape[1]))
print('recv_buf shape: {}'.format(recv_buf.shape))

comm.Scatter(send_buf, recv_buf)
print('[{0}] recv_buf: {1}'.format(comm.rank, recv_buf))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

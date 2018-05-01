import numpy as np
import theano
import theano.tensor as T
import time


x = T.matrix('x')
y = T.matrix('y')
z = T.dot(x, y)
gpu_matrix_dot = theano.function([x, y], z, allow_input_downcast=True)


def matrix_dot(a, b, mode='cpu'):
    assert(mode == 'cpu' or mode == 'gpu')
    if mode == 'cpu':
        dot_func = np.dot
    elif mode == 'gpu':
        dot_func = gpu_matrix_dot
    return dot_func(a, b)


def benchmark_matrix_dot(d1, d2, d3, iteration, mode):
    t0 = time.time()
    for i in xrange(iteration):
        m1 = np.random.rand(d1, d2)
        m2 = np.random.rand(d2, d3)
        m3 = matrix_dot(m1, m2, mode)
    t1 = time.time()
    print('{5} time ({0},{1},{2}) {3} iterations: {4} secs'.format(
        d1, d2, d3, iteration, t1 - t0, mode)
    )
    return t1 - t0


# If the matrices are too small (400,20 x 20,1), GPU is slower than CPU
#benchmark_matrix_dot(400, 20, 1, 100, 'cpu')
#benchmark_matrix_dot(400, 20, 1, 100, 'gpu')

# If the matrices are large (5000,400 x 400,20), GPU is far fast than CPU
#benchmark_matrix_dot(5000, 400, 20, 100, 'cpu')
#benchmark_matrix_dot(5000, 400, 20, 100, 'gpu')


size = np.asarray([100, 10, 1])
sizes = []
cpu_times = []
gpu_times = []
#for scalar in xrange(0, 101, 10):
for scalar in xrange(0, 21):
    m, n, o = size + (500*scalar, 50*scalar, 5*scalar)
    print('size ({0}, {1}, {2})'.format(m, n, o))
    cpu_time = benchmark_matrix_dot(m, n, o, 100, 'cpu')
    gpu_time = benchmark_matrix_dot(m, n, o, 100, 'gpu')
    print('')

    sizes.append((m, n, o))
    cpu_times.append(cpu_time)
    gpu_times.append(gpu_time)

print('sizes = {}'.format(sizes))
print('cpu_times = {}'.format(cpu_times))
print('gpu_times = {}'.format(gpu_times))

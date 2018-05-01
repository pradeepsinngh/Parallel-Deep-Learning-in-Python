# Profiling GPU computing example

CPU mode

    ...

Most time will be spent in numpy.core.multiarray.dot

GPU mode

    $ python -m cProfile -o gpu.log <>
    []: import pstats
    []: p = pstats.Stats('gpu.log')
    []: p.sort_stats('tottime').print_stats(10)

Most time will be spent in Theano's compiled function.


The matrix dot will be called around 1.2M times.


# Tips

## Numpy object alignment issue

If Theano complains the numpy object is not aligned, you can use `<obj>.flags`
to check the alignment ([discussion](https://groups.google.com/forum/#!topic/theano-users/HocacZSNafg)).

I personally meet this issue when using `scipy.io.loadmat()`, the numpy array
objects are not aligned.

## If Nvidia GPU only supports float32

If you are using Theano to manipulate float64 data, but the Nvidia GPU only
supports float32, you can indicate the parameter `allow_input_downcast=True`
in `theano.function()` when building the expression.


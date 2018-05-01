# Introduction

Simple examples demonstrating techniques of deep learning model manipulation.

 * `mnist-nn.py`: 3-layer neural network as digit recognizer (MNIST).
 * `mnist-nn-gpu.py`: mnist-nn with GPU computing.
 * `mnist-nn-data-parallelism.py`: mnist-nn with data parallelism (MPI and GPU).

Job file is the wrapper of a MPI command.  To distribute a program via
MPI, you can indicate the configurations of worknodes (address or
hostname, processor numbers, etc.) in a hostfile or CLI.  For more
details about hostfile, you can refer to [Open MPI's Q&A](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile).

# Setup development environment

Please refer to <tech blog post>.

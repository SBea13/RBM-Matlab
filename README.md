# RBM-Matlab
A simple implementation of a Restricted Boltzmann Machine, able to perfrom a supervised classification task on the MNIST database of handwritten digits.

The core file `my_RBM.m` train a RBM with easily customizable parameters, and produces:
- a training error vs. epoch plot
- two confusion matrices for test and training accuracy
- a weights grey-scale visual representation

It also store in the `t_end` variable the training time.

It requires the support files `compute_gradient.m`, `rbm_CD_k.m`, `training.m`, which includes the main functions for the training process.

In `test_performance.m`, a big nested loop is coded to try different combinations of hyper-parameters.

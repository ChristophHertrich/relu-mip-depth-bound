# Towards Lower Bounds on the Depth of ReLU Neural Networks

This repository contains code for a mixed-integer program (MIP) proving that no (so-called) H-conform 3-layer neural network can precisely compute the maximum of five numbers. It basicly contains two files:

1. `max5num.sage` is a SageMath script that generates the MIP and solves it with the Parma Polyhedral Library using exact rational arithmetics.
2. `mip.mps` contains the MIP in the standardized `.mps` format, making it possible to solve the MIP with any solver of choice.

For details, we refer to our paper (... link to be inserted ...)

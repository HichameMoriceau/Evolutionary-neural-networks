# Evolutionary Neural Networks

Neural Networks are a very popular technique for supervised learning challenges. This project focuses on the automation of the search for the most adequate neural net architecture and weights for any given use-case. 

Given a maximum size neural network topology (architecture) the population based optimization algorithm(s) autonomously find an appropriate topology and set of weights. The algorithm was tested on 3 data sets: Breast Cancer Malignant (Diagnostic), Breast Cancer Recurrence and Haberman's survival test.

This work contains implementations of the following techniques:
 - Vectorized Neural Network of any topology (using Linear Algebra)
 - Differential Evolution
 - Particle Swarm Optimization
 - Neural Network Ensemble (The population of neural networks vote to make a prediction)
 - N-Fold Cross Validation Method

The program leverages the [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library and each replicate of the experiment is ran as an [OpenMP](http://openmp.org/wp/) thread.



## Training Algorithms

For anyone interested in implementing a highly reliable and versatile optimization algorithm I would recommend taking a peek at Differential Evolution first since it is simple and powerful (near state-of-the-art results). Both techniques are constructive stochastic algorithms and therefore do not guarantee that the best solution will be found. As opposed to gradient-based techniques (e.g. Back Propagation) they are inherently better at global search and tend to more rarely fall into local optimas. 

Although more work could be done to make the algorithms more efficient, here are the current implementations.

Optimization algorithm 1: [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution):
Optimization algorithm 2: [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization):


## Documentation

Please refer to `/dissertation/memoir.pdf` for more details as well as the results and conclusions of the experiment.
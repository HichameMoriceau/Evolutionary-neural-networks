# Evolutionary Neural Networks 
###### [Pursuing the work done for my bachelor's dissertation over the course of a research internship]


Neural Networks are a very popular technique for supervised learning challenges. This project focuses on the automation of the search for the most adequate neural net architecture and weights for any given use-case. 

Given a maximum size neural network topology (architecture) the population based optimization algorithm(s) autonomously find an appropriate topology and set of weights. The algorithm was tested on 3 data sets: Breast Cancer Malignant (Diagnostic), Breast Cancer Recurrence and Haberman's survival test.

This work contains implementations of the following techniques:
 - [Vectorized Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) of any topology (using Linear Algebra)
 - [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution)
 - [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
 - [Neural Network Ensemble](http://www.sciencedirect.com/science/article/pii/S000437020200190X)
 - [K-Fold Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation)

The program leverages the [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library and each replicate of the experiment is ran as an [OpenMP](http://openmp.org/wp/) thread.


## Training Algorithms

For anyone interested in implementing a highly reliable and versatile optimization algorithm I would recommend taking a peek at Differential Evolution first since it is simple and powerful (results are comparable to the state-of-the-art). Both techniques are constructive stochastic algorithms and therefore do not guarantee that the best solution will be found. As opposed to gradient-based techniques (e.g. Back Propagation) they are inherently better at global search and tend to more rarely fall into local optimas. 


## Documentation

Please refer to `/dissertation/memoir.pdf` for more details as well as the results and conclusions of the experiment.
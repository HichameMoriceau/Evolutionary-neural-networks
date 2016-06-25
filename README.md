# Evolutionary Neural Networks 
###### [Pursuing the work done for my bachelor's dissertation over the course of a research internship]


Neural Networks are probably the most used model class for solving supervised learning problems. This project focuses on the automation of the design of the most adequate architecture and weights for solving any classification problem (PSO, DE).

This work contains implementations of the following techniques:
 - [Vectorized Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) of any topology (using Linear Algebra)
 - [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) (DE)
 - [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO)
 - [Artificial Immune System: Clonal Selection](https://en.wikipedia.org/wiki/Artificial_immune_system) (AIS)
 - [Training, Validation and Test data subsets](https://class.coursera.org/ml-005/lecture/61)
 - [F1 score](https://en.wikipedia.org/wiki/F1_score) measure of prediction ability, [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) and %accuracy
 - Basic statistics (mean, variance) => (an implementation of the [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) method can be found but wasn't used for the experiment)
 - [Neural Network Ensemble](http://www.sciencedirect.com/science/article/pii/S000437020200190X) (*commented for later study*)

## Used libraries
The application leverages the following libraries:
 - [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library 
 - [FANN C++](http://leenissen.dk/fann/wp/) library (implements the Gradient Descent/Back Propagation algorithm)
 - [NEAT C++](http://nn.cs.utexas.edu/?neat-c) (Neuro Evolution of Augmenting Topologies => adapted to solve classification problems)

## Performance considerations
For improved performances, some tasks are ran *concurrently*: Each replicate of the experiment is ran as an [OpenMP](http://openmp.org/wp/) thread.


## Documentation

My bachelor's dissertation is accessible at `/dissertation/memoir.pdf` if you want to find out more on the theory/background of neural networks, evolutionary algorithms and see the results of the initial experiment.
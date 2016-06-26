# Automated Optimization of Neural Network Architecture Design 
###### [Pursuing the work done for my bachelor's dissertation over the course of a research internship]

In Machine Learning, Neural network have demonstrated flexibility and robustness properties. It is known that neural nets can be used for solving a wide variety of problems, provided that the topology is appropriately chosen. There are two main schools of thought when it comes to training neural networks: the use of gradient based methods with the *back propagation algorithm* and the use of *evolutionary algorithms*. This research project researches the automation of the design of the most adequate architecture and weights for solving various supervised learning problem.

The platform is composed of 3 benchmarks:
 - The BP experiment
 - The NEAT experiment
 - The Evolutionary nets experiment
The benchmark can be ran as a whole (see `run_all_benchmarks.sh`) or independently (see each `run_experiment.sh` script). See the below for more information on the *libraries* and *acronyms* used here.

Algorithms:
 - [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) (referred to as: *DE*)
 - [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (referred to as: *PSO*)
 - [Artificial Immune System: Clonal Selection](https://en.wikipedia.org/wiki/Artificial_immune_system) (referred to as: *AIS*)
 - [Neuro Evolution of Augmenting Topologies](http://nn.cs.utexas.edu/?neat-c) (referred to as: *NEAT*)
 - [Gradient Descent with Back Propagation](http://neuralnetworksanddeeplearning.com/chap2.html) (referred to as: *BP*)

This work also contains implementations of the following techniques:
 - [Vectorized Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) of any topology (using Linear Algebra)
 - [Training, Validation and Test data subsets](https://class.coursera.org/ml-005/lecture/61)
 - [F1 score](https://en.wikipedia.org/wiki/F1_score) measure of prediction ability, [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) and %accuracy
 - [Neural Network Ensemble](http://www.sciencedirect.com/science/article/pii/S000437020200190X) (*commented for later study*)
 - Basic statistics (mean, variance) => (an implementation of the [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) method can be found but wasn't used for the experiment)

## Used libraries
The application leverages the following libraries:
 - [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library 
 - [FANN C++](http://leenissen.dk/fann/wp/) library (implements the Gradient Descent/Back Propagation algorithm)
 - [NEAT C++](http://nn.cs.utexas.edu/?neat-c) (Neuro Evolution of Augmenting Topologies => adapted to solve classification problems)

## Performance considerations
For improved performances, each replicate of the experiment is ran *concurrently* as an [OpenMP](http://openmp.org/wp/) thread.

## Documentation

My bachelor's dissertation is accessible at `/dissertation/memoir.pdf` if you want to find out more on the theory/background of neural networks, evolutionary algorithms and see the results of the initial experiment (+ paper currently being written).
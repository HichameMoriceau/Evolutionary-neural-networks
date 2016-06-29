# Automated Optimization of Neural Network Architecture Design 
###### [Pursuing the work done for my bachelor's dissertation over the course of a research internship]
###### *Currently under development*

In Machine Learning, Neural network have demonstrated flexibility and robustness properties. It is known that neural nets can be used for solving a wide variety of problems, provided that the topology is appropriately chosen. There are two main schools of thought when it comes to training neural networks: the use of gradient based methods with the *back propagation algorithm* and the use of *evolutionary algorithms*. This research project researches the automation of the design of the most adequate architecture and weights for solving various supervised learning problem.

## Overview
This CLI tool is composed of 3 benchmarks and some additional directories:
 - The `BP_experiment` directory contains the BP benchmark (using the FANN library)
 - The `NEAT_experiment` directory contains the NEAT benchmark (using NEAT library)
 - The `evolutionary_nets` directory contains the evolutionary nets benchmark (PSO, DE & AIS)
 - The `formatting_scripts` directory contains C++ scripts to perform CSV to FANN & FANN to CSV data set conversion. (see section on *Adding more data sets*)
 - The `data` directory contains the data sets to be used for the experiment. It is also in this directory that results are written. 

The benchmark can be ran as a whole using `run_all_benchmarks.sh`. It is also possible to run each benchmark independently using the `run_experiment.sh` script of each experiment's directory. See below for more information on the *libraries* and *main acronyms* used in this project.

Algorithms:
 - [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) (referred to as: *DE*)
 - [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (referred to as: *PSO*)
 - [Artificial Immune System: Clonal Selection](https://en.wikipedia.org/wiki/Artificial_immune_system) (referred to as: *AIS*)
 - [Neuro Evolution of Augmenting Topologies](http://nn.cs.utexas.edu/?neat-c) (referred to as: *NEAT*)
 - [Gradient Descent with Back Propagation](http://neuralnetworksanddeeplearning.com/chap2.html) (referred to as: *BP*)

This work also contains implementations of the following techniques:
 - [Vectorized Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) of any topology (using Linear Algebra)
 - [Segmentation of data set into Training, Validation and Test data subsets](https://class.coursera.org/ml-005/lecture/61)
 - [F1 score](https://en.wikipedia.org/wiki/F1_score), [MSE](https://en.wikipedia.org/wiki/Mean_squared_error), %accuracy
 - [Neural Network Ensemble](http://www.sciencedirect.com/science/article/pii/S000437020200190X) (*commented for later study*)
 - Basic statistics (mean, variance, etc.) => (an implementation of the [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) method can be found but isn't currentl used for the experiment)

## Installation
The application leverages the following libraries:

 - [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library 
 - [FANN C++](http://leenissen.dk/fann/wp/) library (implements the Gradient Descent/Back Propagation algorithm)
 - [NEAT C++](http://nn.cs.utexas.edu/?neat-c) (Neuro Evolution of Augmenting Topologies => adapted to solve classification problems)

Simply run:

`$ sudo apt-get install libarmadillo-dev libfann-dev octave`. 

Octave is optional but allows you to generate plots by running pre-written scripts such as `$ ./plot_all_results.sh`, which generates plots such as the MSE, F1 score and %accuracy against the number of calls made to the error function and so on (with error bars).

## Running the benchmark

```
$ # Download the repository:
$ git clone https://github.com/HichameMoriceau/Evolutionary-neural-networks.git
$ cd Evolutionary-neural-networks/
$ # give execution permissions
$ chmod +x run_all_benchmarks.sh
$ # Execute benchmark (all 5 algorithms on all data sets)
$ ./run_all_benchmarks.sh
```

Once you're all set, you might be interested in modifying the hard coded parameters (Number of replicates, population size etc.) in the `run_all_benchmarks.sh` script.

#### Deleting the benchmark

```
rm -rf Evolutionary-neural-networks/
sudo apt-get remove libarmadillo-dev libfann-dev octave
```

## Adding more data sets

#### Before adding your data set

Please make sure that the data set only contains *numerical* values (you might want to do some pre-processing using a tool such as [OpenRefine](http://openrefine.org/)). The target attribute *must* be the last column of the data set. You'll see post transformation, I typically call these <name>+"-transformed.csv". 

Feature scaling will then be automatically applied when the benchmark loads the data set. The benchmark support classification problem with any number of attributes or prediction classes (2 or more).

#### Adding your data set

 - Add your data set in the `data` directory 
 - In `data`, create a directory named after your data set following my convention ('-' must be replaced by "_", directory name must end with "_results")
 - Make sure that `BP_experiment/data/` contains data sets in the FANN format and with a `.data` extension.
 - Create a genome file required for NEAT to run (same convention except the filename must end with "startgenes"), look at the deprecated but insightful `NEATDOC.ps` documentation for how to write these. (It defines the initial topology to be evolved).
 - Add its path as CLI argument within the `run_all_benchmarks.sh` script (always using a .csv extension).


#### Converting your data set in a FANN readable format

In the `formatting_scripts` directory you'll find C++ scripts to help you convert your .CSV data set into a .DATA format that the FANN library used in the `BP_experiment` will be able to use.



## Compilation and execution

If you wish to make changes to a benchmark or simply to manually run any C++ code here, you'll be able to find the compilation and execution commands by running the following commands. Feel free to look at the `run_experiment.sh` scripts to see how each experiment is ran.

```
cat main.cpp | grep "Compile"
cat main.cpp | grep "Run"
```

#### For NEAT

Run `make` within the `NEAT_experiment` directory. The code used here is the original NEAT C++ benchmark application and comes with a Makefile.

#### For evolutionary_nets

`evolutionary_nets` is a QT Creator project. Either build it from the IDE or follow [these instructions](http://stackoverflow.com/questions/19206462/compile-a-qt-project-from-command-line) to build it from CLI.

### Performance considerations

For improved performances, each replicate of the experiment is ran *concurrently* as an [OpenMP](http://openmp.org/wp/) thread.

## Documentation

My bachelor's dissertation is accessible at `/dissertation/memoir.pdf` if you want to find out more on the theory/background of neural networks, evolutionary algorithms and see the results of the initial experiment (+ paper currently being written).
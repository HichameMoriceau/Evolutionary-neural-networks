#!/bin/bash


# This script executes the following algorithms:
# - Gradient Descent with Back Propagation
# - Neuro Evolution of Augmenting Topologies (NEAT)
# - Differential Evolution (DE)
# - Particle Swarm Optimization (PSO)
# - Clonal Selection: Artificial Immune System (AIS)

# It requires Execution and Reading permission:
# $ chmod o+x run_all_benchmarks.sh

NB_REPS=1
NB_GENS=100 # to be removed asap
NB_ERR_FUNC_CALLS=10100
POP_SIZE=100


# BP-CASCADE TOPOLOGY EXPERIMENTS
./BP_experiment/runme 4 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 5 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 6 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 7 $NB_REPS $NB_ERR_FUNC_CALLS

# Neuro Evolution of Augmenting Topologies
./NEAT_experiment/neat test.ne 0 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE 
./NEAT_experiment/neat test.ne 1 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE
./NEAT_experiment/neat test.ne 2 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE
#./NEAT_experiment/neat test.ne 3 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE

# Differential Evolution, Particle Swarm Optimization and Clonal Selection (AIS)
./evolutionary_nets/build-evolutionary_nets-Desktop-Debug/evolutionary_nets data/breast-cancer-malignantOrBenign-data-transformed.csv \
    data/breast-cancer-recurrence-data-transformed.csv \
    data/iris-data-transformed.csv \
    data/wine-data-transformed.csv \
    $NB_REPS \
    $NB_GENS \
    $POP_SIZE


# Please be patient! Running each benchmark can take some time! 
# (Even though each replicate run on a different omp thread)

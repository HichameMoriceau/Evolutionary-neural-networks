#!/bin/bash

# This script executes the following algorithms:
# - Gradient Descent with Back Propagation
# - Neuro Evolution of Augmenting Topologies (NEAT)
# - Differential Evolution (DE)
# - Particle Swarm Optimization (PSO)
# - Clonal Selection: Artificial Immune System (AIS)

# It requires Execution and Reading permission:
# $ chmod o+x run_all_benchmarks.sh

# Please be patient! Running each benchmark can take some time! 

NB_REPS=20
NB_ERR_FUNC_CALLS=150
POP_SIZE=100

# BP-CASCADE TOPOLOGY EXPERIMENTS
./BP_experiment/runme 0 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 1 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 2 $NB_REPS $NB_ERR_FUNC_CALLS
./BP_experiment/runme 3 $NB_REPS $NB_ERR_FUNC_CALLS

# Neuro Evolution of Augmenting Topologies
./NEAT_experiment/neat NEAT_experiment/test.ne 0 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE 
./NEAT_experiment/neat NEAT_experiment/test.ne 1 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE
./NEAT_experiment/neat NEAT_experiment/test.ne 2 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE
./NEAT_experiment/neat NEAT_experiment/test.ne 3 $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE

# Differential Evolution, Particle Swarm Optimization and Clonal Selection (AIS)
./evolutionary_nets/build-evolutionary_nets-Desktop-Debug/evolutionary_nets data/breast-cancer-malignantOrBenign-data-transformed.csv \
    data/breast-cancer-recurrence-data-transformed.csv \
    data/iris-data-transformed.csv \
    data/wine-data-transformed.csv \
    $NB_REPS \
    $NB_ERR_FUNC_CALLS \
    $POP_SIZE

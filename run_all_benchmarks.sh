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
DATA_SETS="data/breast-cancer-malignantOrBenign-data-transformed.csv data/breast-cancer-recurrence-data-transformed.csv data/iris-data-transformed.csv data/wine-data-transformed.csv"

NB_REPS=1
NB_ERR_FUNC_CALLS=150
POP_SIZE=100

# BP-CASCADE TOPOLOGY EXPERIMENTS
./BP_experiment/runme $DATA_SETS $NB_REPS $NB_ERR_FUNC_CALLS

# Neuro Evolution of Augmenting Topologies
./NEAT_experiment/neat $DATA_SETS NEAT_experiment/test.ne $NB_REPS $NB_ERR_FUNC_CALLS $POP_SIZE 

# Differential Evolution, Particle Swarm Optimization and Clonal Selection (AIS)
./evolutionary_nets/build-evolutionary_nets-Desktop-Debug/evolutionary_nets data/breast-cancer-malignantOrBenign-data-transformed.csv \
    data/breast-cancer-recurrence-data-transformed.csv \
    data/iris-data-transformed.csv \
    data/wine-data-transformed.csv \
    $NB_REPS \
    $NB_ERR_FUNC_CALLS \
    $POP_SIZE

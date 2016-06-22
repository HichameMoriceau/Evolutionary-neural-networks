#!/bin/bash

# This script requires Execution and Reading permission:
# $ chmod o+x run_benchmark.sh

# Arguments: 
# 1) paths to all data sets
# 2) NB replicates
# 3) NB generations
# 4) Population size

./evolutionary_nets data/breast-cancer-malignantOrBenign-data-transformed.csv \
    data/breast-cancer-recurrence-data-transformed.csv \
    data/iris-data-transformed.csv \
    data/wine-data-transformed.csv \
    10 \
    100 \
    100

# Please note that running each algorithm:
# - Differential Evolution (DE)
# - Particle Swarm Optimization (PSO)
# - Artificial Immune System (AIS)
# Can take some time! (Even though each replicate run on a different omp thread)

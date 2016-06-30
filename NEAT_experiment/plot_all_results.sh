#!/bin/bash

./plotter.m data/wine_data_transformed_results/NEAT-results.mat NEAT BCM
./plotter.m data/iris_data_transformed_results/NEAT-results.mat NEAT IRIS
./plotter.m data/breast_cancer_malignantOrBenign_data_transformed_results/NEAT-results.mat NEAT WINE
./plotter.m data/breast_cancer_recurrence_data_transformed_results/NEAT-results.mat NEAT BCR

#!/bin/bash

./plotter.m data/wine_data_transformed_results/DE-results.mat DE Wine
./plotter.m data/iris_data_transformed_results/DE-results.mat DE Iris
./plotter.m data/breast_cancer_malignantOrBenign_data_transformed_results/DE-results.mat DE BCM
./plotter.m data/breast_cancer_recurrence_data_transformed_results/DE-results.mat DE BCR

./plotter.m data/wine_data_transformed_results/PSO-results.mat PSO Wine
./plotter.m data/iris_data_transformed_results/PSO-results.mat PSO Iris
./plotter.m data/breast_cancer_malignantOrBenign_data_transformed_results/PSO-results.mat PSO BCM
./plotter.m data/breast_cancer_recurrence_data_transformed_results/PSO-results.mat PSO BCR

./plotter.m data/wine_data_transformed_results/AIS-results.mat AIS Wine
./plotter.m data/iris_data_transformed_results/AIS-results.mat AIS Iris
./plotter.m data/breast_cancer_malignantOrBenign_data_transformed_results/AIS-results.mat AIS BCM
./plotter.m data/breast_cancer_recurrence_data_transformed_results/AIS-results.mat AIS BCR
